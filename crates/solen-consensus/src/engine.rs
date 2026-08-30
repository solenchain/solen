//! Core consensus engine.
//!
//! Implements a simplified Tendermint-style BFT protocol:
//! - Round-robin block proposers
//! - 2/3+ stake-weighted attestation quorum for finality
//! - Epoch-based reward distribution
//! - Slashing for double-sign and downtime

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use solen_crypto::blake3_hash;
use solen_execution::executor::BlockExecutor;
use solen_execution::proof::ProofVerifierRegistry;
use solen_execution::receipt::BlockResult;
use solen_intents::pool::IntentPool;
use solen_intents::solver::{DirectTransferSolver, IntentSolver};
use solen_storage::StateStore;
use solen_types::block::BlockHeader;
use solen_types::transaction::UserOperation;
use solen_types::{BlockHeight, Hash, ValidatorId};
use thiserror::Error;
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};

use crate::epoch::EpochManager;
use crate::events::NodeEvent;
use crate::mempool::Mempool;
use crate::validator::{ValidatorInfo, ValidatorSet};

#[derive(Debug, Error)]
pub enum ConsensusError {
    #[error("engine already running")]
    AlreadyRunning,
    #[error("engine stopped")]
    Stopped,
    #[error("not the proposer for this height")]
    NotProposer,
}

/// Configuration for the consensus engine.
#[derive(Clone)]
pub struct EngineConfig {
    pub block_time_ms: u64,
    pub max_ops_per_block: usize,
    pub validator_id: ValidatorId,
    pub chain_id: u64,
    /// Prune mode: delete blocks older than retention window to save disk.
    /// Default is false (archive mode — keep all history).
    pub prune: bool,
    /// Activation height for attestation-aware fork choice ("fork choice v2").
    /// Below this height the engine uses the legacy single-pending, attest-once
    /// rule. At/above it, the engine tracks competing blocks + per-validator
    /// votes (vote-change allowed), converges all nodes onto the deterministic
    /// vote leader, and finalizes whichever block hash reaches 2/3 — fixing the
    /// 2-down competing-block liveness deadlock (mainnet halt 2026-06-26).
    /// Defaults to u64::MAX = OFF, so a deployed binary is byte-for-byte
    /// behaviourally identical to the old one until an activation height is set
    /// (flag-day: deploy dormant everywhere, then activate at one height).
    pub fork_choice_v2_height: u64,
    /// Activation height for the supply-conserving fee settlement + max_fee cap
    /// (C-01 / H-01). Below it the executor applies the legacy settlement
    /// byte-for-byte; at/above it the treasury is credited only what the sender
    /// was actually debited and the fee is capped at the signed max_fee. Like
    /// fork_choice_v2_height this is CONSENSUS-AFFECTING, so it defaults to
    /// u64::MAX = OFF (deploy dormant everywhere, then activate at one height).
    pub fee_fix_height: u64,
    /// Authenticate blocks received over the sync path (C-02). When true,
    /// `replay_synced_block` rejects any synced block whose header is not signed
    /// by an active validator (proposer in the set + valid proposer signature),
    /// closing the block-forgery / fake-chain-injection / eclipse vector where an
    /// unsolicited SyncBlocks could inject attacker-crafted blocks. Honest sync
    /// is unaffected (real blocks carry valid proposer signatures). Defaults to
    /// false so it deploys dormant and is activated fleet-wide deliberately
    /// (instantly reversible by restarting without the flag).
    pub authenticate_sync_blocks: bool,
    /// Activation height for the WASM determinism hardening (C-04 bounded linear
    /// memory + C-05 deterministic relaxed-SIMD). Below it the VM uses the legacy
    /// config byte-for-byte; at/above it the strict config applies. Like the
    /// other height gates this is CONSENSUS-AFFECTING (changes execution results
    /// for memory-growing / relaxed-SIMD contracts), so it defaults to u64::MAX =
    /// OFF (deploy dormant everywhere, then activate at one height).
    pub determinism_fix_height: u64,
    /// Activation height for post-quantum (ML-DSA-65) account auth
    /// (`AuthMethod::MlDsa`). CONSENSUS-AFFECTING (changes which ops are valid),
    /// so defaults to u64::MAX = OFF — ships dormant, activate at one height.
    pub pq_auth_height: u64,
    /// Activation height for the grind-resistant epoch-randomness accumulator
    /// (variant B). Below it the proposer seed is `blake3(state_root)` of the
    /// boundary block (the shipped fix); at/above it consensus reads the seed the
    /// accumulator committed to state, and the executor writes the accumulator
    /// each block. CONSENSUS-AFFECTING (adds a state write + changes the proposer
    /// schedule at activation), so defaults to u64::MAX = OFF — ships dormant,
    /// activate at one height (flag-day).
    pub epoch_randomness_height: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            block_time_ms: 2000,
            max_ops_per_block: 5000,
            validator_id: [0u8; 32],
            chain_id: 0,
            prune: false,
            fork_choice_v2_height: u64::MAX,
            fee_fix_height: u64::MAX,
            authenticate_sync_blocks: false,
            determinism_fix_height: u64::MAX,
            pq_auth_height: u64::MAX,
            epoch_randomness_height: u64::MAX,
        }
    }
}

/// A finalized block with header, execution result, and attestations.
#[derive(Debug, Clone)]
pub struct FinalizedBlock {
    pub header: BlockHeader,
    pub result: BlockResult,
    pub attestations: Vec<Attestation>,
    /// Original operations, stored for sync replay.
    pub operations: Vec<UserOperation>,
}

/// A validator's attestation of a block.
#[derive(Debug, Clone)]
pub struct Attestation {
    pub validator_id: ValidatorId,
    pub block_height: u64,
    pub block_hash: Hash,
}

/// Result of producing a block.
pub struct ProducedBlock {
    /// The finalized block (only set in single-validator mode).
    pub finalized: Option<FinalizedBlock>,
    /// The block header (always set).
    pub header: BlockHeader,
    /// The operations included in the block (for broadcasting to peers).
    pub operations: Vec<UserOperation>,
}

/// A block waiting for attestations before finalization.
struct PendingBlock {
    header: BlockHeader,
    operations: Vec<UserOperation>,
    proposed_at: std::time::Instant,
    /// True if we produced this block (state already applied to store).
    /// False if we received it from a peer (NOT yet executed — Tendermint pattern).
    already_executed: bool,
    /// Execution result, only present if already_executed.
    result: Option<BlockResult>,
    /// Reverse-delta captured when we executed this block (only present if
    /// `already_executed`). Moved into the rollback journal on finalization so
    /// the journal stays aligned with the chain.
    revert: Option<solen_execution::executor::BlockRevert>,
    /// Count of attestation hash mismatches — other validators have a different block.
    mismatch_count: u32,
}

/// How many recent blocks' reverse-deltas to retain — the maximum fork depth
/// recoverable by in-place rollback before falling back to a snapshot restore.
const ROLLBACK_JOURNAL_CAP: usize = 256;

/// Consecutive un-appliable canonical sync blocks before we declare ourselves
/// stranded on a forked tip and trigger a resync. Above 1 so a single transient
/// bad-fork block from a minority peer doesn't force a resync.
const SYNC_REVERT_RESYNC_THRESHOLD: u32 = 5;

/// In-memory ring of per-block reverse-deltas. Lets a shallow fork be rolled
/// back to its common ancestor in place (apply the undo-deltas for the forked
/// suffix) instead of restoring a snapshot or re-downloading the chain. Bounded;
/// a fork deeper than the retained range falls back to the local-snapshot /
/// remote path. Lost on restart (the snapshot path covers that). Recorded in
/// lock-step with the chain Vec, so entry heights are contiguous and ascending.
struct RollbackJournal {
    entries: std::collections::VecDeque<(u64, solen_execution::executor::BlockRevert)>,
    cap: usize,
}

impl RollbackJournal {
    fn new(cap: usize) -> Self {
        Self {
            entries: std::collections::VecDeque::new(),
            cap: cap.max(1),
        }
    }

    /// Record block `height`'s reverse-delta. A non-contiguous height (after a
    /// snapshot restore / reset) clears the journal so we never stitch a
    /// rollback across a discontinuity.
    fn record(&mut self, height: u64, revert: solen_execution::executor::BlockRevert) {
        if let Some((last_h, _)) = self.entries.back() {
            if height != last_h + 1 {
                self.entries.clear();
            }
        }
        self.entries.push_back((height, revert));
        while self.entries.len() > self.cap {
            self.entries.pop_front();
        }
    }

    /// Lowest height we can roll back to: rolling back "to" `target` means
    /// undoing every block above it, so we need entries for `target+1..=tip`.
    /// That bottoms out one below the oldest retained entry.
    fn min_rollback_target(&self) -> Option<u64> {
        self.entries.front().map(|(h, _)| h.saturating_sub(1))
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

/// The consensus engine manages block production, validation, and finality.
pub struct ConsensusEngine {
    config: EngineConfig,
    store: Arc<RwLock<Box<dyn StateStore>>>,
    mempool: Mempool,
    executor: BlockExecutor,
    /// Keypair for signing block headers. Optional — non-validator nodes don't sign.
    signing_keypair: Option<solen_crypto::Keypair>,
    chain: Arc<RwLock<Vec<FinalizedBlock>>>,
    validator_set: Arc<RwLock<ValidatorSet>>,
    epoch_manager: Arc<RwLock<EpochManager>>,
    /// Pending attestations for blocks not yet finalized, keyed by block height.
    pending_attestations: Arc<RwLock<HashMap<u64, Vec<Attestation>>>>,
    /// Proposed blocks waiting for attestations before finalization.
    pending_blocks: Arc<RwLock<HashMap<u64, PendingBlock>>>,
    /// Reward events from epoch transitions, included in the next block's receipts.
    pending_reward_receipts: Arc<RwLock<Vec<solen_execution::receipt::ExecutionReceipt>>>,
    /// Intent pool for intent-aware execution.
    intent_pool: Arc<IntentPool>,
    /// Rollup proof verification registry.
    proof_registry: Arc<RwLock<ProofVerifierRegistry>>,
    /// Queued slashing evidence to include in the next block.
    pending_slashing: Arc<std::sync::Mutex<Vec<crate::slashing::SlashingEvidence>>>,
    /// Consecutive force-finalizations without quorum. When this exceeds a
    /// threshold, the validator stops producing blocks (likely partitioned).
    consecutive_force_finalizes: Arc<std::sync::atomic::AtomicU32>,
    /// Wall-clock time of the last partition-recovery probe. While latched in the
    /// partitioned state, the node is allowed to attempt production once every
    /// `PARTITION_PROBE_INTERVAL` (see `partition_probe_due`) so it can rejoin if
    /// the partition has cleared. Tracked separately from force-finalize activity
    /// so continuous no-quorum retries can't starve the probe.
    last_partition_probe_at: Arc<RwLock<Option<std::time::Instant>>>,
    /// Buffered attestations for blocks not yet received. Bounded and time-limited
    /// to prevent memory exhaustion while allowing out-of-order gossip delivery.
    early_attestations: Arc<RwLock<Vec<(ValidatorId, u64, Hash, std::time::Instant)>>>,
    /// Epoch seed for randomized proposer selection. Derived from the last block
    /// hash of the previous epoch. Updated at each epoch boundary.
    epoch_seed: Arc<RwLock<[u8; 32]>>,
    /// Dynamic finalized checkpoints — agreed by 2/3+ validators at epoch boundaries.
    finalized_checkpoints: Arc<RwLock<crate::checkpoint::FinalizedCheckpointStore>>,
    /// Trusted checkpoints for long-range attack protection.
    trusted_checkpoints: crate::checkpoint::TrustedCheckpoints,
    /// Height of a dropped block (our proposal that had attestation mismatches).
    /// The node layer reads and clears this to trigger a sync request.
    dropped_block_height: Arc<RwLock<Option<u64>>>,
    /// Set when the node needs a full snapshot resync (state diverged irrecoverably).
    needs_resync: Arc<std::sync::atomic::AtomicBool>,
    /// Consecutive synced canonical blocks we couldn't apply (reverted on a state
    /// root mismatch). A node stuck on a forked tip rejects every canonical block
    /// at its next height; once this crosses a threshold the node is genuinely
    /// stranded (not transiently rejecting one bad-fork block) and triggers a
    /// resync. Reset whenever a synced block applies. Precise fork-strand signal:
    /// it never fires during a partition-halt (no higher blocks arrive to revert).
    consecutive_sync_reverts: Arc<std::sync::atomic::AtomicU32>,
    /// True while a snapshot restore is in progress — blocks should not be finalized.
    resyncing: Arc<std::sync::atomic::AtomicBool>,
    /// Per-block reverse-deltas for in-place shallow-fork rollback.
    rollback_journal: Arc<RwLock<RollbackJournal>>,
    /// --- Attestation-aware fork choice (v2), gated by config.fork_choice_v2_height ---
    /// Competing block candidates per height, keyed by block hash. Unlike the
    /// single `pending_blocks`, v2 keeps EVERY valid candidate so it can finalize
    /// whichever hash reaches 2/3 even if it isn't the one we locally proposed.
    v2_blocks: Arc<RwLock<HashMap<u64, HashMap<Hash, PendingBlock>>>>,
    /// Each validator's CURRENT vote per height (vote-change allowed: a later
    /// attestation replaces its earlier one, net one vote per validator). The
    /// source of truth for v2 quorum.
    v2_votes: Arc<RwLock<HashMap<u64, HashMap<ValidatorId, Hash>>>>,
    /// Queue of (height, hash) the node layer must broadcast as our (possibly
    /// changed) attestation, appended when our deterministic vote leader moves.
    v2_revotes: Arc<std::sync::Mutex<Vec<(u64, Hash)>>>,
    /// Block hashes proven EXECUTION-INVALID at a height: we executed the block
    /// on our (parent-verified, canonical) tip and its state root did not match
    /// the proposer's claim. Such a block is objectively bad — it must never be
    /// re-accepted, re-voted, or chosen as the v2 leader; the backup proposer
    /// produces a valid block at this height that the fleet converges on instead.
    /// This is what lets the honest majority reject a bad proposer block and keep
    /// producing, rather than every node latching into a resync halt when they all
    /// blind-attest a bad header then reject it on execution (mainnet 2026-07-16).
    /// Gated on `fc_v2_active` (convergence on the backup needs v2 vote-change).
    /// GC'd as the height is passed.
    v2_invalid: Arc<RwLock<HashMap<u64, std::collections::HashSet<Hash>>>>,
    /// Broadcast channel for node events (WebSocket subscriptions, indexers).
    event_tx: tokio::sync::broadcast::Sender<NodeEvent>,
}

impl ConsensusEngine {
    /// Create with a single validator (backward compatible).
    pub fn new(config: EngineConfig, store: Box<dyn StateStore>, mempool: Mempool) -> Self {
        let validator_set = ValidatorSet::new(vec![ValidatorInfo::new(config.validator_id, 1000)]);
        Self::with_validators(config, store, mempool, validator_set)
    }

    /// Create with a multi-validator set. Restores chain height from
    /// persisted metadata if available.
    pub fn with_validators(
        config: EngineConfig,
        mut store: Box<dyn StateStore>,
        mempool: Mempool,
        validator_set: ValidatorSet,
    ) -> Self {
        // Try to load persisted chain metadata.
        let (restored_height, restored_epoch) = load_chain_meta(&*store);

        // BOOT RECONCILE: if a previous process eager-committed a block above the
        // finalized tip and died/was-superseded before finalizing (its in-memory
        // reverts now gone), the store is durably drifted — height N but N+1's
        // trie state. Replay any durable eager-revert records for heights above
        // the restored tip so we boot on the true finalized state, not the leak.
        // (mainnet validator5, 2026-08-06: without this it booted with block
        // 1,336,900's epoch-transition state at height 1,336,899 and could never
        // reapply the canonical boundary block.) Done BEFORE the placeholder
        // below reads `store.state_root()`, so the placeholder root is correct.
        if apply_durable_reverts_above(store.as_mut(), restored_height) {
            warn!(
                tip = restored_height,
                "boot: repaired durable store drift from a leaked eager-commit before starting"
            );
        }

        let mut chain = Vec::new();
        if restored_height > 0 {
            // Insert a placeholder finalized block so height() returns
            // the correct value. We don't have the full block data, but
            // we need the height to be correct for proposer rotation.
            let placeholder = FinalizedBlock {
                header: BlockHeader {
                    height: restored_height,
                    epoch: restored_epoch,
                    parent_hash: [0u8; 32],
                    state_root: store.state_root(),
                    transactions_root: [0u8; 32],
                    receipts_root: [0u8; 32],
                    proposer: [0u8; 32],
                    timestamp_ms: 0,
                    proposer_signature: vec![],
                },
                result: BlockResult {
                    state_root: store.state_root(),
                    receipts: vec![],
                    gas_used: 0,
                },
                attestations: vec![],
                operations: vec![],
            };
            chain.push(placeholder);
            info!(
                height = restored_height,
                epoch = restored_epoch,
                "restored chain height from state"
            );
        }

        let mut epoch_manager = EpochManager::new();
        epoch_manager.current_epoch = restored_epoch;

        let chain_id = config.chain_id;
        let fee_fix_height = config.fee_fix_height;
        let determinism_fix_height = config.determinism_fix_height;
        let pq_auth_height = config.pq_auth_height;
        let epoch_randomness_height = config.epoch_randomness_height;
        Self {
            config,
            store: Arc::new(RwLock::new(store)),
            mempool,
            executor: BlockExecutor::new()
                .with_chain_id(chain_id)
                .with_fee_fix_height(fee_fix_height)
                .with_determinism_fix_height(determinism_fix_height)
                .with_pq_auth_height(pq_auth_height)
                .with_epoch_randomness_height(epoch_randomness_height),
            signing_keypair: None, // set via set_signing_keypair() after construction
            chain: Arc::new(RwLock::new(chain)),
            validator_set: Arc::new(RwLock::new(validator_set)),
            epoch_manager: Arc::new(RwLock::new(epoch_manager)),
            pending_attestations: Arc::new(RwLock::new(HashMap::new())),
            pending_blocks: Arc::new(RwLock::new(HashMap::new())),
            pending_reward_receipts: Arc::new(RwLock::new(Vec::new())),
            intent_pool: Arc::new(IntentPool::new(10_000)),
            epoch_seed: Arc::new(RwLock::new([0u8; 32])), // genesis epoch uses round-robin
            finalized_checkpoints: Arc::new(RwLock::new(
                crate::checkpoint::FinalizedCheckpointStore::new(),
            )),
            proof_registry: {
                let mut reg = ProofVerifierRegistry::new();
                reg.register_verifier(Arc::new(solen_execution::proof::MockVerifier));
                Arc::new(RwLock::new(reg))
            },
            pending_slashing: Arc::new(std::sync::Mutex::new(Vec::new())),
            consecutive_force_finalizes: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            last_partition_probe_at: Arc::new(RwLock::new(None)),
            early_attestations: Arc::new(RwLock::new(Vec::new())),
            trusted_checkpoints: match chain_id {
                1 => crate::checkpoint::TrustedCheckpoints::mainnet(),
                9000 | 9001 => crate::checkpoint::TrustedCheckpoints::testnet(),
                _ => crate::checkpoint::TrustedCheckpoints::devnet(),
            },
            dropped_block_height: Arc::new(RwLock::new(None)),
            needs_resync: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            consecutive_sync_reverts: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            resyncing: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            rollback_journal: Arc::new(RwLock::new(RollbackJournal::new(ROLLBACK_JOURNAL_CAP))),
            v2_blocks: Arc::new(RwLock::new(HashMap::new())),
            v2_votes: Arc::new(RwLock::new(HashMap::new())),
            v2_revotes: Arc::new(std::sync::Mutex::new(Vec::new())),
            v2_invalid: Arc::new(RwLock::new(HashMap::new())),
            event_tx: tokio::sync::broadcast::channel(8192).0,
        }
    }

    pub fn validator_id(&self) -> ValidatorId {
        self.config.validator_id
    }

    /// Set the signing keypair for block header signatures.
    pub fn set_signing_keypair(&mut self, kp: solen_crypto::Keypair) {
        self.signing_keypair = Some(kp);
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Subscribe to node events (new blocks, tx confirmations, validator changes).
    pub fn subscribe_events(&self) -> tokio::sync::broadcast::Receiver<NodeEvent> {
        self.event_tx.subscribe()
    }

    /// Get the event sender for sharing with the RPC layer.
    pub fn event_sender(&self) -> tokio::sync::broadcast::Sender<NodeEvent> {
        self.event_tx.clone()
    }

    /// Emit events for a finalized block — block notification + per-tx confirmations.
    fn emit_block_events(&self, block: &FinalizedBlock) {
        let bh = block_hash(&block.header);

        // Block finalized event.
        let _ = self.event_tx.send(NodeEvent::BlockFinalized {
            height: block.header.height,
            epoch: block.header.epoch,
            block_hash: bh,
            state_root: block.header.state_root,
            proposer: block.header.proposer,
            timestamp_ms: block.header.timestamp_ms,
            tx_count: block.result.receipts.len(),
            gas_used: block.result.gas_used,
        });

        // Per-transaction confirmation events.
        for (i, receipt) in block.result.receipts.iter().enumerate() {
            let tx_hash = solen_crypto::receipt_tx_hash(
                block.header.height,
                i as u32,
                &receipt.sender,
                receipt.nonce,
            );
            let _ = self.event_tx.send(NodeEvent::TxIncluded {
                block_height: block.header.height,
                tx_hash,
                sender: receipt.sender,
                nonce: receipt.nonce,
                success: receipt.success,
                gas_used: receipt.gas_used,
            });
        }
    }

    /// Persist the last attested block to prevent amnesia after crash.
    /// A restarted validator must not attest to a different block at the same height.
    pub fn persist_last_attestation(&self, height: u64, block_hash: &Hash) {
        let mut data = Vec::with_capacity(40);
        data.extend_from_slice(&height.to_le_bytes());
        data.extend_from_slice(block_hash);
        let mut store = self.store.write().unwrap();
        let _ = store.put(b"__last_attestation__", &data);
    }

    /// Check if we already attested at this height (crash recovery).
    pub fn last_attested_block(&self) -> Option<(u64, Hash)> {
        let store = self.store.read().unwrap();
        store
            .get(b"__last_attestation__")
            .ok()
            .flatten()
            .and_then(|data| {
                if data.len() >= 40 {
                    let mut h = [0u8; 8];
                    h.copy_from_slice(&data[..8]);
                    let height = u64::from_le_bytes(h);
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&data[8..40]);
                    Some((height, hash))
                } else {
                    None
                }
            })
    }

    /// Current epoch seed for proposer selection randomization.
    pub fn epoch_seed(&self) -> [u8; 32] {
        *self.epoch_seed.read().unwrap()
    }

    /// Finalized checkpoint store.
    pub fn finalized_checkpoints(
        &self,
    ) -> Arc<RwLock<crate::checkpoint::FinalizedCheckpointStore>> {
        self.finalized_checkpoints.clone()
    }

    /// Get the pending checkpoint info for broadcasting our attestation.
    pub fn pending_checkpoint(&self) -> Option<(u64, Hash, Hash)> {
        let cp_store = self.finalized_checkpoints.read().unwrap();
        cp_store
            .pending
            .as_ref()
            .map(|p| (p.height, p.block_hash, p.state_root))
    }

    /// Add a checkpoint attestation from a validator.
    /// The caller must verify the signature. This method additionally checks
    /// that the attestation is for the CURRENT pending checkpoint to prevent
    /// replay of old attestations.
    pub fn attest_checkpoint_with_data(
        &self,
        validator_id: ValidatorId,
        signature: Vec<u8>,
        height: u64,
        block_hash: &Hash,
        state_root: &Hash,
    ) -> bool {
        // Verify the attestation is for the current pending checkpoint.
        {
            let cp_store = self.finalized_checkpoints.read().unwrap();
            if let Some(ref pending) = cp_store.pending {
                if pending.height != height
                    || pending.block_hash != *block_hash
                    || pending.state_root != *state_root
                {
                    return false; // Attestation is for a different/old checkpoint.
                }
            } else {
                return false; // No pending checkpoint.
            }
        }
        self.attest_checkpoint(validator_id, signature)
    }

    /// Add a checkpoint attestation from a validator (internal — skips data check).
    pub fn attest_checkpoint(&self, validator_id: ValidatorId, signature: Vec<u8>) -> bool {
        let vs = self.validator_set.read().unwrap();
        let mut cp_store = self.finalized_checkpoints.write().unwrap();
        let finalized = cp_store.add_attestation(validator_id, signature, &vs);
        if finalized {
            // Persist the finalized checkpoint.
            let store = self.store.read().unwrap();
            // Save to a non-state key so it doesn't affect state root.
            if let Some(ref cp) = cp_store.latest {
                if let Ok(data) = serde_json::to_vec(cp) {
                    drop(store);
                    let mut store = self.store.write().unwrap();
                    let _ = store.put(b"__finalized_checkpoint__", &data);
                }
            }
            info!(
                height = cp_store.latest.as_ref().map(|c| c.height).unwrap_or(0),
                "checkpoint FINALIZED (2/3+ quorum)"
            );
        }
        finalized
    }

    pub fn store(&self) -> Arc<RwLock<Box<dyn StateStore>>> {
        self.store.clone()
    }

    pub fn executor(&self) -> &BlockExecutor {
        &self.executor
    }

    /// Check if a block was dropped due to attestation mismatch. Returns and clears the height.
    pub fn take_dropped_block_height(&self) -> Option<u64> {
        self.dropped_block_height.write().unwrap().take()
    }

    /// How often a latched (partitioned) validator is allowed to attempt a
    /// recovery probe — see `partition_probe_due`. Roughly five block times.
    const PARTITION_PROBE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

    /// Returns true if the validator appears to be partitioned from the network
    /// (too many consecutive force-finalizations without quorum).
    pub fn is_likely_partitioned(&self) -> bool {
        self.consecutive_force_finalizes
            .load(std::sync::atomic::Ordering::Relaxed)
            > 3
    }

    /// While partitioned, returns true at most once per `PARTITION_PROBE_INTERVAL`
    /// to let the node attempt a single recovery probe (produce a block as usual)
    /// rather than staying silent forever. Self-healing: if the partition has
    /// cleared, the probe block gathers quorum, finalizes, and resets the latch
    /// on every node (each clears its own flag on accepting a valid block). If
    /// the partition is real, the probe simply fails to reach quorum (the
    /// force-finalize quorum gate refuses to finalize without 2/3+ stake), so it
    /// cannot create a divergent chain. This breaks the all-validator deadlock
    /// where every node latches, stops producing, and so never receives the
    /// block that would clear its flag. Calling this consumes the probe slot
    /// (advances the timer), so call it only when about to act on the result.
    pub fn partition_probe_due(&self) -> bool {
        let mut last = self.last_partition_probe_at.write().unwrap();
        let now = std::time::Instant::now();
        match *last {
            Some(t) if now.duration_since(t) < Self::PARTITION_PROBE_INTERVAL => false,
            _ => {
                *last = Some(now);
                true
            }
        }
    }

    /// Reset partition state — called when connectivity is restored.
    pub fn reset_partition_state(&self) {
        self.consecutive_force_finalizes
            .store(0, std::sync::atomic::Ordering::Relaxed);
        *self.last_partition_probe_at.write().unwrap() = None;
    }

    /// Signal that the node needs a full snapshot resync.
    pub fn request_resync(&self) {
        self.needs_resync
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Boot self-heal: if a 2/3-attested finalized checkpoint exists AT our
    /// current tip height and our store's root disagrees with it, our persisted
    /// tip is corrupt at an authoritative point — request a resync instead of
    /// participating (proposing / attesting / serving) with a bad tip. Returns
    /// true if a conflict was found and a resync was requested.
    ///
    /// Complements the durable eager-revert boot reconcile (which undoes a
    /// produce-path leak with a local undo-record): this covers drift that left
    /// NO local record — a crash mid-commit, or a sync-path advance — by trusting
    /// the supermajority-attested checkpoint over the local store. It fires only
    /// on a DEFINITIVE same-height root conflict (never on a mere "we're behind"),
    /// so a healthy node is unaffected. Checkpoints land at epoch boundaries, so
    /// this is a boundary-anchored check; the accept-path restore is what
    /// prevents the wedge in the first place.
    pub fn reconcile_tip_with_finalized_checkpoint(&self) -> bool {
        let tip = self.height();
        if tip == 0 {
            return false;
        }
        let expected = {
            let cp = self.finalized_checkpoints.read().unwrap();
            match &cp.latest {
                Some(c) if c.height == tip => c.state_root,
                _ => return false, // no attested checkpoint at our exact tip
            }
        };
        let ours = self.store.read().unwrap().state_root();
        if ours != expected {
            warn!(
                tip,
                ours = ?&ours[..4],
                expected = ?&expected[..4],
                "boot: store tip root conflicts with the 2/3-attested finalized checkpoint — requesting resync"
            );
            self.request_resync();
            return true;
        }
        false
    }

    /// Check and clear the resync flag.
    pub fn take_resync_request(&self) -> bool {
        self.needs_resync
            .swap(false, std::sync::atomic::Ordering::Relaxed)
    }

    /// Peek whether a resync has been requested, without clearing it. Used by
    /// the consensus loop's "syncing → continue" gate so a pending resync is
    /// not skipped: a sync-starved node is in syncing mode, and the resync
    /// executor lives after that gate.
    pub fn resync_requested(&self) -> bool {
        self.needs_resync.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if a snapshot restore is in progress.
    pub fn is_resyncing(&self) -> bool {
        self.resyncing.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set_resyncing(&self, v: bool) {
        self.resyncing
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Reset the engine to a specific height/epoch after a snapshot restore.
    pub fn reset_to_height(&self, height: u64, epoch: u64) {
        // Clear pending blocks and attestations.
        self.pending_blocks.write().unwrap().clear();
        self.pending_attestations.write().unwrap().clear();
        self.early_attestations.write().unwrap().clear();
        // v2 fork-choice state can't span a snapshot discontinuity.
        self.v2_blocks.write().unwrap().clear();
        self.v2_votes.write().unwrap().clear();
        self.v2_revotes.lock().unwrap().clear();
        self.v2_invalid.write().unwrap().clear();
        self.mempool.clear();
        self.consecutive_force_finalizes
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.consecutive_sync_reverts
            .store(0, std::sync::atomic::Ordering::Relaxed);
        *self.last_partition_probe_at.write().unwrap() = None;
        *self.dropped_block_height.write().unwrap() = None;
        // The journal can't span a snapshot discontinuity.
        self.rollback_journal.write().unwrap().clear();

        // Update epoch manager.
        {
            let mut em = self.epoch_manager.write().unwrap();
            em.current_epoch = epoch;
        }

        // Set chain to a single marker block at the snapshot height so height() returns correctly.
        {
            // The marker's result.state_root MUST carry the real restored root.
            // restore_store_to_finalized_tip() (the accept-path guard added for the
            // 2026-08-06 v5 fork) reads chain.last().result.state_root as the finalized
            // tip target. Leaving it [0;32] here — while the store holds the real root —
            // makes that guard see false "drift" after every snapshot resync, latch
            // needs_resync, and re-trigger resync forever → node hang (mainnet 2026-08-08,
            // validator9/11). The boot path (with_validators) already sets the real root
            // in this field, which is why a restart clears the hang. Keep the two paths
            // consistent: header and result both carry the restored root.
            let restored_root = {
                let store = self.store.read().unwrap();
                store.state_root()
            };
            let mut chain = self.chain.write().unwrap();
            chain.clear();
            chain.push(FinalizedBlock {
                header: BlockHeader {
                    height,
                    epoch,
                    parent_hash: [0u8; 32],
                    state_root: restored_root,
                    transactions_root: [0u8; 32],
                    receipts_root: [0u8; 32],
                    proposer: self.config.validator_id,
                    timestamp_ms: 0,
                    proposer_signature: vec![],
                },
                result: BlockResult {
                    state_root: restored_root,
                    receipts: vec![],
                    gas_used: 0,
                },
                attestations: vec![],
                operations: vec![],
            });
        }

        // Sync validator set from restored staking state.
        let staking = {
            let store = self.store.read().unwrap();
            solen_system_contracts::staking::StakingContract::load(store.as_ref())
        };
        {
            let mut vs = self.validator_set.write().unwrap();
            for sv in &staking.validators {
                if sv.is_active {
                    if let Some(v) = vs.get_mut(&sv.id) {
                        v.stake = sv.total_stake();
                    }
                }
            }
        }

        info!(height, epoch, "engine state reset after snapshot restore");
    }

    /// After the store has been wholesale-replaced from a local RocksDB
    /// checkpoint, re-initialise the in-memory chain/epoch from the restored
    /// chain metadata (the checkpoint carries its own `__chain_meta__`). Returns
    /// the restored `(height, epoch)`.
    pub fn reset_to_store_meta(&self) -> (u64, u64) {
        let (height, epoch) = {
            let store = self.store.read().unwrap();
            load_chain_meta(store.as_ref())
        };
        self.reset_to_height(height, epoch);
        (height, epoch)
    }

    /// The lowest height the in-memory rollback journal can rewind to, if any.
    /// The node layer uses this to bound its common-ancestor search before
    /// attempting an in-place rollback.
    pub fn min_rollback_target(&self) -> Option<u64> {
        self.rollback_journal.read().unwrap().min_rollback_target()
    }

    /// Roll the store back to `target` height in place by applying the journaled
    /// reverse-deltas for every block above it (newest-first), then truncating
    /// the chain, journal, and pending state to `target`. The caller must have
    /// confirmed `target` is a real common ancestor with the canonical chain
    /// (e.g. matching state roots) — as a safety net we verify the post-rollback
    /// state root equals `expected_root`.
    ///
    /// Returns true on success. Returns false either (a) early, before any
    /// mutation, when `target` is out of journal range; or (b) after partially
    /// applying reverts, if the safety-net root check fails (only reachable on a
    /// journal/target bug). In case (b) the store is left rolled back but the
    /// chain is NOT truncated, so it is inconsistent — the caller MUST fall back
    /// to a snapshot restore, which wipes and overwrites the store wholesale.
    pub fn rollback_to_height(&self, target: u64, expected_root: &Hash) -> bool {
        let tip = self.height();
        if target >= tip {
            return false;
        }
        // Must have contiguous journal coverage for target+1..=tip.
        match self.rollback_journal.read().unwrap().min_rollback_target() {
            Some(min) if target >= min => {}
            _ => {
                warn!(
                    target,
                    tip, "rollback target outside journal range — cannot roll back in place"
                );
                return false;
            }
        }

        // Apply undo-deltas newest-first under the store lock.
        {
            let journal = self.rollback_journal.read().unwrap();
            let mut store = self.store.write().unwrap();
            for (h, revert) in journal.entries.iter().rev() {
                if *h <= target {
                    break;
                }
                if let Err(e) = store.apply_batch_atomic(revert, true) {
                    error!(height = *h, error = %e, "rollback revert apply failed — aborting");
                    // Partial rollback applied; the caller's snapshot-restore
                    // fallback will overwrite the store wholesale, so no
                    // permanent corruption results.
                    return false;
                }
            }
            store.commit_root();

            // Safety net: the rolled-back root must match the canonical root at
            // `target`. If not, our journal/target was wrong; bail to snapshot.
            let rolled_root = store.state_root();
            if &rolled_root != expected_root {
                warn!(
                    target,
                    ours = ?&rolled_root[..4],
                    expected = ?&expected_root[..4],
                    "post-rollback state root mismatch — aborting in-place rollback"
                );
                return false;
            }
        }

        // Truncate chain, journal, pending state, and persisted meta to `target`.
        {
            let mut chain = self.chain.write().unwrap();
            chain.retain(|b| b.header.height <= target);
        }
        {
            let mut journal = self.rollback_journal.write().unwrap();
            while matches!(journal.entries.back(), Some((h, _)) if *h > target) {
                journal.entries.pop_back();
            }
        }
        self.pending_blocks.write().unwrap().clear();
        self.pending_attestations.write().unwrap().clear();
        self.early_attestations.write().unwrap().clear();
        *self.dropped_block_height.write().unwrap() = None;
        self.consecutive_force_finalizes
            .store(0, std::sync::atomic::Ordering::Relaxed);

        let target_epoch = target / crate::epoch::EPOCH_LENGTH;
        {
            let mut em = self.epoch_manager.write().unwrap();
            em.current_epoch = target_epoch;
        }
        {
            let mut store = self.store.write().unwrap();
            save_chain_meta(store.as_mut(), target, target_epoch);
        }

        info!(
            from = tip,
            to = target,
            "rolled back to common ancestor in place (no snapshot)"
        );
        true
    }

    pub fn chain(&self) -> Arc<RwLock<Vec<FinalizedBlock>>> {
        self.chain.clone()
    }

    pub fn mempool(&self) -> &Mempool {
        &self.mempool
    }

    pub fn validator_set(&self) -> Arc<RwLock<ValidatorSet>> {
        self.validator_set.clone()
    }

    /// Verify only that a header was signed by an active validator: proposer is
    /// in the active set AND the proposer signature is valid over the block
    /// hash. This deliberately does NOT check height contiguity, parent linkage,
    /// proposer rotation, or execute the block — it just proves the header is
    /// not an anonymous forgery.
    ///
    /// Used to gate height-tracking / sync triggers (C-03): a single
    /// unauthenticated `NewBlock` with a far-future height must not be allowed
    /// to poison a node's view of the network height and lock it into a
    /// permanent resync DoS. (Rotation is intentionally omitted: a legitimate
    /// far-future block can fall in a later epoch whose seed we don't yet know,
    /// which would false-reject; set membership + signature is the right bar for
    /// "a real validator produced this".)
    pub fn header_is_validator_signed(&self, header: &BlockHeader) -> bool {
        if header.proposer_signature.len() != 64 {
            return false;
        }
        {
            let vs = self.validator_set.read().unwrap();
            if !vs.active().iter().any(|v| v.id == header.proposer) {
                return false;
            }
        }
        let bh = block_hash(header);
        let mut sig = [0u8; 64];
        sig.copy_from_slice(&header.proposer_signature);
        solen_crypto::verify(&header.proposer, &bh, &sig).is_ok()
    }

    pub fn intent_pool(&self) -> Arc<IntentPool> {
        self.intent_pool.clone()
    }

    pub fn proof_registry(&self) -> Arc<RwLock<ProofVerifierRegistry>> {
        self.proof_registry.clone()
    }

    /// Simulate an operation using the engine's executor (with correct chain_id).
    pub fn simulate(
        &self,
        op: &UserOperation,
        store: &dyn solen_storage::StateStore,
    ) -> solen_execution::receipt::ExecutionReceipt {
        // Simulate as if the op were included in the next block, so contracts
        // reading `sdk::block_height()` see a realistic height.
        let next_height = self.height().saturating_add(1);
        self.executor.simulate(store, op, next_height)
    }

    pub fn height(&self) -> BlockHeight {
        let chain = self.chain.read().unwrap();
        chain.last().map(|b| b.header.height).unwrap_or(0)
    }

    /// Ensure our store reflects exactly the finalized tip before we produce.
    ///
    /// The produce path (`execute_block_journaled`) commits to the live store
    /// EAGERLY, before finalization, stashing an undo-delta in the pending block.
    /// That delta is only replayed into the rollback journal when the block
    /// FINALIZES. If a produce attempt is superseded first — by re-proposal, the
    /// backup proposer, the partition prober, or a competing block winning — its
    /// eager writes stay in the store with nothing to undo them. On a normal
    /// (empty / non-boundary) block that is harmless because production is
    /// idempotent, but at an EPOCH BOUNDARY the eager writes include the epoch
    /// reward distribution (`stage_block`, height % 100 == 0), so the next produce
    /// runs on a DOUBLE-rewarded store and computes a state root no other validator
    /// reproduces — the block the whole fleet rejects on execution, halting it
    /// (mainnet 2026-07-16, block 1028501: an empty block that diverged on the
    /// proposer alone). Restoring the store to the finalized tip here makes
    /// production start from clean, canonical state regardless of any superseded
    /// attempt. No-op in the common case (store already == tip), so it does not
    /// change the state root of any correctly-produced block.
    fn restore_store_to_finalized_tip(&self) {
        let tip_root = match self.chain.read().unwrap().last() {
            Some(b) => b.result.state_root,
            None => return,
        };
        if self.store.read().unwrap().state_root() == tip_root {
            return; // clean — the overwhelmingly common path
        }
        // Drift detected. Undo our own executed-but-unfinalized blocks using the
        // reverts stashed at produce time (newest height first). Applying a
        // reverse-delta is idempotent, so this is safe even if partially undone.
        let mut reverts: Vec<(u64, solen_execution::executor::BlockRevert)> = Vec::new();
        {
            let mut v2 = self.v2_blocks.write().unwrap();
            for (h, cands) in v2.iter_mut() {
                let ours: Vec<Hash> = cands
                    .iter()
                    .filter(|(_, pb)| pb.already_executed && pb.revert.is_some())
                    .map(|(hash, _)| *hash)
                    .collect();
                for hash in ours {
                    if let Some(pb) = cands.remove(&hash) {
                        if let Some(r) = pb.revert {
                            reverts.push((*h, r));
                        }
                    }
                }
            }
        }
        {
            let mut pend = self.pending_blocks.write().unwrap();
            let ours: Vec<u64> = pend
                .iter()
                .filter(|(_, pb)| pb.already_executed && pb.revert.is_some())
                .map(|(h, _)| *h)
                .collect();
            for h in ours {
                if let Some(pb) = pend.remove(&h) {
                    if let Some(r) = pb.revert {
                        reverts.push((h, r));
                    }
                }
            }
        }
        reverts.sort_by(|a, b| b.0.cmp(&a.0)); // newest height first
        {
            let mut store = self.store.write().unwrap();
            for (h, revert) in &reverts {
                if let Err(e) = store.apply_batch_atomic(revert, true) {
                    warn!(height = *h, error = %e, "failed to undo leaked eager-commit before produce");
                }
            }
            store.commit_root();
        }
        let now_root = self.store.read().unwrap().state_root();
        if now_root == tip_root {
            warn!("store had drifted from the finalized tip — undid a leaked eager-commit before producing");
        } else {
            // A deeper drift we can't undo locally (e.g. reverts already GC'd).
            // Resync is the safe fallback; with the finalize-mismatch fix a single
            // resyncing node no longer halts the fleet.
            // In-memory reverts didn't fully undo the drift — the leak likely
            // predates this process (a superseded eager-commit persisted, then a
            // restart dropped the in-memory reverts). Try the DURABLE reverts
            // before giving up on local recovery.
            if self.reconcile_store_from_durable_reverts(self.height()) {
                let repaired = self.store.read().unwrap().state_root();
                if repaired == tip_root {
                    warn!("store had drifted from the finalized tip — repaired via durable eager-revert (survived restart)");
                    return;
                }
            }
            warn!("store still drifted from finalized tip after undo — requesting resync to avoid proposing a divergent block");
            self.needs_resync
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Delete the durable eager-revert record for `height`. Called when a block
    /// FINALIZES: it is now canonical, so its eager writes must never be undone.
    fn clear_eager_revert(&self, height: u64) {
        let mut store = self.store.write().unwrap();
        if let Err(e) = store.delete(&solen_storage::eager_revert_key(height)) {
            warn!(height, error = %e, "failed to clear durable eager-revert record");
        }
    }

    /// Repair a store that drifted above the finalized tip by replaying DURABLE
    /// eager-revert records for every height > `tip_height`, newest-first. This
    /// is the restart-safe counterpart to `restore_store_to_finalized_tip`: the
    /// in-memory reverts live only for the current process, but these records
    /// survive on disk (written atomically with the eager writes they undo).
    ///
    /// Returns true if any record was applied. After applying, the caller should
    /// verify the resulting state root equals the expected finalized-tip root.
    fn reconcile_store_from_durable_reverts(&self, tip_height: u64) -> bool {
        let mut store = self.store.write().unwrap();
        apply_durable_reverts_above(store.as_mut(), tip_height)
    }

    pub fn get_block(&self, height: BlockHeight) -> Option<FinalizedBlock> {
        let chain = self.chain.read().unwrap();
        chain.iter().find(|b| b.header.height == height).cloned()
    }

    pub fn latest_block(&self) -> Option<FinalizedBlock> {
        let chain = self.chain.read().unwrap();
        chain.last().cloned()
    }

    /// Produce a single block. Returns the block, its operations, and
    /// whether it was immediately finalized (single-validator mode).
    pub fn produce_block(&self) -> ProducedBlock {
        // The produce path commits eagerly; guarantee we build on the finalized
        // tip's state, not on writes left behind by a superseded earlier attempt
        // (which at an epoch boundary would double-apply the reward and diverge).
        self.restore_store_to_finalized_tip();

        let mut ops = self.mempool.drain(self.config.max_ops_per_block);

        // Filter out operations with stale nonces (already finalized by peer blocks).
        // This prevents state root mismatches when gossiped txs land in peer blocks
        // before we include them.
        {
            let store = self.store.read().unwrap();
            ops.retain(|op| {
                let key = {
                    let mut k = b"acc/".to_vec();
                    k.extend_from_slice(&op.sender);
                    k
                };
                match store.get(&key) {
                    Ok(Some(data)) => {
                        if let Ok(account) =
                            borsh::from_slice::<solen_types::account::Account>(&data)
                        {
                            op.nonce >= account.nonce
                        } else {
                            true
                        }
                    }
                    _ => true,
                }
            });
        }

        // Include queued slashing evidence as deterministic system operations.
        {
            let mut queue = self.pending_slashing.lock().unwrap();
            for evidence in queue.drain(..) {
                let penalty_bps = evidence.reason.penalty_bps();
                // Build args: offender[32] + penalty_bps[8]
                let mut args = Vec::with_capacity(40);
                args.extend_from_slice(&evidence.offender);
                args.extend_from_slice(&penalty_bps.to_le_bytes());

                ops.push(solen_types::transaction::UserOperation {
                    sender: self.config.validator_id,
                    nonce: 0,
                    actions: vec![solen_types::transaction::Action::Call {
                        target: solen_types::system::STAKING_ADDRESS,
                        method: "slash".to_string(),
                        args,
                    }],
                    max_fee: 0,
                    signature: vec![0xFF], // system-authorized
                });
            }
        }

        // Expire intents past current height.
        let current_h = self.height();
        let expired = self.intent_pool.expire(current_h);
        if expired > 0 {
            debug!(expired, height = current_h, "expired intents");
        }

        // Solve pending intents and include as system operations in the block.
        // MEV protection: prefer external solver solutions. The block proposer's
        // built-in solver is only used as a fallback after the intent has been
        // pending for at least 2 blocks (giving external solvers priority).
        let pending = self.intent_pool.pending_intents();
        if !pending.is_empty() {
            let solver = DirectTransferSolver {
                id: self.config.validator_id,
            };
            let current_height = self.height();

            for intent in &pending {
                // Prefer externally-submitted solutions.
                let external_solution = self.intent_pool.select_best_solution(intent.id).ok();

                let solution = if external_solution.is_some() {
                    external_solution
                } else {
                    // Only use built-in solver if intent has been pending for > 2 blocks.
                    // This gives external solvers a fair window to submit solutions.
                    let blocks_pending =
                        current_height.saturating_sub(intent.expiry_height.saturating_sub(500));
                    if blocks_pending >= 2 {
                        solver.solve(intent)
                    } else {
                        None // Skip — wait for external solvers.
                    }
                };

                if let Some(sol) = solution {
                    // Build a system call operation: sender calls INTENT_ADDRESS.fulfill(...)
                    // Args: intent_id[8] + solver[32] + claimed_tip[16]
                    //       + num_transfers[4] + (to[32]+amount[16])*N
                    //       + num_constraints[4] + encoded_constraints
                    let mut args = Vec::new();
                    args.extend_from_slice(&intent.id.to_le_bytes()); // intent_id[8]
                    args.extend_from_slice(&sol.solver); // solver[32]
                    args.extend_from_slice(&sol.claimed_tip.to_le_bytes()); // claimed_tip[16]

                    let mut transfer_count: u32 = 0;
                    let count_pos = args.len();
                    args.extend_from_slice(&0u32.to_le_bytes()); // placeholder

                    for op in &sol.operations {
                        for action in &op.actions {
                            if let solen_types::transaction::Action::Transfer { to, amount } =
                                action
                            {
                                args.extend_from_slice(to);
                                args.extend_from_slice(&amount.to_le_bytes());
                                transfer_count += 1;
                            }
                        }
                    }

                    // Patch transfer count.
                    args[count_pos..count_pos + 4].copy_from_slice(&transfer_count.to_le_bytes());

                    // Encode constraints so the system call can verify them.
                    // Format: num_constraints[4] + (type[1] + constraint_data)*N
                    //   type 0 = MinBalance: account[32] + min_amount[16]
                    //   type 1 = MaxSpend:   account[32] + max_amount[16]
                    //   type 2 = RequireTransfer: from[32] + to[32] + min_amount[16]
                    //   type 3 = RequireCall: target[32] + method_len[4] + method_bytes
                    let num_constraints = intent.constraints.len() as u32;
                    args.extend_from_slice(&num_constraints.to_le_bytes());
                    for c in &intent.constraints {
                        use solen_intents::types::Constraint;
                        match c {
                            Constraint::MinBalance {
                                account,
                                min_amount,
                            } => {
                                args.push(0);
                                args.extend_from_slice(account);
                                args.extend_from_slice(&min_amount.to_le_bytes());
                            }
                            Constraint::MaxSpend {
                                account,
                                max_amount,
                            } => {
                                args.push(1);
                                args.extend_from_slice(account);
                                args.extend_from_slice(&max_amount.to_le_bytes());
                            }
                            Constraint::RequireTransfer {
                                from,
                                to,
                                min_amount,
                            } => {
                                args.push(2);
                                args.extend_from_slice(from);
                                args.extend_from_slice(to);
                                args.extend_from_slice(&min_amount.to_le_bytes());
                            }
                            Constraint::RequireCall { target, method } => {
                                args.push(3);
                                args.extend_from_slice(target);
                                let method_bytes = method.as_bytes();
                                args.extend_from_slice(&(method_bytes.len() as u32).to_le_bytes());
                                args.extend_from_slice(method_bytes);
                            }
                            Constraint::CrossChainSwap {
                                input_amount,
                                min_output,
                                destination_chain,
                                destination_address,
                                output_token,
                            } => {
                                args.push(4); // type 4 = CrossChainSwap
                                args.extend_from_slice(&input_amount.to_le_bytes());
                                args.extend_from_slice(&min_output.to_le_bytes());
                                args.extend_from_slice(&destination_chain.to_le_bytes());
                                args.extend_from_slice(destination_address);
                                args.extend_from_slice(output_token);
                            }
                            Constraint::Custom { .. } => {
                                // Custom constraints require verifier contract execution —
                                // skip for now, will be added with WASM verifier support.
                            }
                        }
                    }

                    ops.push(solen_types::transaction::UserOperation {
                        sender: intent.sender,
                        nonce: 0, // system ops use nonce 0
                        actions: vec![solen_types::transaction::Action::Call {
                            target: solen_types::system::INTENT_ADDRESS,
                            method: "fulfill".to_string(),
                            args,
                        }],
                        max_fee: intent.max_fee,
                        signature: vec![0xFF], // marker for system-authorized intent ops
                    });

                    let _ = self.intent_pool.fulfill(intent.id);
                    info!(intent_id = intent.id, "intent solution included in block");
                }
            }
        }

        let op_count = ops.len();

        let (parent_hash, height, parent_ts) = {
            let chain = self.chain.read().unwrap();
            let parent = chain
                .last()
                .map(|b| block_hash(&b.header))
                .unwrap_or([0u8; 32]);
            let h = chain.last().map(|b| b.header.height + 1).unwrap_or(1);
            let pts = chain.last().map(|b| b.header.timestamp_ms).unwrap_or(0);
            (parent, h, pts)
        };

        // Execute block with height so the executor handles epoch rewards
        // deterministically. Capture the reverse-delta for the rollback journal
        // (recorded on finalization, keeping the journal aligned with the chain).
        let (result, revert) = {
            let mut store = self.store.write().unwrap();
            // Durable-journaled: the eager writes AND their undo-record commit in
            // one atomic batch, so a superseded produce attempt (or a crash
            // between produce and finalize) can be undone even after a restart.
            // The record is cleared when this block finalizes, or replayed to
            // repair drift at boot / before the next produce.
            self.executor
                .execute_block_journaled_durable(store.as_mut(), &ops, height)
        };

        let epoch = {
            let em = self.epoch_manager.read().unwrap();
            em.epoch_for_height(height)
        };

        let mut header = BlockHeader {
            height,
            epoch,
            parent_hash,
            state_root: result.state_root,
            transactions_root: compute_tx_root(&ops),
            receipts_root: compute_receipts_root(&result),
            proposer: self.config.validator_id,
            // DETERMINISTIC block timestamp = parent's + one block interval, NOT
            // wall-clock. This makes block production idempotent: re-proposing a
            // height (after a quorum-timeout drop, by the partition prober, or
            // after a proposer restart) yields the SAME block hash, so
            // attestations accumulate on one block and the fleet converges
            // instead of splitting across ever-changing hashes — the wedge the
            // devnet drill surfaced and the root of the 2026-06 fork cascade.
            // Monotonic by construction (> parent); stays <= wall-clock because
            // production is rate-limited to one block per interval, so it never
            // trips the future-drift check. On a live chain it continues from the
            // last real timestamp, so block times stay wall-clock-aligned.
            timestamp_ms: parent_ts.saturating_add(self.config.block_time_ms),
            proposer_signature: vec![],
        };
        // Sign the header (signature covers all fields except itself).
        // This proves the proposer actually authored the block and prevents
        // attribution attacks where a relay changes the proposer field.
        if let Some(ref kp) = self.signing_keypair {
            let bh_for_sig = block_hash(&header);
            header.proposer_signature = kp.sign(&bh_for_sig).to_vec();
        }

        let bh = block_hash(&header);

        let is_single = self.validator_set.read().unwrap().active_count() <= 1;

        if is_single {
            // Epoch rewards are handled by the executor via execute_block_with_height.

            let attestations = vec![Attestation {
                validator_id: self.config.validator_id,
                block_height: height,
                block_hash: bh,
            }];

            let block = FinalizedBlock {
                header: header.clone(),
                result,
                attestations,
                operations: ops.clone(),
            };

            self.chain.write().unwrap().push(block.clone());
            self.rollback_journal
                .write()
                .unwrap()
                .record(height, revert);
            // Block is canonical now — drop its durable eager-revert so it is
            // never undone at a later boot/reconcile.
            self.clear_eager_revert(height);
            self.persist_block_and_meta(&block);
            self.emit_block_events(&block);
            self.mempool.remove_finalized(&block.operations);

            self.try_epoch_transition(height);

            info!(
                height,
                ops = op_count,
                epoch,
                "block finalized (single validator)"
            );

            ProducedBlock {
                finalized: Some(block),
                header: header.clone(),
                operations: ops,
            }
        } else {
            // Epoch rewards are handled by the executor via execute_block_with_height.

            // v2: register OUR executed block as a candidate (preserving the
            // already-executed result + revert so finalization doesn't re-run it)
            // and cast our self-vote. v2_record_vote enqueues a revote for the
            // node layer to broadcast.
            if self.fc_v2_active(height) {
                self.v2_blocks
                    .write()
                    .unwrap()
                    .entry(height)
                    .or_default()
                    .insert(
                        bh,
                        PendingBlock {
                            header: header.clone(),
                            operations: ops.clone(),
                            proposed_at: std::time::Instant::now(),
                            already_executed: true,
                            result: Some(result),
                            revert: Some(revert),
                            mismatch_count: 0,
                        },
                    );
                self.persist_last_attestation(height, &bh);
                // Cast our self-vote and ENQUEUE it for the node layer to
                // broadcast (v2_reevaluate only enqueues on a vote *change*, but
                // here we're the proposer casting a fresh vote for our own block).
                self.v2_votes
                    .write()
                    .unwrap()
                    .entry(height)
                    .or_default()
                    .insert(self.config.validator_id, bh);
                self.v2_revotes.lock().unwrap().push((height, bh));
                self.v2_reevaluate(height);
                info!(
                    height,
                    ops = op_count,
                    epoch,
                    "block proposed (v2), waiting for attestations"
                );
                return ProducedBlock {
                    finalized: None,
                    header,
                    operations: ops,
                };
            }

            // Store as pending, self-attest,
            // wait for peer attestations to reach quorum.
            self.pending_blocks.write().unwrap().insert(
                height,
                PendingBlock {
                    header: header.clone(),
                    operations: ops.clone(),
                    proposed_at: std::time::Instant::now(),
                    already_executed: true,
                    result: Some(result),
                    revert: Some(revert),
                    mismatch_count: 0,
                },
            );

            // Persist attestation WAL before attesting (prevents amnesia on crash).
            self.persist_last_attestation(height, &bh);

            // Self-attest.
            self.accept_attestation(self.config.validator_id, height, bh);

            info!(
                height,
                ops = op_count,
                epoch,
                "block proposed, waiting for attestations"
            );

            ProducedBlock {
                finalized: None,
                header,
                operations: ops,
            }
        }
    }

    /// Check if this node is the proposer for the next block.
    pub fn is_next_proposer(&self) -> bool {
        let next_height = self.height() + 1;
        let seed = self.epoch_seed();
        let vs = self.validator_set.read().unwrap();
        vs.proposer_for_height_with_seed(next_height, &seed)
            .map(|id| id == self.config.validator_id)
            .unwrap_or(false)
    }

    /// Check if this node should act as backup proposer.
    ///
    /// When the designated proposer is offline, backup proposers take over
    /// in deterministic order. The first backup (next in round-robin after
    /// the designated proposer) waits 3x block_time. Subsequent backups
    /// wait an additional 2x block_time each.
    ///
    /// To prevent multiple validators from proposing competing blocks at
    /// the same height (which causes attestation hash mismatches), we also
    /// check if we've already received a pending block at this height from
    /// another validator. If so, we don't propose — we wait for that block
    /// to either reach quorum or timeout.
    /// Whether THIS node is the designated backup proposer right now, for when
    /// the primary proposer is slow/down.
    ///
    /// DETERMINISTIC across the fleet: the stall is measured against a SHARED
    /// reference — the wall-clock timestamp the last finalized block was produced
    /// with (carried in its header) — not each node's own `last_finalized_at`.
    /// With NTP-synced clocks every node computes the same backup round, so
    /// exactly ONE backup produces per round. The previous version keyed the
    /// round off each node's *local* elapsed time, so nodes that finalized a few
    /// hundred ms apart each believed they were the backup and emitted competing
    /// blocks at the same height — splitting attestations into a fork (root cause
    /// of 2026-06-08 and the 2026-06-24 fork cascade). The partition prober
    /// already fixed this for the latched state; this fixes normal operation.
    pub fn is_backup_proposer(&self, stalled_for: std::time::Duration) -> bool {
        // Local timing GATE: only consider stepping in once the primary has had a
        // few block-intervals to produce. `stalled_for` is per-node (not shared),
        // but it only gates WHEN we act, not WHO acts — selection below is
        // deterministic — so it cannot cause competing blocks, only briefly delay
        // the single chosen backup.
        let min_wait = std::time::Duration::from_millis(self.config.block_time_ms * 3);
        if stalled_for < min_wait {
            return false;
        }

        let next_height = self.height() + 1;
        let seed = self.epoch_seed();
        let order = {
            let vs = self.validator_set.read().unwrap();
            vs.proposer_order_for_height(next_height, &seed)
        };
        if order.len() <= 1 {
            return false;
        }

        // Deterministic backup SELECTION from shared wall-clock (NTP), rotating
        // each window — the same mechanism as the partition prober. Every node
        // picks the SAME single backup, so no competing blocks. Crucially this is
        // independent of block timestamps: those are now logical (parent +
        // block_time, for idempotent re-proposal), so `now - last_block_ts` is no
        // longer a real elapsed time and must NOT drive backup timing.
        let interval_secs = ((self.config.block_time_ms * 2).max(4000) / 1000).max(1);
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let round = (now_secs / interval_secs) as usize;

        // Position in the proposer order: 0 = designated, 1 = first backup, etc.
        let backup_position = (round + 1) % order.len();

        order[backup_position] == self.config.validator_id
    }

    /// Deterministic, wall-clock-synchronized recovery-probe proposer.
    ///
    /// While the network is latched (`is_likely_partitioned`), every node must
    /// agree on a SINGLE validator to re-attempt production each
    /// `PARTITION_PROBE_INTERVAL` window, otherwise multiple nodes act as backup
    /// proposer — each deriving its turn from its own *local* `stalled_for` — and
    /// emit competing blocks at the wedged height whose attestations split, so no
    /// block reaches quorum and the latch never clears (the deadlock seen on
    /// 2026-06-08). We pick the prober from shared wall-clock time so all nodes
    /// choose the same one, rotating through the proposer order each window so a
    /// down proposer is skipped in the next window. Recovery therefore converges
    /// within `order.len()` windows even if several proposers are offline.
    pub fn partition_probe_proposer(&self) -> Option<ValidatorId> {
        let next_height = self.height() + 1;
        let seed = self.epoch_seed();
        let vs = self.validator_set.read().unwrap();
        let order = vs.proposer_order_for_height(next_height, &seed);
        if order.is_empty() {
            return None;
        }
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let window = now_secs / Self::PARTITION_PROBE_INTERVAL.as_secs().max(1);
        Some(order[(window as usize) % order.len()])
    }

    /// Whether this node is the recovery-probe proposer for the current window.
    pub fn is_partition_probe_proposer(&self) -> bool {
        self.partition_probe_proposer()
            .map(|id| id == self.config.validator_id)
            .unwrap_or(false)
    }

    /// Accept a block proposed by another validator.
    ///
    /// Following the Tendermint pattern: do NOT execute the block here.
    /// Just validate header consistency (height, parent hash, epoch) and
    /// store as pending. Execution happens in `finalize_pending_block`
    /// after quorum is reached. This prevents state corruption from
    /// rejected blocks.
    pub fn accept_block(&self, header: &BlockHeader, operations: &[UserOperation]) -> bool {
        // Don't accept blocks while a snapshot restore is in progress.
        if self.is_resyncing() {
            return false;
        }

        let (our_height, expected_height, fork_detected) = {
            let chain = self.chain.read().unwrap();
            let our_height = chain.last().map(|b| b.header.height).unwrap_or(0);
            let expected_height = our_height + 1;

            if header.height < expected_height {
                return false; // Old block, ignore.
            }

            let fork = if header.height == expected_height {
                if let Some(last_block) = chain.last() {
                    // A real parent exists, so the block MUST chain to it. There
                    // is no zero-parent_hash exception here — accepting [0;32]
                    // would let a forged block skip parent-linkage entirely and
                    // attach off our actual head.
                    let expected_parent = block_hash(&last_block.header);
                    header.parent_hash != expected_parent
                } else {
                    false
                }
            } else {
                false
            };

            (our_height, expected_height, fork)
        };

        if fork_detected {
            debug!(
                height = header.height,
                "parent hash mismatch — rejecting block, waiting for sync"
            );
            return false;
        }

        let bh = block_hash(header);

        // Never re-accept a block we already proved execution-invalid at this
        // height. The forked proposer keeps re-gossiping it, and (being the
        // designated rank-0 proposer) v2 would otherwise keep re-selecting it as
        // leader — livelocking against the backup's valid block. Ignoring it here
        // lets the backup take the slot. (Empty in the common case.)
        if self
            .v2_invalid
            .read()
            .unwrap()
            .get(&header.height)
            .is_some_and(|s| s.contains(&bh))
        {
            debug!(height = header.height, hash = ?&bh[..4],
                   "ignoring known execution-invalid block");
            return false;
        }

        // Validate against trusted checkpoints (long-range attack protection).
        if let Some(expected) = self.trusted_checkpoints.validate(header.height, &bh) {
            warn!(
                height = header.height,
                expected = ?&expected[..4],
                got = ?&bh[..4],
                "block violates trusted checkpoint — rejecting (possible long-range attack)"
            );
            return false;
        }

        // Validate against dynamic finalized checkpoints.
        {
            let cp_store = self.finalized_checkpoints.read().unwrap();
            if let Some(expected) = cp_store.validate_block(header.height, &bh) {
                warn!(
                    height = header.height,
                    expected = ?&expected[..4],
                    got = ?&bh[..4],
                    "block conflicts with finalized checkpoint — rejecting"
                );
                return false;
            }
        }

        if header.height > expected_height {
            debug!(
                our_height,
                block_height = header.height,
                gap = header.height - expected_height,
                "block ahead of our height — waiting for sync"
            );
            return false;
        }

        // Validate block timestamp.
        // Must be monotonically increasing and within 30s of local time.
        // Prevents proposers from manipulating time-dependent logic.
        {
            let chain = self.chain.read().unwrap();
            if let Some(last) = chain.last() {
                if header.timestamp_ms <= last.header.timestamp_ms {
                    warn!(
                        height = header.height,
                        block_ts = header.timestamp_ms,
                        prev_ts = last.header.timestamp_ms,
                        "block timestamp not monotonically increasing — rejecting"
                    );
                    return false;
                }
            }
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        const MAX_TIMESTAMP_DRIFT_MS: u64 = 30_000; // 30 seconds
        if header.timestamp_ms > now_ms + MAX_TIMESTAMP_DRIFT_MS {
            warn!(
                height = header.height,
                block_ts = header.timestamp_ms,
                local_ts = now_ms,
                drift_ms = header.timestamp_ms - now_ms,
                "block timestamp too far in the future — rejecting"
            );
            return false;
        }

        // Validate epoch.
        let expected_epoch = header.height / crate::epoch::EPOCH_LENGTH;
        if header.epoch != expected_epoch {
            warn!(
                height = header.height,
                expected = expected_epoch,
                got = header.epoch,
                "invalid epoch — rejecting block"
            );
            return false;
        }

        // Validate proposer is a known active validator AND is in the
        // proposer rotation for this height (designated or backup).
        {
            let vs = self.validator_set.read().unwrap();
            let is_valid_proposer = vs.active().iter().any(|v| v.id == header.proposer);
            if !is_valid_proposer {
                warn!(
                    height = header.height,
                    proposer = ?header.proposer[..4],
                    "proposer not in active validator set — rejecting block"
                );
                return false;
            }
            // Check proposer is in the rotation order for this height.
            // Accept designated proposer + any backup (they take over if designated is offline).
            let seed = self.epoch_seed();
            let order = vs.proposer_order_for_height(header.height, &seed);
            if !order.contains(&header.proposer) {
                warn!(
                    height = header.height,
                    proposer = ?&header.proposer[..4],
                    "proposer not in rotation order — rejecting block"
                );
                return false;
            }
        }

        // Verify proposer signature to prevent forged block headers.
        // Every non-genesis block MUST carry a valid 64-byte signature from its
        // proposer. block_hash() excludes the signature field, so an unsigned
        // forgery would otherwise hash identically to a legitimate block and
        // collect honest attestations. There is no empty-signature exception:
        // all validators sign the blocks they produce.
        if header.height > 0 {
            if header.proposer_signature.len() != 64 {
                warn!(
                    height = header.height,
                    sig_len = header.proposer_signature.len(),
                    proposer = ?&header.proposer[..4],
                    "missing or malformed proposer signature — rejecting block"
                );
                return false;
            }
            let bh = block_hash(header);
            let mut sig = [0u8; 64];
            sig.copy_from_slice(&header.proposer_signature);
            if solen_crypto::verify(&header.proposer, &bh, &sig).is_err() {
                warn!(
                    height = header.height,
                    proposer = ?&header.proposer[..4],
                    "invalid proposer signature — rejecting block"
                );
                return false;
            }
        }

        // Check for duplicate pending/finalized blocks.
        {
            let chain = self.chain.read().unwrap();
            if let Some(finalized) = chain.iter().find(|b| b.header.height == header.height) {
                // A different block at the same height from the same proposer is
                // equivocation even after one side finalized. Evaluate it for
                // slashing before dropping it — otherwise a proposer that
                // equivocates across a partition escapes punishment once either
                // side commits. check_double_sign requires both headers to be
                // validly signed, so this cannot be abused to frame a validator.
                let finalized_header = finalized.header.clone();
                drop(chain);
                if finalized_header.proposer == header.proposer
                    && block_hash(&finalized_header) != block_hash(header)
                {
                    if let Some(evidence) =
                        crate::slashing::check_double_sign(&finalized_header, header)
                    {
                        warn!(
                            height = header.height,
                            proposer = ?&header.proposer[..4],
                            "DOUBLE SIGN DETECTED against finalized block — queuing slashing evidence"
                        );
                        self.process_slashing(&evidence);
                    }
                }
                return false; // Already finalized — do not re-accept.
            }
            drop(chain);

            // v2: keep EVERY valid candidate (not a single rank-chosen pending),
            // so whichever hash the fleet's votes converge on can finalize.
            if self.fc_v2_active(header.height) {
                self.v2_record_block(header, operations);
                return true;
            }

            let pending = self.pending_blocks.read().unwrap();
            if let Some(existing) = pending.get(&header.height) {
                let existing_hash = block_hash(&existing.header);
                let existing_header = existing.header.clone();
                let is_same_proposer = existing.header.proposer == header.proposer;
                let was_already_executed = existing.already_executed;
                let new_hash = block_hash(header);
                drop(pending);

                if existing_hash == new_hash {
                    return false; // identical block, already have it
                }

                // Different block at same height from the same proposer = double-sign.
                if is_same_proposer {
                    if let Some(evidence) =
                        crate::slashing::check_double_sign(&existing_header, header)
                    {
                        warn!(
                            height = header.height,
                            proposer = ?&header.proposer[..4],
                            "DOUBLE SIGN DETECTED — queuing slashing evidence"
                        );
                        self.process_slashing(&evidence);
                    }
                    return false;
                }

                // Competing block from a different proposer. Use stake-weighted
                // fork scoring: prefer the block from the higher-priority proposer
                // (lower position in the proposer order for this height).
                let seed = self.epoch_seed();
                let vs = self.validator_set.read().unwrap();
                let order = vs.proposer_order_for_height(header.height, &seed);
                drop(vs);

                let existing_rank = order
                    .iter()
                    .position(|id| *id == existing_header.proposer)
                    .unwrap_or(usize::MAX);
                let new_rank = order
                    .iter()
                    .position(|id| *id == header.proposer)
                    .unwrap_or(usize::MAX);

                if new_rank < existing_rank {
                    // Check if we already executed our own block for this height.
                    // If so, we can't replace it without corrupting the store — the
                    // execution already mutated state. Let the timeout/resync handle it.
                    if was_already_executed {
                        warn!(
                            height = header.height,
                            existing_rank,
                            new_rank,
                            "cannot replace already-executed block — keeping ours, will resync if needed"
                        );
                        return false;
                    }

                    // New block is from a higher-priority proposer — replace.
                    info!(
                        height = header.height,
                        existing_rank,
                        new_rank,
                        "replacing pending block with higher-priority proposer's block"
                    );

                    // Remove old pending block and its attestations.
                    self.pending_blocks.write().unwrap().remove(&header.height);
                    self.pending_attestations
                        .write()
                        .unwrap()
                        .remove(&header.height);
                    // Fall through to accept the new block below.
                } else {
                    debug!(
                        height = header.height,
                        existing_rank,
                        new_rank,
                        "keeping existing block (higher priority proposer)"
                    );
                    return false;
                }
            } else {
                drop(pending);
            }
        }

        // Store as pending WITHOUT executing. Execution happens on finalization.
        // This is the key Tendermint pattern: validate header, vote, execute on commit.
        self.pending_blocks.write().unwrap().insert(
            header.height,
            PendingBlock {
                header: header.clone(),
                operations: operations.to_vec(),
                proposed_at: std::time::Instant::now(),
                already_executed: false,
                result: None,
                revert: None,
                mismatch_count: 0,
            },
        );

        info!(
            height = header.height,
            proposer = ?header.proposer[..4],
            "accepted block from peer"
        );

        // Drain any early-buffered attestations that match this block.
        let bh = block_hash(header);
        let early: Vec<(ValidatorId, u64, Hash)> = {
            let mut buf = self.early_attestations.write().unwrap();
            let matching: Vec<_> = buf
                .iter()
                .filter(|(_, h, hash, _)| *h == header.height && *hash == bh)
                .map(|(v, h, hash, _)| (*v, *h, *hash))
                .collect();
            buf.retain(|(_, h, _, _)| *h != header.height);
            matching
        };
        for (vid, height, hash) in early {
            self.accept_attestation(vid, height, hash);
        }

        true
    }

    /// Accept an attestation with signature verification.
    /// This is the preferred entry point — enforces cryptographic authentication
    /// at the consensus boundary, not just the P2P layer.
    pub fn accept_verified_attestation(
        &self,
        validator_id: ValidatorId,
        block_height: u64,
        attested_hash: Hash,
        signature: &[u8; 64],
    ) -> bool {
        // Verify signature over the domain-separated attestation payload.
        let payload =
            Self::attestation_signing_payload(self.config.chain_id, block_height, &attested_hash);
        if solen_crypto::verify(&validator_id, &payload, signature).is_err() {
            warn!(
                height = block_height,
                "attestation signature verification failed — rejecting"
            );
            return false;
        }
        self.accept_attestation(validator_id, block_height, attested_hash)
    }

    /// Domain-separated attestation payload for signing/verification.
    /// Includes chain_id to prevent cross-network replay.
    pub fn attestation_signing_payload(chain_id: u64, height: u64, block_hash: &Hash) -> Vec<u8> {
        let mut payload = Vec::with_capacity(56);
        payload.extend_from_slice(b"SOLEN_ATT"); // domain separator
        payload.extend_from_slice(&chain_id.to_le_bytes());
        payload.extend_from_slice(&height.to_le_bytes());
        payload.extend_from_slice(block_hash);
        payload
    }

    // ===================== Attestation-aware fork choice (v2) =====================
    // Gated by `config.fork_choice_v2_height`. Fixes the 2-down competing-block
    // liveness deadlock: tracks EVERY candidate block + each validator's CURRENT
    // vote (vote-change allowed), converges all nodes onto the deterministic vote
    // leader, and finalizes whichever hash reaches 2/3. Safety is unchanged: a
    // block still needs 2/3 to finalize, so two hashes can only both finalize if
    // >1/3 of validators equivocate (the standard BFT bound); no block hashes
    // change, so even a mixed-binary window cannot cause a safety fork.

    /// True iff attestation-aware fork choice is active at this height.
    pub fn fc_v2_active(&self, height: u64) -> bool {
        height >= self.config.fork_choice_v2_height
    }

    /// Drain (height, hash) pairs the node layer must broadcast as our updated
    /// attestation (our vote moved to the deterministic leader). Empty under v1.
    pub fn take_v2_revotes(&self) -> Vec<(u64, Hash)> {
        std::mem::take(&mut *self.v2_revotes.lock().unwrap())
    }

    /// Record a competing block candidate (v2), then re-evaluate the leader.
    fn v2_record_block(&self, header: &BlockHeader, operations: &[UserOperation]) {
        let height = header.height;
        let bh = block_hash(header);
        {
            let mut blocks = self.v2_blocks.write().unwrap();
            blocks
                .entry(height)
                .or_default()
                .entry(bh)
                .or_insert_with(|| PendingBlock {
                    header: header.clone(),
                    operations: operations.to_vec(),
                    proposed_at: std::time::Instant::now(),
                    already_executed: false,
                    result: None,
                    revert: None,
                    mismatch_count: 0,
                });
        }
        self.v2_reevaluate(height);
    }

    /// Record/replace a validator's vote (v2 — vote-change allowed), then
    /// re-evaluate the leader.
    fn v2_record_vote(&self, validator_id: ValidatorId, height: u64, hash: Hash) {
        {
            let mut votes = self.v2_votes.write().unwrap();
            votes.entry(height).or_default().insert(validator_id, hash);
        }
        self.v2_reevaluate(height);
    }

    /// Tally current votes for a height into hash -> voters.
    fn v2_tally(&self, height: u64) -> HashMap<Hash, Vec<ValidatorId>> {
        let votes = self.v2_votes.read().unwrap();
        let mut t: HashMap<Hash, Vec<ValidatorId>> = HashMap::new();
        if let Some(per) = votes.get(&height) {
            for (vid, h) in per {
                t.entry(*h).or_default().push(*vid);
            }
        }
        t
    }

    /// Deterministic vote leader for a height: the hash with the most attesting
    /// stake, tie-broken by the LOWEST proposer rank (so every honest node picks
    /// the same leader). Only considers candidate blocks we actually hold.
    fn v2_leader(&self, height: u64, tally: &HashMap<Hash, Vec<ValidatorId>>) -> Option<Hash> {
        let seed = self.epoch_seed();
        let vs = self.validator_set.read().unwrap();
        let order = vs.proposer_order_for_height(height, &seed);
        let blocks = self.v2_blocks.read().unwrap();
        let candidates = blocks.get(&height)?;
        let invalid = self.v2_invalid.read().unwrap();
        let invalid_at = invalid.get(&height);
        let mut best: Option<(Hash, u128, usize)> = None; // (hash, stake, rank)
        for (hash, voters) in tally {
            // Never elect a block we've proven execution-invalid, even if stray
            // votes for it linger before they're retracted/GC'd.
            if invalid_at.is_some_and(|s| s.contains(hash)) {
                continue;
            }
            // Must hold the block to be a viable leader.
            let Some(pb) = candidates.get(hash) else {
                continue;
            };
            let stake = vs.stake_of(voters);
            let rank = order
                .iter()
                .position(|id| *id == pb.header.proposer)
                .unwrap_or(usize::MAX);
            let better = match best {
                None => true,
                // Total order: stake, then proposer rank, then block hash. The final
                // hash tiebreak is REQUIRED — without it, two candidates with equal
                // stake AND equal rank (e.g. the same proposer's empty vs. non-empty
                // variant of one height) are ordered only by HashMap iteration order,
                // which is randomized per process, so different nodes elect different
                // leaders and consensus livelocks (mainnet halt 2026-08-29, block
                // 1513793). Hash is deterministic and identical fleet-wide, so this
                // makes leader selection converge.
                Some((bhash, bstake, brank)) => {
                    stake > bstake
                        || (stake == bstake && rank < brank)
                        || (stake == bstake && rank == brank && *hash < bhash)
                }
            };
            if better {
                best = Some((*hash, stake, rank));
            }
        }
        best.map(|(h, _, _)| h)
    }

    /// Re-evaluate the next-height vote leader: finalize a 2/3 winner we hold,
    /// else move our own vote toward the leader (enqueue a revote to broadcast).
    fn v2_reevaluate(&self, height: u64) {
        if height != self.height() + 1 {
            return; // only ever decide the immediate next block
        }
        let tally = self.v2_tally(height);
        if tally.is_empty() {
            return;
        }

        // 1) Finalize any hash that has reached 2/3 AND whose block we hold.
        {
            let vs = self.validator_set.read().unwrap();
            let invalid = self.v2_invalid.read().unwrap();
            let invalid_at = invalid.get(&height);
            let mut winner: Option<Hash> = None;
            for (h, voters) in &tally {
                // Never finalize a block we've proven execution-invalid, even if
                // stray votes for it still show quorum before they're retracted.
                if invalid_at.is_some_and(|s| s.contains(h)) {
                    continue;
                }
                if vs.has_quorum(voters) {
                    let have = self
                        .v2_blocks
                        .read()
                        .unwrap()
                        .get(&height)
                        .map(|m| m.contains_key(h))
                        .unwrap_or(false);
                    if have {
                        winner = Some(*h);
                        break;
                    }
                }
            }
            drop(invalid);
            drop(vs);
            if let Some(h) = winner {
                self.v2_finalize(height, h);
                return;
            }
        }

        // 2) Move our own vote to the deterministic leader if it differs, so the
        //    fleet converges. Recording our vote may itself complete quorum.
        let leader = match self.v2_leader(height, &tally) {
            Some(l) => l,
            None => return,
        };
        let our_id = self.config.validator_id;
        let we_are_validator = {
            let vs = self.validator_set.read().unwrap();
            vs.active().iter().any(|v| v.id == our_id)
        };
        if !we_are_validator {
            return;
        }
        let our_vote = self
            .v2_votes
            .read()
            .unwrap()
            .get(&height)
            .and_then(|m| m.get(&our_id))
            .copied();
        if our_vote != Some(leader) {
            self.v2_votes
                .write()
                .unwrap()
                .entry(height)
                .or_default()
                .insert(our_id, leader);
            self.v2_revotes.lock().unwrap().push((height, leader));
            // Our own move may have completed quorum for the leader.
            let voters = self.v2_tally(height).remove(&leader).unwrap_or_default();
            let has_q = { self.validator_set.read().unwrap().has_quorum(&voters) };
            if has_q {
                self.v2_finalize(height, leader);
            }
        }
    }

    /// Finalize a specific v2 candidate block by hash, reusing the legacy
    /// finalize path (execution + state-root verification + chain push).
    fn v2_finalize(&self, height: u64, hash: Hash) {
        let pb = self
            .v2_blocks
            .write()
            .unwrap()
            .get_mut(&height)
            .and_then(|m| m.remove(&hash));
        let Some(pb) = pb else { return };
        let atts: Vec<Attestation> = self
            .v2_votes
            .read()
            .unwrap()
            .get(&height)
            .map(|m| {
                m.iter()
                    .filter(|(_, h)| **h == hash)
                    .map(|(vid, _)| Attestation {
                        validator_id: *vid,
                        block_height: height,
                        block_hash: hash,
                    })
                    .collect()
            })
            .unwrap_or_default();
        self.pending_blocks.write().unwrap().insert(height, pb);
        self.pending_attestations
            .write()
            .unwrap()
            .insert(height, atts);
        self.finalize_pending_block(height);
        // Drop v2 bookkeeping at/below the finalized height.
        self.v2_blocks.write().unwrap().retain(|h, _| *h > height);
        self.v2_votes.write().unwrap().retain(|h, _| *h > height);
    }

    /// Accept an attestation (already verified by caller).
    /// Prefer accept_verified_attestation() for external inputs.
    pub fn accept_attestation(
        &self,
        validator_id: ValidatorId,
        block_height: u64,
        attested_hash: Hash,
    ) -> bool {
        // Reject attestations from non-validators to prevent memory DoS.
        {
            let vs = self.validator_set.read().unwrap();
            if !vs.active().iter().any(|v| v.id == validator_id) {
                return false;
            }
        }

        // v2: record the vote (vote-change allowed) and let fork choice converge
        // + finalize. A vote for a block we don't yet hold is still recorded, so
        // it counts the moment the block arrives — no early-attestation buffer.
        if self.fc_v2_active(block_height) {
            // Bound the height an attestation may target. Without this, a Byzantine
            // validator can sign votes for arbitrary far-future heights (up to
            // u64::MAX), each creating a new v2_votes entry that GC never reclaims
            // (v2_finalize/clear_stale_pending only prune h <= current) — unbounded
            // memory DoS that OOM-crashes every node. Fork choice only ever decides
            // height()+1, so a small look-ahead window is all that is ever useful:
            // tolerate brief gossip lead, reject anything past it or already final.
            const MAX_V2_VOTE_LOOKAHEAD: u64 = 64;
            let before = self.height();
            if block_height <= before || block_height > before + MAX_V2_VOTE_LOOKAHEAD {
                return false;
            }
            self.v2_record_vote(validator_id, block_height, attested_hash);
            return self.height() > before;
        }

        // Only accept attestations for blocks we already have in pending.
        // Accepting attestations without the block enables split-quorum attacks
        // where an attacker sends attestations for a block that doesn't exist
        // or differs from what honest nodes have.
        {
            let pending = self.pending_blocks.read().unwrap();
            match pending.get(&block_height) {
                Some(pb) => {
                    let expected_hash = block_hash(&pb.header);
                    if expected_hash != attested_hash {
                        warn!(
                            height = block_height,
                            "attestation block hash mismatch — ignoring"
                        );
                        drop(pending);
                        // Track that other validators have a different block.
                        let mut pending_w = self.pending_blocks.write().unwrap();
                        if let Some(pb) = pending_w.get_mut(&block_height) {
                            pb.mismatch_count += 1;
                        }
                        return false;
                    }
                }
                None => {
                    // We don't have this block yet. Buffer for short period
                    // to handle out-of-order gossip delivery. Bounded to prevent
                    // memory exhaustion from attestation spam.
                    const MAX_EARLY_ATTESTATIONS: usize = 200;
                    const EARLY_ATT_TTL_SECS: u64 = 10;
                    let mut buf = self.early_attestations.write().unwrap();
                    // Expire old entries.
                    buf.retain(|(_, _, _, t)| t.elapsed().as_secs() < EARLY_ATT_TTL_SECS);
                    if buf.len() < MAX_EARLY_ATTESTATIONS {
                        // Dedup before inserting.
                        if !buf
                            .iter()
                            .any(|(v, h, _, _)| *v == validator_id && *h == block_height)
                        {
                            buf.push((
                                validator_id,
                                block_height,
                                attested_hash,
                                std::time::Instant::now(),
                            ));
                        }
                    }
                    return false;
                }
            }
        }

        // Add to pending attestations.
        {
            let mut atts = self.pending_attestations.write().unwrap();
            let entry = atts.entry(block_height).or_default();

            // Don't accept duplicate attestations.
            if entry.iter().any(|a| a.validator_id == validator_id) {
                return false;
            }

            entry.push(Attestation {
                validator_id,
                block_height,
                block_hash: attested_hash,
            });
        }

        // Check if we have quorum.
        let has_quorum = {
            let atts = self.pending_attestations.read().unwrap();
            let vs = self.validator_set.read().unwrap();
            if let Some(attestations) = atts.get(&block_height) {
                let ids: Vec<ValidatorId> = attestations.iter().map(|a| a.validator_id).collect();
                vs.has_quorum(&ids)
            } else {
                false
            }
        };

        if has_quorum {
            self.finalize_pending_block(block_height);
        }

        has_quorum
    }

    /// Finalize a pending block after quorum is reached (or timeout).
    ///
    /// If we produced this block (already_executed=true), the state is already
    /// applied — just push to chain. If we received it from a peer
    /// (already_executed=false), execute now and verify state root.
    fn finalize_pending_block(&self, height: u64) {
        let pending_block = self.pending_blocks.write().unwrap().remove(&height);
        let attestations = self
            .pending_attestations
            .write()
            .unwrap()
            .remove(&height)
            .unwrap_or_default();
        let had_quorum_attestations = !attestations.is_empty();

        let Some(pb) = pending_block else { return };

        let (result, block_revert) = if pb.already_executed {
            // We produced this block — state already applied; revert was stashed.
            let result = pb.result.unwrap_or(BlockResult {
                state_root: pb.header.state_root,
                receipts: vec![],
                gas_used: 0,
            });
            (result, pb.revert.unwrap_or_default())
        } else {
            // Received from peer — execute now (Tendermint "Commit" phase),
            // committing only if our computed root matches the proposer's claim.
            // On mismatch the block is reverted (cheap reverse-delta), so the
            // store stays clean at the previous height instead of being left
            // corrupted by a half-applied divergent block.
            //
            // BUT the "execute on OUR height-1 state" assumption below only holds
            // if the store actually reflects the finalized tip. A leaked eager-
            // commit from a superseded PRODUCE attempt at THIS height (rewards at
            // an epoch boundary) leaves the store drifted, so executing the peer's
            // canonical block on it double-applies and yields a mismatch — making
            // us REJECT the canonical block and wedge (validator5, block
            // 1,344,600, 2026-08-06). Restore to the finalized tip first (no-op in
            // the common, undrifted case) so the base state is truly canonical.
            self.restore_store_to_finalized_tip();
            let exec_result = {
                let mut store = self.store.write().unwrap();
                self.executor.execute_block_checked(
                    store.as_mut(),
                    &pb.operations,
                    height,
                    &pb.header.state_root,
                )
            };

            // Mismatch handling. `accept_block` already verified this block chains
            // to our tip, so OUR height-1 state is canonical; executing the block's
            // ops on it yields the correct root, and a mismatch means the PROPOSER's
            // claimed root is objectively wrong — the block is bad, we are NOT
            // diverged. The block is reverted above, leaving our state clean at the
            // prior height.
            let Some((exec_result, revert)) = exec_result else {
                let bad = block_hash(&pb.header);
                if self.fc_v2_active(height) {
                    // v2 path: reject the bad block and let the backup proposer
                    // produce a valid one at this height, which v2 vote-change
                    // converges on. Do NOT resync — that would latch us (and, when
                    // the whole honest majority blind-attests a bad header then
                    // rejects it on execution, EVERY node at once) out of consensus
                    // via the `is_resyncing` gate in accept_block, halting the chain
                    // until an operator intervenes (mainnet 2026-07-16). A genuinely
                    // behind/forked node is still recovered by the sync path's
                    // `consecutive_sync_reverts` resync instead.
                    warn!(
                        height,
                        proposer = ?&pb.header.proposer[..4],
                        theirs = ?&pb.header.state_root[..4],
                        "state root mismatch on finalization — rejected invalid block, awaiting backup proposer"
                    );
                    self.v2_invalid
                        .write()
                        .unwrap()
                        .entry(height)
                        .or_default()
                        .insert(bad);
                    // Retract our own vote for the bad hash so it loses our quorum
                    // weight; every honest node does the same, so it can never be
                    // re-selected as the v2 leader.
                    if let Some(m) = self.v2_votes.write().unwrap().get_mut(&height) {
                        if m.get(&self.config.validator_id) == Some(&bad) {
                            m.remove(&self.config.validator_id);
                        }
                    }
                    return;
                }
                // Pre-v2: without vote-change the fleet cannot converge on a backup
                // block, so keep the historical resync behaviour.
                warn!(
                    height,
                    proposer = ?&pb.header.proposer[..4],
                    theirs = ?&pb.header.state_root[..4],
                    "state root mismatch on finalization — reverted, requesting snapshot resync"
                );
                self.needs_resync
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                return;
            };

            (exec_result, revert)
        };

        let block = FinalizedBlock {
            header: pb.header.clone(),
            result,
            attestations,
            operations: pb.operations,
        };

        self.chain.write().unwrap().push(block.clone());
        self.rollback_journal
            .write()
            .unwrap()
            .record(height, block_revert);
        // Canonical now — drop any durable eager-revert for this height.
        self.clear_eager_revert(height);
        self.persist_block_and_meta(&block);
        self.emit_block_events(&block);
        // We advanced past `height`; execution-invalid markers at/below it are now
        // obsolete. (Only reached on a SUCCESSFUL finalize — the mismatch branch
        // returns early, so the marker it just set survives while we stay put.)
        self.v2_invalid.write().unwrap().retain(|h, _| *h > height);

        // Remove finalized operations from our mempool to prevent
        // re-including already-executed transactions in future blocks.
        self.mempool.remove_finalized(&block.operations);

        // Track missed blocks for downtime slashing.
        // If the actual proposer differs from the designated proposer,
        // the designated one missed their slot.
        {
            let seed = self.epoch_seed();
            let vs = self.validator_set.read().unwrap();
            let designated = vs.proposer_for_height_with_seed(height, &seed);
            drop(vs);

            if let Some(designated_id) = designated {
                if designated_id != block.header.proposer {
                    let mut vs = self.validator_set.write().unwrap();
                    if let Some(evidence) =
                        crate::slashing::record_missed_block(&mut vs, &designated_id)
                    {
                        drop(vs);
                        self.process_slashing(&evidence);
                    }
                } else {
                    // Proposer produced on time — reset missed counter.
                    let mut vs = self.validator_set.write().unwrap();
                    if let Some(v) = vs.get_mut(&designated_id) {
                        v.missed_blocks = 0;
                    }
                }
            }
        }

        self.try_epoch_transition(height);

        // Only reset force-finalization counter if we actually had quorum
        // (attestations present). Force-finalized blocks have no attestations.
        if had_quorum_attestations {
            self.consecutive_force_finalizes
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }

        info!(
            height,
            epoch = pb.header.epoch,
            "block finalized with quorum"
        );
    }

    /// Force-finalize a pending block immediately. Used after sync when
    /// accepting a live block that the network has already agreed on —
    /// no need to wait for attestations.
    pub fn force_finalize_block(&self, height: u64) {
        self.finalize_pending_block(height);
    }

    /// Finalize a pending block ONLY when a 2/3 attestation quorum exists for
    /// it (or in single-validator / early-bootstrap modes). Returns true if it
    /// finalized. The sync path uses this so a single peer's block cannot be
    /// committed on a syncing node without real network agreement — without the
    /// quorum gate, one Byzantine peer could plant arbitrary state. When quorum
    /// is not yet present the block stays pending and is finalized later through
    /// the normal quorum-gated timeout path (or the node re-syncs).
    pub fn force_finalize_block_if_quorum(&self, height: u64) -> bool {
        let has_quorum = {
            let atts = self.pending_attestations.read().unwrap();
            let att_ids: Vec<ValidatorId> = atts
                .get(&height)
                .map(|a| a.iter().map(|att| att.validator_id).collect())
                .unwrap_or_default();
            let vs = self.validator_set.read().unwrap();
            // Single-validator chains and the first few bootstrap blocks finalize
            // without quorum (peers need time to connect), matching the
            // finalize_timed_out_blocks bootstrap carve-out.
            vs.active_count() <= 1 || height <= 3 || vs.has_quorum(&att_ids)
        };
        if has_quorum {
            self.finalize_pending_block(height);
        }
        has_quorum
    }

    /// Legacy rewards distribution — now handled by the executor via
    /// execute_block_with_height at epoch boundaries. Kept for reference only.
    #[allow(dead_code)]
    fn _distribute_epoch_rewards_legacy(&self) {
        // Distribute staking rewards from the staking pool account.
        // ~317 SOLEN per epoch (50M/year ÷ 157,680 epochs), with 8 decimals.
        let reward_per_epoch: u128 = 31_700_000_000; // 317 SOLEN in base units
        let mut store = self.store.write().unwrap();

        // Load staking state.
        let staking = solen_system_contracts::staking::StakingContract::load(store.as_ref());

        let active = staking.active_validators();
        let total_stake = staking.total_active_stake();

        if total_stake == 0 || active.is_empty() {
            return;
        }

        let total_reward = reward_per_epoch;

        // Check staking pool balance — rewards stop when pool is empty.
        let pool_key = {
            let mut k = b"acc/".to_vec();
            k.extend_from_slice(&solen_types::system::STAKING_POOL_ADDRESS);
            k
        };

        let pool_balance = if let Ok(Some(data)) = store.get(&pool_key) {
            if let Ok(account) =
                <solen_types::account::Account as borsh::BorshDeserialize>::try_from_slice(&data)
            {
                account.balance
            } else {
                0
            }
        } else {
            0
        };

        if pool_balance == 0 {
            let epoch = self.epoch_manager.read().unwrap().current_epoch;
            info!(epoch, "staking pool exhausted — no rewards this epoch");
            return;
        }

        // Cap reward to available pool balance.
        let actual_reward = total_reward.min(pool_balance);

        // Deduct from staking pool.
        if let Ok(Some(data)) = store.get(&pool_key) {
            if let Ok(mut pool_account) =
                <solen_types::account::Account as borsh::BorshDeserialize>::try_from_slice(&data)
            {
                pool_account.balance = pool_account.balance.saturating_sub(actual_reward);
                if let Ok(encoded) = borsh::to_vec(&pool_account) {
                    let _ = store.put(&pool_key, &encoded);
                }
            }
        }

        // Distribute to validators and delegators proportionally.
        let mut reward_events = Vec::new();

        for validator in &active {
            let validator_share = actual_reward * validator.total_stake() / total_stake;
            if validator_share == 0 {
                continue;
            }

            // Split between validator (self-stake + commission) and delegators.
            let delegator_pool = if validator.total_delegated > 0 {
                validator_share * validator.total_delegated / validator.total_stake()
            } else {
                0
            };
            let commission = delegator_pool * validator.commission_rate_bps as u128 / 10_000;
            let delegator_net = delegator_pool.saturating_sub(commission);

            // Validator gets: self-stake share + commission from delegators.
            let validator_reward = validator_share.saturating_sub(delegator_pool) + commission;

            // Credit validator account.
            credit_account(store.as_mut(), &validator.id, validator_reward);

            let mut event_data = Vec::with_capacity(48);
            event_data.extend_from_slice(&validator.id);
            event_data.extend_from_slice(&validator_reward.to_le_bytes());
            reward_events.push(solen_execution::receipt::Event {
                emitter: solen_types::system::STAKING_POOL_ADDRESS,
                topic: b"epoch_reward".to_vec(),
                data: event_data,
            });

            // Distribute remaining rewards to delegators proportionally.
            if delegator_net > 0 && validator.total_delegated > 0 {
                let delegations = staking.delegations_for_validator(&validator.id);
                for delegation in delegations {
                    let del_share = delegator_net * delegation.amount / validator.total_delegated;
                    if del_share == 0 {
                        continue;
                    }

                    credit_account(store.as_mut(), &delegation.delegator, del_share);

                    let mut event_data = Vec::with_capacity(48);
                    event_data.extend_from_slice(&delegation.delegator);
                    event_data.extend_from_slice(&del_share.to_le_bytes());
                    reward_events.push(solen_execution::receipt::Event {
                        emitter: solen_types::system::STAKING_POOL_ADDRESS,
                        topic: b"delegator_reward".to_vec(),
                        data: event_data,
                    });
                }
            }
        }

        // Create a synthetic receipt for the reward distribution.
        if !reward_events.is_empty() {
            let receipt = solen_execution::receipt::ExecutionReceipt {
                sender: solen_types::system::STAKING_POOL_ADDRESS,
                nonce: 0,
                success: true,
                gas_used: 0,
                error: None,
                events: reward_events,
                auth_method: "system".to_string(),
            };
            self.pending_reward_receipts.write().unwrap().push(receipt);
        }

        let epoch = self.epoch_manager.read().unwrap().current_epoch;
        info!(
            epoch,
            validators = active.len(),
            reward = actual_reward,
            pool_remaining = pool_balance.saturating_sub(actual_reward),
            "epoch rewards distributed from staking pool"
        );
    }

    /// Queue slashing evidence to be included in the next block as a
    /// deterministic system operation. All validators will execute the
    /// slash identically as part of block execution.
    fn process_slashing(&self, evidence: &crate::slashing::SlashingEvidence) {
        // Reset only the offender's missed-block COUNTER so we don't regenerate
        // the same evidence every block. Do NOT mutate stake or status here:
        // those are consensus-visible (they change proposer selection and the
        // quorum denominator), and each node would apply them at a slightly
        // different moment, so the fleet would diverge on the validator set and
        // compute different proposers → competing blocks → fork (the 2026-06-24
        // cascade). The penalty + jailing apply deterministically via the queued
        // on-chain slash below: it executes identically in every node's block,
        // and the consensus set picks up the resulting `is_active=false` at the
        // next epoch sync (removing the offender for everyone at once).
        {
            let mut vs = self.validator_set.write().unwrap();
            if let Some(v) = vs.get_mut(&evidence.offender) {
                v.missed_blocks = 0;
            }
        }

        // Queue for deterministic on-chain execution in the next block.
        let mut queue = self.pending_slashing.lock().unwrap();
        // Dedup: don't queue the same offender twice.
        if !queue.iter().any(|e| e.offender == evidence.offender) {
            warn!(
                validator = ?&evidence.offender[..4],
                reason = ?evidence.reason,
                "slashing evidence queued for next block"
            );
            queue.push(evidence.clone());
        }
    }

    /// Persist a finalized block and update chain metadata atomically.
    fn persist_block_and_meta(&self, block: &FinalizedBlock) {
        let key = format!("block/{}", block.header.height);
        if let Ok(data) = serde_json::to_vec(&SerializableBlock::from(block)) {
            let mut store = self.store.write().unwrap();
            // Write block data and chain metadata together.
            if let Err(e) = store.put(key.as_bytes(), &data) {
                warn!(height = block.header.height, error = %e, "failed to persist block");
                return;
            }
            save_chain_meta(store.as_mut(), block.header.height, block.header.epoch);
        }

        // Prune old blocks beyond retention window.
        self.prune_old_blocks(block.header.height);
    }

    /// Remove blocks older than the retention window (opt-in).
    fn prune_old_blocks(&self, current_height: u64) {
        if !self.config.prune {
            return; // Default: archive mode, keep everything.
        }
        const BLOCK_RETENTION: u64 = 10_000_000;
        if current_height <= BLOCK_RETENTION {
            return;
        }
        let prune_below = current_height - BLOCK_RETENTION;
        // Prune in small batches to avoid holding the lock too long.
        let mut store = self.store.write().unwrap();
        for h in (prune_below.saturating_sub(10))..prune_below {
            if h == 0 {
                continue;
            }
            let key = format!("block/{}", h);
            let _ = store.delete(key.as_bytes());
        }
    }

    /// Load persisted blocks from the state store (for indexer replay).
    /// Loads at most `max_blocks` starting from `from_height`.
    pub fn load_persisted_blocks_range(
        &self,
        from_height: u64,
        max_blocks: usize,
    ) -> Vec<FinalizedBlock> {
        let store = self.store.read().unwrap();
        let mut blocks = Vec::new();
        let mut height = from_height;

        while blocks.len() < max_blocks {
            let key = format!("block/{}", height);
            match store.get(key.as_bytes()) {
                Ok(Some(data)) => {
                    if let Ok(sb) = serde_json::from_slice::<SerializableBlock>(&data) {
                        blocks.push(sb.into());
                        height += 1;
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        blocks
    }

    /// Load all persisted blocks (convenience wrapper, capped at current height).
    pub fn load_persisted_blocks(&self) -> Vec<FinalizedBlock> {
        let max = self.height() as usize;
        self.load_persisted_blocks_range(1, max)
    }

    /// Get blocks for sync — loads from persistent storage.
    /// Returns up to `max_blocks` starting from `from_height`.
    pub fn get_blocks_for_sync(&self, from_height: u64, max_blocks: usize) -> Vec<FinalizedBlock> {
        let store = self.store.read().unwrap();
        let mut blocks = Vec::new();
        let mut height = from_height;

        while blocks.len() < max_blocks {
            let key = format!("block/{}", height);
            match store.get(key.as_bytes()) {
                Ok(Some(data)) => {
                    if let Ok(sb) = serde_json::from_slice::<SerializableBlock>(&data) {
                        blocks.push(sb.into());
                        height += 1;
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        blocks
    }

    /// Replay a synced block: execute operations and finalize.
    /// Used during initial sync from peers.
    ///
    /// If `synced_receipts` are provided (from the peer's persisted block),
    /// they are used for indexing instead of the re-execution receipts.
    /// This preserves transaction history during sync.
    /// Replay a synced block. Returns `true` if applied, `false` if skipped (gap or duplicate).
    pub fn replay_synced_block(
        &self,
        header: &BlockHeader,
        operations: &[UserOperation],
        synced_receipts: Vec<solen_execution::receipt::ExecutionReceipt>,
    ) -> bool {
        let height = header.height;

        // Reject if we already have this block or there's a gap.
        {
            let chain = self.chain.read().unwrap();
            if let Some(last) = chain.last() {
                if height <= last.header.height {
                    return false; // Already have this height.
                }
                if height != last.header.height + 1 {
                    debug!(
                        height,
                        our_height = last.header.height,
                        "sync block has gap — skipping"
                    );
                    return false; // Gap — can't apply.
                }
                // Note: we intentionally do NOT check parent_hash during sync.
                // Synced blocks are authoritative from peers. A wiped node may
                // have produced its own divergent blocks during startup, so the
                // parent hashes won't match. The gap check above ensures blocks
                // are applied sequentially, and state root verification on the
                // next live block confirms correctness.
            }
        }

        // Validate epoch matches expected value.
        let expected_epoch = height / crate::epoch::EPOCH_LENGTH;
        if header.epoch != expected_epoch {
            warn!(
                height,
                expected = expected_epoch,
                got = header.epoch,
                "invalid epoch in synced block — skipping"
            );
            return false;
        }

        // C-02: authenticate the synced block before applying it. The sync path
        // is otherwise unauthenticated — the only integrity gate is the state-root
        // match below, which an attacker trivially satisfies by computing the root
        // of their OWN forged operations. Combined with unsolicited SyncBlocks
        // broadcasts, that lets a peer inject an attacker-crafted, fully-finalized
        // fake chain (eclipse / forgery). Requiring the header to be signed by an
        // active validator (proposer in the set + valid proposer signature over
        // the block hash) means a forger needs a validator's private key. Honest
        // sync is unaffected — real blocks carry valid proposer signatures, which
        // travel in the serialized header. Gated by config so it can be activated
        // fleet-wide deliberately. (Sync deliberately skips the parent-hash and
        // quorum checks that accept_block does; proposer-signature authentication
        // is the forgery defense available without a directed-sync/attestation
        // protocol — see the remediation notes.)
        if self.config.authenticate_sync_blocks && !self.header_is_validator_signed(header) {
            warn!(
                height,
                proposer = ?&header.proposer[..4],
                "unauthenticated synced block (bad/absent proposer signature or non-validator) — rejecting"
            );
            return false;
        }

        // Execute on the real store but commit ONLY if the state root matches.
        // On mismatch (different fork) the block is reverted, leaving our state
        // untouched, so a wrong-fork block can't corrupt us — the state
        // self-corrects when the correct chain's blocks arrive. Previously the
        // mismatching block was committed and left in place, which permanently
        // diverged the node and forced a full-snapshot re-download to recover.
        //
        // Restore to the finalized tip first (no-op when undrifted): a leaked
        // eager-commit from a superseded produce at this height would otherwise
        // make the canonical synced block mismatch and be rejected, driving a
        // needless resync loop (the validator5 wedge class).
        self.restore_store_to_finalized_tip();
        let exec_result = {
            let mut store = self.store.write().unwrap();
            self.executor.execute_block_checked(
                store.as_mut(),
                operations,
                height,
                &header.state_root,
            )
        };

        let Some((exec_result, revert)) = exec_result else {
            // We received a canonical block at our next height but couldn't apply
            // it (root mismatch — our tip diverges). One such revert can be a
            // transient bad-fork block from a minority peer; but if we revert
            // CONSECUTIVE canonical blocks at our next height, our committed tip
            // is genuinely forked and normal sync can never advance us — we're
            // stranded (validator6's 2026-06-25 case). Trigger a resync so the
            // tiered recovery rolls us back to the common ancestor and forward.
            let n = self
                .consecutive_sync_reverts
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            debug!(
                height,
                consecutive = n,
                theirs = ?&header.state_root[..4],
                "state root mismatch in synced block — reverted, rejecting"
            );
            if n >= SYNC_REVERT_RESYNC_THRESHOLD && !self.is_resyncing() {
                warn!(
                    height,
                    consecutive = n,
                    "stranded on a forked tip (canonical blocks won't apply) — requesting resync"
                );
                self.needs_resync
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                self.consecutive_sync_reverts
                    .store(0, std::sync::atomic::Ordering::Relaxed);
            }
            return false;
        };
        // A synced block applied cleanly — we're tracking canonical, not stranded.
        self.consecutive_sync_reverts
            .store(0, std::sync::atomic::Ordering::Relaxed);

        // Use synced receipts if available (they include user tx events).
        // Fall back to execution receipts (which only have epoch rewards).
        let receipts = if !synced_receipts.is_empty() {
            synced_receipts
        } else {
            exec_result.receipts
        };

        let result = BlockResult {
            state_root: exec_result.state_root,
            receipts,
            gas_used: exec_result.gas_used,
        };

        let block = FinalizedBlock {
            header: header.clone(),
            result,
            attestations: vec![],
            operations: operations.to_vec(),
        };

        self.chain.write().unwrap().push(block.clone());
        self.rollback_journal
            .write()
            .unwrap()
            .record(height, revert);
        // Canonical now — drop any durable eager-revert for this height.
        self.clear_eager_revert(height);
        self.persist_block_and_meta(&block);

        // Advance epoch counter (rewards already handled by executor).
        self.try_epoch_transition(height);
        true
    }

    /// Process epoch transition if at a boundary. Syncs the consensus
    /// validator set with the staking contract so new validators are
    /// included in proposer rotation and quorum calculations.
    fn try_epoch_transition(&self, height: u64) {
        let mut em = self.epoch_manager.write().unwrap();
        if em.is_epoch_boundary(height) {
            let mut vs = self.validator_set.write().unwrap();
            em.process_epoch_transition(&mut vs);

            // Sync new validators from staking contract into consensus set.
            let staking = {
                let store = self.store.read().unwrap();
                solen_system_contracts::staking::StakingContract::load(store.as_ref())
            };

            // Build the set of active staking validators.
            let active_ids: std::collections::HashSet<_> = staking
                .validators
                .iter()
                .filter(|sv| sv.is_active)
                .map(|sv| sv.id)
                .collect();

            // Limit validator set changes to 1/3 of current total stake per epoch.
            // This preserves BFT safety — the validator set can't change faster
            // than honest validators can detect and respond.
            // Exception: during bootstrapping (< 4 validators), allow unlimited changes.
            let active_count = vs.active_count();
            let current_total_stake: u128 = vs
                .all()
                .iter()
                .filter(|v| v.is_active())
                .map(|v| v.stake)
                .sum();
            let max_stake_change = if active_count >= 4 {
                current_total_stake / 3
            } else {
                u128::MAX
            };
            let mut stake_changed: u128 = 0;

            // Add new validators and reactivate unjailed ones.
            for sv in &staking.validators {
                if !sv.is_active {
                    continue;
                }
                if let Some(v) = vs.get_mut(&sv.id) {
                    // Update stake and reactivate if unjailed on-chain.
                    let old_stake = v.stake;
                    v.stake = sv.total_stake();
                    stake_changed = stake_changed
                        .saturating_add((v.stake as i128 - old_stake as i128).unsigned_abs());
                    if !v.is_active() {
                        v.status = crate::validator::ValidatorStatus::Active;
                        v.missed_blocks = 0;
                        tracing::info!(
                            validator = ?&sv.id[..4],
                            stake = sv.total_stake(),
                            "validator reactivated (unjailed)"
                        );
                    }
                } else {
                    // New validator — check if adding them exceeds the change limit.
                    let new_stake = sv.total_stake();
                    if stake_changed.saturating_add(new_stake) > max_stake_change
                        && max_stake_change > 0
                    {
                        tracing::warn!(
                            validator = ?&sv.id[..4],
                            stake = new_stake,
                            "validator set change limit reached — deferring to next epoch"
                        );
                        continue;
                    }
                    stake_changed = stake_changed.saturating_add(new_stake);
                    let new_info = crate::validator::ValidatorInfo::new(sv.id, sv.total_stake());
                    vs.add(new_info);
                    tracing::info!(
                        validator = ?&sv.id[..4],
                        stake = sv.total_stake(),
                        "new validator joined consensus set"
                    );
                }
            }

            // Remove validators that exited from staking (also bounded by change limit).
            let to_remove: Vec<_> = vs
                .all()
                .iter()
                .filter(|v| !active_ids.contains(&v.id))
                .map(|v| (v.id, v.stake))
                .collect();
            for (id, stake) in &to_remove {
                if stake_changed.saturating_add(*stake) > max_stake_change && max_stake_change > 0 {
                    tracing::warn!(
                        validator = ?&id[..4],
                        "validator removal deferred — change limit reached"
                    );
                    continue;
                }
                stake_changed = stake_changed.saturating_add(*stake);
                vs.remove(id);
                tracing::info!(
                    validator = ?&id[..4],
                    "validator removed from consensus set (exited staking)"
                );
            }

            // Emit validator set changed event.
            let _ = self.event_tx.send(NodeEvent::ValidatorSetChanged {
                epoch: em.current_epoch,
                active_count: vs.active_count(),
            });

            // Propose a new checkpoint at the epoch boundary.
            // The block at this height becomes the checkpoint candidate.
            {
                let chain = self.chain.read().unwrap();
                if let Some(last_block) = chain.last() {
                    let bh = block_hash(&last_block.header);
                    let mut cp_store = self.finalized_checkpoints.write().unwrap();
                    cp_store.propose_checkpoint(
                        last_block.header.height,
                        em.current_epoch,
                        bh,
                        last_block.header.state_root,
                    );
                    info!(
                        height = last_block.header.height,
                        epoch = em.current_epoch,
                        "checkpoint proposed at epoch boundary"
                    );
                }
            }

            // Drop validator_set write lock before checkpoint attestation.
            // attest_checkpoint needs validator_set read lock — holding write would deadlock.
            drop(vs);

            // Self-attest the checkpoint if we're a validator.
            {
                let cp_store = self.finalized_checkpoints.read().unwrap();
                if let Some(ref pending) = cp_store.pending {
                    let msg = crate::checkpoint::FinalizedCheckpointStore::signing_message(
                        pending.height,
                        &pending.block_hash,
                        &pending.state_root,
                    );
                    drop(cp_store);
                    if let Some(ref kp) = self.signing_keypair {
                        let sig = kp.sign(&msg).to_vec();
                        self.attest_checkpoint(self.config.validator_id, sig);
                    }
                }
            }

            // Update epoch seed for randomized proposer selection.
            // Seed = blake3(state_root of the epoch's last block). MUST use an
            // agreed, canonical on-chain value: block_hash() includes proposer +
            // timestamp_ms, which legitimately differ across state-equivalent
            // re-proposals / force-finalizations (slashing.rs "same logical block"
            // rule), so a block_hash-derived seed diverges across nodes at the
            // boundary → different proposer schedules → chain halts. state_root is
            // identical on every honest node.
            let chain = self.chain.read().unwrap();
            if let Some(last_block) = chain.last() {
                let new_seed = if last_block.header.height >= self.config.epoch_randomness_height {
                    // Grind-resistant path (variant B): read the seed the
                    // accumulator committed to state this block. Sourced from
                    // agreed on-chain state → identical on every node and
                    // restart-safe (no in-memory reset, no history lookback).
                    let store = self.store.read().unwrap();
                    solen_system_contracts::epoch_randomness::EpochRandomness::load(store.as_ref())
                        .seed
                } else {
                    // Legacy path: blake3 of the boundary block's agreed state_root.
                    solen_crypto::blake3_hash(&last_block.header.state_root)
                };
                *self.epoch_seed.write().unwrap() = new_seed;
                tracing::info!(
                    epoch = em.current_epoch,
                    seed = ?&new_seed[..4],
                    "epoch seed updated for proposer selection"
                );
            }
        }
    }

    /// Check if a block is pending at the given height (proposed but not finalized).
    /// Clear all pending blocks and attestations at or below the given height.
    /// Called after sync to prevent stale blocks from being force-finalized.
    pub fn clear_stale_pending(&self, current_height: u64) {
        let mut pending = self.pending_blocks.write().unwrap();
        let before = pending.len();
        pending.retain(|h, _| *h > current_height);
        let mut atts = self.pending_attestations.write().unwrap();
        atts.retain(|h, _| *h > current_height);
        // v2 fork-choice candidates + votes at/below the synced height are stale.
        self.v2_blocks
            .write()
            .unwrap()
            .retain(|h, _| *h > current_height);
        self.v2_votes
            .write()
            .unwrap()
            .retain(|h, _| *h > current_height);
        self.v2_invalid
            .write()
            .unwrap()
            .retain(|h, _| *h > current_height);
        let cleared = before - pending.len();
        if cleared > 0 {
            info!(
                cleared,
                current_height, "cleared stale pending blocks after sync"
            );
        }
    }

    pub fn has_pending_block(&self, height: u64) -> bool {
        self.pending_blocks.read().unwrap().contains_key(&height)
    }

    /// Force-finalize any pending blocks that have been waiting longer than
    /// the timeout. This prevents the chain from stalling when validators
    /// are offline and quorum can't be reached.
    pub fn finalize_timed_out_blocks(&self, timeout: std::time::Duration) -> usize {
        let current_height = self.height();

        // First, discard any pending blocks that are at or below the current
        // chain height. These are stale from before a sync and must never be
        // finalized — doing so would push an old block to the end of the
        // chain vector, effectively rolling the node backwards.
        {
            let mut pending = self.pending_blocks.write().unwrap();
            let before = pending.len();
            pending.retain(|h, _| *h > current_height);
            let discarded = before - pending.len();
            if discarded > 0 {
                info!(discarded, current_height, "discarded stale pending blocks");
            }
        }

        let stale_heights: Vec<u64> = {
            let blocks = self.pending_blocks.read().unwrap();
            blocks
                .iter()
                .filter(|(_, pb)| pb.proposed_at.elapsed() > timeout)
                .map(|(h, _)| *h)
                .collect()
        };

        let mut count = 0;
        for height in stale_heights {
            // Double-check: only finalize the NEXT expected block.
            if height != self.height() + 1 {
                debug!(
                    height,
                    our_height = self.height(),
                    "skipping stale pending block"
                );
                continue;
            }

            // Before force-finalizing, check if the pending block was proposed by us
            // and we've seen attestation mismatches. If other validators are attesting
            // to a different block at this height, drop ours and wait for sync —
            // this prevents divergent force-finalization.
            let should_drop = {
                let pending = self.pending_blocks.read().unwrap();
                if let Some(pb) = pending.get(&height) {
                    pb.header.proposer == self.config.validator_id && pb.mismatch_count > 0
                } else {
                    false
                }
            };

            if should_drop {
                info!(
                    height,
                    "dropping own block — other validators have a different block at this height"
                );
                self.pending_blocks.write().unwrap().remove(&height);
                // Signal that we need sync by storing the dropped height.
                *self.dropped_block_height.write().unwrap() = Some(height);
                continue;
            }

            // Before force-finalizing, check if attesting validators have enough
            // stake for quorum. Uses the same 2/3+ stake threshold as normal
            // finalization. This prevents minority-stake partitions from
            // force-finalizing divergent chains.
            {
                let atts = self.pending_attestations.read().unwrap();
                let att_ids: Vec<_> = atts
                    .get(&height)
                    .map(|a| a.iter().map(|att| att.validator_id).collect())
                    .unwrap_or_default();
                let att_count = att_ids.len();
                let vs = self.validator_set.read().unwrap();
                let active_count = vs.active_count();
                let has_quorum_stake = vs.has_quorum(&att_ids);
                drop(vs);
                drop(atts);

                // Only force-finalize if attesting validators hold 2/3+ stake,
                // OR if we're in single-validator mode.
                // Allow the first few blocks (height <= 3) without quorum to bootstrap
                // the chain — peers need time to connect and exchange blocks.
                if active_count > 1 && !has_quorum_stake && height > 3 {
                    let force_count = self
                        .consecutive_force_finalizes
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        + 1;
                    if force_count > 2 {
                        warn!(
                            height,
                            attestations = att_count,
                            active_validators = active_count,
                            "partition detected — insufficient stake for quorum, stopping production"
                        );
                        self.pending_blocks.write().unwrap().remove(&height);
                    } else {
                        debug!(
                            height,
                            attestations = att_count,
                            active_validators = active_count,
                            "waiting for quorum stake before force-finalizing"
                        );
                    }
                    continue;
                }
            }

            let force_count = self
                .consecutive_force_finalizes
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;

            // If we've force-finalized too many blocks in a row, we're likely
            // partitioned from the network. Stop finalizing to prevent divergence.
            const MAX_CONSECUTIVE_FORCE_FINALIZES: u32 = 3;
            if force_count > MAX_CONSECUTIVE_FORCE_FINALIZES {
                warn!(
                    height,
                    force_count,
                    "too many consecutive force-finalizations — stopping production (likely partitioned)"
                );
                self.pending_blocks.write().unwrap().remove(&height);
                continue;
            }

            warn!(
                height,
                force_count, "quorum timeout — force-finalizing block"
            );
            self.finalize_pending_block(height);
            count += 1;
        }

        // Clean up orphaned attestations for heights already finalized.
        let current_height = self.height();
        let mut atts = self.pending_attestations.write().unwrap();
        atts.retain(|h, _| *h > current_height);

        count
    }

    /// Number of active validators.
    pub fn active_validator_count(&self) -> usize {
        self.validator_set.read().unwrap().active_count()
    }

    /// Run the block production loop. In multi-validator mode, only
    /// proposes when it's this node's turn.
    pub async fn run(&self, cancel: tokio::sync::watch::Receiver<bool>) {
        let mut tick = interval(Duration::from_millis(self.config.block_time_ms));

        let is_single_validator = {
            let vs = self.validator_set.read().unwrap();
            info!(
                block_time_ms = self.config.block_time_ms,
                validators = vs.active_count(),
                total_stake = %vs.total_active_stake(),
                "consensus engine started"
            );
            vs.active_count() <= 1
        };

        loop {
            tick.tick().await;

            if *cancel.borrow() {
                info!("consensus engine stopping");
                break;
            }

            if is_single_validator {
                // Single validator: always produce (devnet mode).
                self.produce_block();
            } else if self.is_next_proposer() {
                // Multi-validator: only produce when it's our turn.
                self.produce_block();
            }
            // Otherwise: wait for blocks from the proposer via accept_block().
        }
    }
}

/// Hash a block header to get the block hash.
/// Panics if serialization fails — a block header must always be serializable.
/// A fallback hash would create collisions between blocks at the same height.
/// Hash a block header. The proposer_signature is excluded so that the
/// hash is the same whether or not the block is signed — this prevents
/// consensus breaks during rolling upgrades where some validators sign
/// and others don't yet.
pub fn block_hash(header: &BlockHeader) -> Hash {
    let mut h = header.clone();
    h.proposer_signature = vec![];
    let data = serde_json::to_vec(&h).expect("block header serialization must not fail");
    blake3_hash(&data)
}

fn compute_tx_root(ops: &[solen_types::transaction::UserOperation]) -> Hash {
    if ops.is_empty() {
        return [0u8; 32];
    }
    match serde_json::to_vec(ops) {
        Ok(data) => blake3_hash(&data),
        Err(e) => {
            warn!(error = %e, "tx serialization failed — using op count hash");
            blake3_hash(&ops.len().to_le_bytes())
        }
    }
}

fn compute_receipts_root(result: &BlockResult) -> Hash {
    if result.receipts.is_empty() {
        return [0u8; 32];
    }
    match serde_json::to_vec(&result.receipts) {
        Ok(data) => blake3_hash(&data),
        Err(e) => {
            warn!(error = %e, "receipts serialization failed — using count hash");
            blake3_hash(&result.receipts.len().to_le_bytes())
        }
    }
}

/// Credit an account balance by the given amount.
fn credit_account(store: &mut dyn StateStore, account_id: &[u8; 32], amount: u128) {
    let key = {
        let mut k = b"acc/".to_vec();
        k.extend_from_slice(account_id);
        k
    };

    if let Ok(Some(data)) = store.get(&key) {
        if let Ok(mut account) =
            <solen_types::account::Account as borsh::BorshDeserialize>::try_from_slice(&data)
        {
            account.balance = account.balance.saturating_add(amount);
            if let Ok(encoded) = borsh::to_vec(&account) {
                let _ = store.put(&key, &encoded);
            }
        }
    }
}

/// Replay DURABLE eager-revert records for every height strictly above
/// `tip_height`, newest-first, undoing leaked eager-commits so the store
/// reflects the true finalized-tip state. Returns true if any record applied.
///
/// Each record was written atomically with the eager writes it undoes (see
/// `executor::execute_block_journaled_durable`) and reverts that block back to
/// the finalized tip; a record's own key is deleted by the revert it encodes.
/// Records at heights <= `tip_height` are canonical (finalized) and left alone.
/// This is the restart-safe repair for the mainnet validator5 wedge (2026-08-06).
fn apply_durable_reverts_above(store: &mut dyn StateStore, tip_height: u64) -> bool {
    let mut pending: Vec<(u64, solen_execution::executor::BlockRevert)> = Vec::new();
    let entries = match store.scan_prefix(solen_storage::EAGER_REVERT_PREFIX) {
        Ok(e) => e,
        Err(e) => {
            warn!(error = %e, "failed to scan durable eager-revert records");
            return false;
        }
    };
    for (k, v) in entries {
        let suffix = &k[solen_storage::EAGER_REVERT_PREFIX.len()..];
        if suffix.len() != 8 {
            continue;
        }
        let mut h = [0u8; 8];
        h.copy_from_slice(suffix);
        let height = u64::from_be_bytes(h);
        if height <= tip_height {
            continue; // finalized/canonical — must not undo
        }
        if let Some(revert) = solen_execution::executor::BlockExecutor::decode_eager_revert(&v) {
            pending.push((height, revert));
        }
    }
    if pending.is_empty() {
        return false;
    }
    // Newest-first. In practice there is at most one (we always produce on the
    // tip), but this is correct if several ever stacked.
    pending.sort_by(|a, b| b.0.cmp(&a.0));
    for (height, revert) in &pending {
        if let Err(e) = store.apply_batch_atomic(revert, true) {
            warn!(height = *height, error = %e, "failed to apply durable eager-revert");
        }
    }
    store.commit_root();
    info!(
        count = pending.len(),
        tip = tip_height,
        "replayed durable eager-revert record(s) to repair store drift"
    );
    true
}

/// Key for persisted chain metadata.
const CHAIN_META_KEY: &[u8] = b"__chain_meta__";

/// Persist chain height and epoch to the state store.
fn save_chain_meta(store: &mut dyn StateStore, height: u64, epoch: u64) {
    let mut data = Vec::with_capacity(16);
    data.extend_from_slice(&height.to_le_bytes());
    data.extend_from_slice(&epoch.to_le_bytes());
    let _ = store.put(CHAIN_META_KEY, &data);
}

/// Load chain height and epoch from the state store.
fn load_chain_meta(store: &dyn StateStore) -> (u64, u64) {
    match store.get(CHAIN_META_KEY) {
        Ok(Some(data)) if data.len() >= 16 => {
            let mut h = [0u8; 8];
            let mut e = [0u8; 8];
            h.copy_from_slice(&data[..8]);
            e.copy_from_slice(&data[8..16]);
            (u64::from_le_bytes(h), u64::from_le_bytes(e))
        }
        _ => (0, 0),
    }
}

/// Serializable block for persistence (BlockResult doesn't derive Serialize).
#[derive(serde::Serialize, serde::Deserialize)]
struct SerializableBlock {
    header: BlockHeader,
    state_root: [u8; 32],
    receipts: Vec<solen_execution::receipt::ExecutionReceipt>,
    gas_used: u64,
    #[serde(default)]
    operations: Vec<UserOperation>,
}

impl From<&FinalizedBlock> for SerializableBlock {
    fn from(b: &FinalizedBlock) -> Self {
        Self {
            header: b.header.clone(),
            state_root: b.result.state_root,
            receipts: b.result.receipts.clone(),
            gas_used: b.result.gas_used,
            operations: b.operations.clone(),
        }
    }
}

impl From<SerializableBlock> for FinalizedBlock {
    fn from(sb: SerializableBlock) -> Self {
        Self {
            header: sb.header,
            result: BlockResult {
                state_root: sb.state_root,
                receipts: sb.receipts,
                gas_used: sb.gas_used,
            },
            attestations: vec![],
            operations: sb.operations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solen_crypto::Keypair;
    use solen_execution::genesis::{apply_genesis, GenesisAccount};
    use solen_storage::MemoryStore;
    use solen_types::account::AuthMethod;
    use solen_types::transaction::{Action, UserOperation};

    fn setup_engine() -> (ConsensusEngine, Keypair, [u8; 32], [u8; 32]) {
        let mut store = MemoryStore::new();
        let kp = Keypair::generate();

        let alice = {
            let mut id = [0u8; 32];
            id[..4].copy_from_slice(b"alic");
            id
        };
        let bob = {
            let mut id = [0u8; 32];
            id[..3].copy_from_slice(b"bob");
            id
        };

        apply_genesis(
            &mut store,
            vec![
                GenesisAccount {
                    id: alice,
                    balance: 100_000,
                    auth_methods: vec![AuthMethod::Ed25519 {
                        public_key: kp.public_key(),
                    }],
                },
                GenesisAccount {
                    id: bob,
                    balance: 1_000,
                    auth_methods: vec![],
                },
            ],
        )
        .unwrap();

        let mempool = Mempool::new(1000);
        let engine = ConsensusEngine::new(EngineConfig::default(), Box::new(store), mempool);

        (engine, kp, alice, bob)
    }

    fn setup_multi_validator_engine() -> ConsensusEngine {
        let store = MemoryStore::new();
        let mempool = Mempool::new(1000);

        let v1 = [1u8; 32];
        let v2 = [2u8; 32];
        let v3 = [3u8; 32];

        let vs = ValidatorSet::new(vec![
            ValidatorInfo::new(v1, 100),
            ValidatorInfo::new(v2, 100),
            ValidatorInfo::new(v3, 100),
        ]);

        let config = EngineConfig {
            validator_id: v1,
            ..Default::default()
        };

        ConsensusEngine::with_validators(config, Box::new(store), mempool, vs)
    }

    #[test]
    fn produce_empty_block() {
        let (engine, _, _, _) = setup_engine();
        let produced = engine.produce_block();
        let block = produced.finalized.unwrap();

        assert_eq!(block.header.height, 1);
        assert_eq!(block.result.receipts.len(), 0);
    }

    #[test]
    fn produce_block_with_transactions() {
        let (engine, kp, alice, bob) = setup_engine();
        let executor = BlockExecutor::new();

        let mut op = UserOperation {
            sender: alice,
            nonce: 0,
            actions: vec![Action::Transfer {
                to: bob,
                amount: 500,
            }],
            max_fee: 1000,
            signature: vec![],
        };
        let msg = executor.operation_signing_message(&op);
        op.signature = kp.sign(&msg).to_vec();

        engine.mempool().submit(op);

        let produced = engine.produce_block();
        let block = produced.finalized.unwrap();
        assert_eq!(block.result.receipts.len(), 1);
        assert!(block.result.receipts[0].success);
    }

    #[test]
    fn chain_grows_with_parent_hashes() {
        let (engine, _, _, _) = setup_engine();

        engine.produce_block();
        engine.produce_block();
        engine.produce_block();

        assert_eq!(engine.height(), 3);

        let b2 = engine.get_block(2).unwrap();
        let b3 = engine.get_block(3).unwrap();
        assert_eq!(b3.header.parent_hash, block_hash(&b2.header));
    }

    #[test]
    fn multi_validator_propose_and_attest() {
        let v1 = [1u8; 32];
        let v2 = [2u8; 32];
        let v3 = [3u8; 32];

        let store = MemoryStore::new();
        let mempool = Mempool::new(1000);
        let vs = ValidatorSet::new(vec![
            ValidatorInfo::new(v1, 100),
            ValidatorInfo::new(v2, 100),
            ValidatorInfo::new(v3, 100),
        ]);

        let config = EngineConfig {
            validator_id: v1,
            ..Default::default()
        };

        let engine = ConsensusEngine::with_validators(config, Box::new(store), mempool, vs);

        // v1 proposes — block should NOT be immediately finalized (multi-validator).
        let produced = engine.produce_block();
        assert!(produced.finalized.is_none());
        assert_eq!(engine.height(), 0); // not finalized yet

        // v1 already self-attested during produce_block.
        // v2 attests — still no quorum (2/3 = 66%, need >66%).
        let bh = block_hash(&produced.header);
        engine.accept_attestation(v2, 1, bh);
        assert_eq!(engine.height(), 0); // still not finalized

        // v3 attests — quorum reached (3/3 = 100%).
        engine.accept_attestation(v3, 1, bh);
        assert_eq!(engine.height(), 1); // finalized!

        let block = engine.get_block(1).unwrap();
        assert_eq!(block.attestations.len(), 3);
    }

    #[test]
    fn multi_validator_accept_block_from_peer() {
        let v1 = [1u8; 32];
        let v2 = [2u8; 32];
        let v3 = [3u8; 32];
        let v4 = [4u8; 32];

        // Engine running as v2.
        let store = MemoryStore::new();
        let mempool = Mempool::new(1000);
        let vs = ValidatorSet::new(vec![
            ValidatorInfo::new(v1, 100),
            ValidatorInfo::new(v2, 100),
            ValidatorInfo::new(v3, 100),
            ValidatorInfo::new(v4, 100),
        ]);

        let config = EngineConfig {
            validator_id: v2,
            ..Default::default()
        };

        let engine = ConsensusEngine::with_validators(config, Box::new(store), mempool, vs);

        // Simulate receiving a block proposed by v1.
        let header = BlockHeader {
            height: 1,
            epoch: 0,
            parent_hash: [0u8; 32],
            state_root: [0u8; 32], // empty state
            transactions_root: [0u8; 32],
            receipts_root: [0u8; 32],
            proposer: v1,
            timestamp_ms: 12345,
            proposer_signature: vec![],
        };

        // Accept the block (no operations, so state root should match).
        let accepted = engine.accept_block(&header, &[]);
        // State root might not match since our store computes differently.
        // The key test is that the flow doesn't panic.
        // In production, both nodes start from the same genesis.
    }

    /// Read alice/any account balance from the engine's live store.
    fn engine_balance(engine: &ConsensusEngine, id: &[u8; 32]) -> u128 {
        let store = engine.store();
        let store = store.read().unwrap();
        solen_execution::state::ReadonlyStateManager::new(store.as_ref())
            .get_balance(id)
            .unwrap()
    }

    #[test]
    fn rollback_to_height_rewinds_state_to_common_ancestor() {
        let (engine, kp, alice, bob) = setup_engine();
        let executor = BlockExecutor::new();

        let mut produce_transfer = |nonce: u64, amount: u128| {
            let mut op = UserOperation {
                sender: alice,
                nonce,
                actions: vec![Action::Transfer { to: bob, amount }],
                max_fee: 1000,
                signature: vec![],
            };
            let msg = executor.operation_signing_message(&op);
            op.signature = kp.sign(&msg).to_vec();
            engine.mempool().submit(op);
            engine.produce_block();
        };

        // Two committed blocks → the common ancestor we'll roll back to.
        produce_transfer(0, 500);
        produce_transfer(1, 300);
        let target_h = engine.height();
        let target_root = engine.get_block(target_h).unwrap().header.state_root;
        // Captured dynamically (the executor also charges fees, so we compare
        // against the captured values rather than hardcoded transfer math).
        let alice_at_target = engine_balance(&engine, &alice);
        let bob_at_target = engine_balance(&engine, &bob);
        assert_eq!(
            bob_at_target,
            1_000 + 800,
            "bob receives transfers (no fees on the recipient)"
        );

        // Three more blocks form the "forked suffix" to undo.
        produce_transfer(2, 100);
        produce_transfer(3, 100);
        produce_transfer(4, 100);
        assert_eq!(engine.height(), target_h + 3);
        assert_ne!(
            engine.get_block(engine.height()).unwrap().header.state_root,
            target_root
        );
        assert_ne!(engine_balance(&engine, &alice), alice_at_target);

        // Roll back in place to the common ancestor.
        assert!(engine.rollback_to_height(target_h, &target_root));
        assert_eq!(engine.height(), target_h);
        assert_eq!(
            engine.get_block(target_h).unwrap().header.state_root,
            target_root
        );
        {
            let store = engine.store();
            let store = store.read().unwrap();
            assert_eq!(store.state_root(), target_root, "live store root restored");
        }
        assert_eq!(
            engine_balance(&engine, &alice),
            alice_at_target,
            "alice balance restored"
        );
        assert_eq!(
            engine_balance(&engine, &bob),
            bob_at_target,
            "bob balance restored"
        );
        // The forked suffix blocks are gone from the chain.
        assert!(engine.get_block(target_h + 1).is_none());

        // The chain can advance again cleanly after rollback.
        produce_transfer(2, 100);
        assert_eq!(engine.height(), target_h + 1);
    }

    /// Reproduces the 2-down competing-block liveness deadlock (mainnet halt
    /// 2026-06-26 at 683764). When the proposer rotation emits competing blocks
    /// at the same height and the first attestations split across them, the
    /// fleet CANNOT converge: `accept_attestation` dedups by validator_id (each
    /// validator votes at most once per height, with no vote-change) and ignores
    /// attestations whose hash differs from the node's single pending block — so
    /// minority-block voters can never move to the majority block and neither
    /// block reaches the 2/3 threshold. The chain wedges.
    ///
    /// This pins the CURRENT (buggy) behaviour: a 2/2 split across two competing
    /// blocks finalizes NOTHING. The attestation-aware fork-choice fix must make
    /// this converge and finalize a single block.
    #[test]
    fn competing_blocks_with_split_attestations_deadlock() {
        let kps: Vec<Keypair> = (0..5).map(|_| Keypair::generate()).collect();
        let ids: Vec<[u8; 32]> = kps.iter().map(|k| k.public_key()).collect();
        let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
        let store = MemoryStore::new();
        let mempool = Mempool::new(1000);
        let config = EngineConfig {
            validator_id: ids[0],
            ..Default::default()
        };
        let engine = ConsensusEngine::with_validators(config, Box::new(store), mempool, vs);

        let head_h = engine.height();
        let (parent_hash, parent_ts) = engine
            .get_block(head_h)
            .map(|b| (block_hash(&b.header), b.header.timestamp_ms))
            .unwrap_or(([0u8; 32], 0));
        let next_h = head_h + 1;

        // A signed empty block at `next_h` from a given proposer.
        let make_block = |proposer_idx: usize| -> BlockHeader {
            let mut h = BlockHeader {
                height: next_h,
                epoch: next_h / crate::epoch::EPOCH_LENGTH,
                parent_hash,
                state_root: [0u8; 32],
                transactions_root: [0u8; 32],
                receipts_root: [0u8; 32],
                proposer: ids[proposer_idx],
                timestamp_ms: parent_ts + 6000,
                proposer_signature: vec![],
            };
            let bh = block_hash(&h);
            h.proposer_signature = kps[proposer_idx].sign(&bh).to_vec();
            h
        };

        // Two competing blocks from different proposers (the backup rotation
        // under 2-down): same parent + timestamp, different proposer => different
        // hash.
        let block_a = make_block(1);
        let block_b = make_block(2);
        let hash_a = block_hash(&block_a);
        let hash_b = block_hash(&block_b);
        assert_ne!(hash_a, hash_b);

        assert!(engine.accept_block(&block_a, &[]), "block_a accepted");
        engine.accept_block(&block_b, &[]); // fork-choice keeps one as pending

        // First attestations split 2/2: {v0,v1} -> A, {v2,v3} -> B. Quorum for 5
        // is 4 (2/3 = 3.33), so neither side can reach it without a voter moving.
        engine.accept_attestation(ids[0], next_h, hash_a);
        engine.accept_attestation(ids[1], next_h, hash_a);
        engine.accept_attestation(ids[2], next_h, hash_b);
        engine.accept_attestation(ids[3], next_h, hash_b);

        // CURRENT (buggy): the split is unrecoverable — nothing finalizes.
        assert_eq!(
            engine.height(),
            head_h,
            "split attestations cannot converge -> chain wedged (the bug under repair)"
        );
    }

    /// With fork-choice v2 active, the same competing-block split that wedges
    /// under v1 CONVERGES: vote-changes are honoured, votes aggregate on one
    /// hash, and the block finalizes at 2/3. This is the fix for the 2-down
    /// liveness deadlock.
    #[test]
    fn competing_blocks_converge_under_fork_choice_v2() {
        let kps: Vec<Keypair> = (0..5).map(|_| Keypair::generate()).collect();
        let ids: Vec<[u8; 32]> = kps.iter().map(|k| k.public_key()).collect();
        let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
        let store = MemoryStore::new();
        let mempool = Mempool::new(1000);
        // v2 active from height 0.
        let config = EngineConfig {
            validator_id: ids[0],
            fork_choice_v2_height: 0,
            ..Default::default()
        };
        let engine = ConsensusEngine::with_validators(config, Box::new(store), mempool, vs);

        let head_h = engine.height();
        let (parent_hash, parent_ts) = engine
            .get_block(head_h)
            .map(|b| (block_hash(&b.header), b.header.timestamp_ms))
            .unwrap_or(([0u8; 32], 0));
        // Empty block leaves state unchanged — use the live empty-store root so
        // finalization's execution + state-root check passes.
        let empty_root = { engine.store().read().unwrap().state_root() };
        let next_h = head_h + 1;

        let make_block = |proposer_idx: usize| -> BlockHeader {
            let mut h = BlockHeader {
                height: next_h,
                epoch: next_h / crate::epoch::EPOCH_LENGTH,
                parent_hash,
                state_root: empty_root,
                transactions_root: [0u8; 32],
                receipts_root: [0u8; 32],
                proposer: ids[proposer_idx],
                timestamp_ms: parent_ts + 6000,
                proposer_signature: vec![],
            };
            let bh = block_hash(&h);
            h.proposer_signature = kps[proposer_idx].sign(&bh).to_vec();
            h
        };

        let block_a = make_block(1);
        let block_b = make_block(2);
        let hash_a = block_hash(&block_a);
        let hash_b = block_hash(&block_b);
        assert_ne!(hash_a, hash_b);

        // Both candidates are tracked (v2 keeps competing blocks).
        assert!(engine.accept_block(&block_a, &[]));
        assert!(engine.accept_block(&block_b, &[]));

        // Initial split: {v1,v2} -> A, {v3,v4} -> B. Quorum for 5 is 4; neither
        // side has it, so nothing finalizes yet.
        engine.accept_attestation(ids[1], next_h, hash_a);
        engine.accept_attestation(ids[2], next_h, hash_a);
        engine.accept_attestation(ids[3], next_h, hash_b);
        engine.accept_attestation(ids[4], next_h, hash_b);
        assert_eq!(engine.height(), head_h, "2/2 split must not finalize");

        // v3 and v4 CHANGE their vote to A (as fork choice would converge them).
        // Under v1 these would be rejected as duplicates; under v2 they replace
        // the earlier vote, giving A four votes -> quorum -> finalize.
        engine.accept_attestation(ids[3], next_h, hash_a);
        engine.accept_attestation(ids[4], next_h, hash_a);

        assert_eq!(
            engine.height(),
            next_h,
            "vote-change converged -> block A finalized"
        );
        let finalized = engine.get_block(next_h).unwrap();
        assert_eq!(
            block_hash(&finalized.header),
            hash_a,
            "the converged hash finalized"
        );
    }

    /// Regression for the 2026-07-16 mainnet halt. A proposer emits a block whose
    /// state root the honest majority does NOT reproduce on execution (a
    /// non-deterministic divergence). The majority blind-attests the header
    /// (execution happens only at finalize), reaches quorum, then every node
    /// rejects the block on execution. It must NOT latch into a resync halt —
    /// instead the block is marked execution-invalid and dropped, and the backup
    /// proposer's valid block at the same height finalizes so the chain advances.
    #[test]
    fn bad_root_block_rejected_without_resync_then_backup_finalizes() {
        let kps: Vec<Keypair> = (0..5).map(|_| Keypair::generate()).collect();
        let ids: Vec<[u8; 32]> = kps.iter().map(|k| k.public_key()).collect();
        let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
        let store = MemoryStore::new();
        let mempool = Mempool::new(1000);
        // We are ids[0], an honest validator; v2 active from height 0.
        let config = EngineConfig {
            validator_id: ids[0],
            fork_choice_v2_height: 0,
            ..Default::default()
        };
        let engine = ConsensusEngine::with_validators(config, Box::new(store), mempool, vs);

        let head_h = engine.height();
        let (parent_hash, parent_ts) = engine
            .get_block(head_h)
            .map(|b| (block_hash(&b.header), b.header.timestamp_ms))
            .unwrap_or(([0u8; 32], 0));
        let empty_root = { engine.store().read().unwrap().state_root() };
        let next_h = head_h + 1;

        let make_block = |proposer_idx: usize, state_root: [u8; 32]| -> BlockHeader {
            let mut h = BlockHeader {
                height: next_h,
                epoch: next_h / crate::epoch::EPOCH_LENGTH,
                parent_hash,
                state_root,
                transactions_root: [0u8; 32],
                receipts_root: [0u8; 32],
                proposer: ids[proposer_idx],
                timestamp_ms: parent_ts + 6000,
                proposer_signature: vec![],
            };
            let bh = block_hash(&h);
            h.proposer_signature = kps[proposer_idx].sign(&bh).to_vec();
            h
        };

        // Proposer ids[1] emits a block claiming a root nobody else reproduces:
        // it's an empty block (real root == empty_root) but the header lies [7;32].
        let bad_block = make_block(1, [7u8; 32]);
        let bad_hash = block_hash(&bad_block);
        assert!(
            engine.accept_block(&bad_block, &[]),
            "bad block accepted as candidate (header valid)"
        );

        // The honest majority BLIND-attests the header — incl. us — to quorum (4/5).
        engine.accept_attestation(ids[0], next_h, bad_hash);
        engine.accept_attestation(ids[1], next_h, bad_hash);
        engine.accept_attestation(ids[2], next_h, bad_hash);
        engine.accept_attestation(ids[3], next_h, bad_hash);

        // On finalize we execute, find the root mismatch, and REJECT — no resync.
        assert_eq!(engine.height(), head_h, "bad block must NOT finalize");
        assert!(
            !engine.resync_requested(),
            "must NOT latch into resync — that is what halted mainnet 2026-07-16"
        );
        assert!(
            !engine.accept_block(&bad_block, &[]),
            "known execution-invalid block must not be re-accepted (else it re-wins as rank-0 proposer)"
        );

        // The backup proposer ids[2] produces a VALID block at the same height.
        let good_block = make_block(2, empty_root);
        let good_hash = block_hash(&good_block);
        assert_ne!(bad_hash, good_hash);
        assert!(
            engine.accept_block(&good_block, &[]),
            "valid backup block accepted"
        );
        engine.accept_attestation(ids[0], next_h, good_hash);
        engine.accept_attestation(ids[1], next_h, good_hash);
        engine.accept_attestation(ids[2], next_h, good_hash);
        engine.accept_attestation(ids[3], next_h, good_hash);

        assert_eq!(
            engine.height(),
            next_h,
            "backup's valid block finalized — chain advanced, no halt"
        );
        assert_eq!(
            block_hash(&engine.get_block(next_h).unwrap().header),
            good_hash
        );
    }

    /// Root-cause regression for the 2026-07-16 divergence: the produce path
    /// commits eagerly, so a superseded produce attempt can leave writes in the
    /// store with nothing to undo them; the next produce then builds on corrupted
    /// state (at an epoch boundary, a double-applied reward) → a root the fleet
    /// rejects. produce_block must first restore the store to the finalized tip.
    #[test]
    fn produce_restores_store_after_leaked_eager_commit() {
        let kps: Vec<Keypair> = (0..3).map(|_| Keypair::generate()).collect();
        let ids: Vec<[u8; 32]> = kps.iter().map(|k| k.public_key()).collect();
        let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
        let store = MemoryStore::new();
        let mempool = Mempool::new(1000);
        let config = EngineConfig {
            validator_id: ids[0],
            fork_choice_v2_height: 0,
            ..Default::default()
        };
        let engine = ConsensusEngine::with_validators(config, Box::new(store), mempool, vs);

        // Establish a real finalized tip: produce an empty block 1 and finalize it
        // with a 2/3 quorum (2 of 3).
        let p1 = engine.produce_block();
        let hash1 = block_hash(&p1.header);
        engine.accept_attestation(ids[1], 1, hash1);
        engine.accept_attestation(ids[2], 1, hash1);
        assert_eq!(
            engine.height(),
            1,
            "block 1 finalized — we have a finalized tip"
        );
        let tip_root = engine.store().read().unwrap().state_root();

        // Simulate a superseded produce attempt at height 2 that eager-committed a
        // write and was never finalized or reverted (the leak).
        let leaked_key = b"acc/leaked-phantom".to_vec();
        {
            let store_arc = engine.store();
            let mut store = store_arc.write().unwrap();
            store.put(&leaked_key, b"phantom").unwrap();
            store.commit_root();
        }
        let drifted_root = engine.store().read().unwrap().state_root();
        assert_ne!(
            drifted_root, tip_root,
            "store drifted from finalized tip (leaked eager-commit)"
        );

        // Stash the matching pending block with its reverse-delta, as produce would.
        let mut header = BlockHeader {
            height: 2,
            epoch: 0,
            parent_hash: hash1,
            state_root: drifted_root,
            transactions_root: [0u8; 32],
            receipts_root: [0u8; 32],
            proposer: ids[0],
            timestamp_ms: 12000,
            proposer_signature: vec![],
        };
        let bh = block_hash(&header);
        header.proposer_signature = kps[0].sign(&bh).to_vec();
        let mut revert: std::collections::HashMap<Vec<u8>, Option<Vec<u8>>> =
            std::collections::HashMap::new();
        revert.insert(leaked_key.clone(), None); // undo = delete (no prior value)
        engine
            .v2_blocks
            .write()
            .unwrap()
            .entry(2)
            .or_default()
            .insert(
                bh,
                PendingBlock {
                    header,
                    operations: vec![],
                    proposed_at: std::time::Instant::now(),
                    already_executed: true,
                    result: None,
                    revert: Some(revert),
                    mismatch_count: 0,
                },
            );

        // Producing must first restore the store to the finalized tip.
        let _ = engine.produce_block();
        assert_eq!(
            engine.store().read().unwrap().state_root(),
            tip_root,
            "produce restored the store to the finalized tip before building the next block"
        );
        assert!(
            engine
                .store()
                .read()
                .unwrap()
                .get(&leaked_key)
                .unwrap()
                .is_none(),
            "the leaked eager-commit write was undone"
        );
    }

    /// The produce path must persist a DURABLE eager-revert record when it
    /// eager-commits, and clear it when the block finalizes.
    #[test]
    fn produce_persists_durable_eager_revert_and_clears_on_finalize() {
        let kps: Vec<Keypair> = (0..3).map(|_| Keypair::generate()).collect();
        let ids: Vec<[u8; 32]> = kps.iter().map(|k| k.public_key()).collect();
        let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
        let config = EngineConfig {
            validator_id: ids[0],
            fork_choice_v2_height: 0,
            ..Default::default()
        };
        let engine = ConsensusEngine::with_validators(
            config,
            Box::new(MemoryStore::new()),
            Mempool::new(1000),
            vs,
        );

        let p1 = engine.produce_block();
        let hash1 = block_hash(&p1.header);
        // After eager-committing block 1 (before finalize) the durable record exists.
        assert!(
            engine
                .store()
                .read()
                .unwrap()
                .get(&solen_storage::eager_revert_key(1))
                .unwrap()
                .is_some(),
            "produce persisted a durable eager-revert for the eager-committed height"
        );
        // Finalize block 1 (2/3 of 3) — the record must be cleared (now canonical).
        engine.accept_attestation(ids[1], 1, hash1);
        engine.accept_attestation(ids[2], 1, hash1);
        assert_eq!(engine.height(), 1);
        assert!(
            engine
                .store()
                .read()
                .unwrap()
                .get(&solen_storage::eager_revert_key(1))
                .unwrap()
                .is_none(),
            "finalize cleared the durable eager-revert — a canonical block is never undone"
        );
    }

    /// Regression for the mainnet validator5 wedge (2026-08-06): a superseded
    /// eager-commit that MUTATED state and survives a RESTART. After a restart the
    /// in-memory reverts are gone, so the in-process repair in
    /// `restore_store_to_finalized_tip` can't fire — the store boots durably
    /// drifted (height N carrying N+1's trie state) and can never reapply the
    /// canonical chain. The DURABLE eager-revert, replayed by the boot reconcile,
    /// must repair the store to the true finalized tip.
    #[test]
    fn boot_reconcile_repairs_durable_leaked_eager_commit_after_restart() {
        let kps: Vec<Keypair> = (0..3).map(|_| Keypair::generate()).collect();
        let ids: Vec<[u8; 32]> = kps.iter().map(|k| k.public_key()).collect();
        let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
        let config = EngineConfig {
            validator_id: ids[0],
            fork_choice_v2_height: 0,
            ..Default::default()
        };
        let engine = ConsensusEngine::with_validators(
            config.clone(),
            Box::new(MemoryStore::new()),
            Mempool::new(1000),
            vs.clone(),
        );

        // Finalized tip at height 1.
        let p1 = engine.produce_block();
        let hash1 = block_hash(&p1.header);
        engine.accept_attestation(ids[1], 1, hash1);
        engine.accept_attestation(ids[2], 1, hash1);
        assert_eq!(engine.height(), 1, "have a finalized tip at height 1");
        let tip_root = engine.store().read().unwrap().state_root();

        // Simulate a durable eager-commit at height 2 that MUTATED state and was
        // never finalized (models the epoch-boundary reward leak): a phantom state
        // write plus the durable undo-record, committed exactly as
        // `execute_block_journaled_durable` would, with chain meta left at the
        // finalized tip (height 1).
        let phantom = b"acc/phantom-reward".to_vec();
        let rk = solen_storage::eager_revert_key(2);
        {
            let store_arc = engine.store();
            let mut store = store_arc.write().unwrap();
            store.put(&phantom, b"leaked").unwrap();
            // Undo = delete the phantom (no prior value) AND delete the record itself.
            let pairs: Vec<(Vec<u8>, Option<Vec<u8>>)> =
                vec![(phantom.clone(), None), (rk.clone(), None)];
            store.put(&rk, &borsh::to_vec(&pairs).unwrap()).unwrap();
            store.commit_root();
        }
        let drifted = engine.store().read().unwrap().state_root();
        assert_ne!(drifted, tip_root, "store drifted (leaked eager-commit)");

        // SIMULATE RESTART: clone the durable store into a fresh store and build a
        // NEW engine — no in-memory journal/pending/chain survives.
        let mut restarted_store = MemoryStore::new();
        for (k, v) in engine.store().read().unwrap().scan_all().unwrap() {
            restarted_store.put(&k, &v).unwrap();
        }
        drop(engine);
        let engine2 = ConsensusEngine::with_validators(
            config,
            Box::new(restarted_store),
            Mempool::new(1000),
            vs,
        );

        // Boot reconcile must have replayed the durable revert back to the tip.
        assert_eq!(engine2.height(), 1, "restart restored the finalized height");
        assert_eq!(
            engine2.store().read().unwrap().state_root(),
            tip_root,
            "boot reconcile repaired the store to the true finalized-tip root"
        );
        assert!(
            engine2
                .store()
                .read()
                .unwrap()
                .get(&phantom)
                .unwrap()
                .is_none(),
            "the leaked eager write was undone"
        );
        assert!(
            engine2.store().read().unwrap().get(&rk).unwrap().is_none(),
            "the durable eager-revert record was consumed by the repair"
        );
        // The boot placeholder now carries the corrected root, not the leaked one,
        // so applying the real canonical block 2 on top will agree.
        assert_eq!(
            engine2.latest_block().unwrap().result.state_root,
            tip_root,
            "boot placeholder carries the corrected root, not the leaked one"
        );
    }

    /// produce_block under v2: our block becomes a candidate + self-vote, and
    /// finalizes once peers' votes reach 2/3.
    #[test]
    fn produce_block_finalizes_under_fork_choice_v2() {
        let kps: Vec<Keypair> = (0..3).map(|_| Keypair::generate()).collect();
        let ids: Vec<[u8; 32]> = kps.iter().map(|k| k.public_key()).collect();
        let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
        let store = MemoryStore::new();
        let mempool = Mempool::new(1000);
        let config = EngineConfig {
            validator_id: ids[0],
            fork_choice_v2_height: 0,
            ..Default::default()
        };
        let engine = ConsensusEngine::with_validators(config, Box::new(store), mempool, vs);

        let produced = engine.produce_block();
        assert!(
            produced.finalized.is_none(),
            "multi-validator: not finalized on production"
        );
        assert_eq!(engine.height(), 0);
        let h = produced.header.height;
        let bh = block_hash(&produced.header);

        // Our self-vote is queued for the node layer to broadcast.
        assert!(
            engine.take_v2_revotes().contains(&(h, bh)),
            "self-vote enqueued"
        );

        // Peers attest -> 3/3 -> finalize (quorum needs all 3 in a 3-set).
        engine.accept_attestation(ids[1], h, bh);
        engine.accept_attestation(ids[2], h, bh);
        assert_eq!(
            engine.height(),
            h,
            "produced block finalized via peer votes (v2)"
        );
    }

    /// Safety: under v2 a validator's vote-change REPLACES its prior vote (never
    /// adds), so it cannot inflate quorum. Uses a non-validator observer engine
    /// so it doesn't auto-vote and we control all 5 votes.
    #[test]
    fn v2_vote_change_replaces_not_adds() {
        let kps: Vec<Keypair> = (0..5).map(|_| Keypair::generate()).collect();
        let ids: Vec<[u8; 32]> = kps.iter().map(|k| k.public_key()).collect();
        let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
        let store = MemoryStore::new();
        let mempool = Mempool::new(1000);
        // Observer: validator_id is NOT in the set, so the engine never self-votes.
        let config = EngineConfig {
            validator_id: [99u8; 32],
            fork_choice_v2_height: 0,
            ..Default::default()
        };
        let engine = ConsensusEngine::with_validators(config, Box::new(store), mempool, vs);

        let empty_root = { engine.store().read().unwrap().state_root() };
        let next_h = engine.height() + 1;
        let make_block = |idx: usize| -> BlockHeader {
            let mut h = BlockHeader {
                height: next_h,
                epoch: next_h / crate::epoch::EPOCH_LENGTH,
                parent_hash: [0u8; 32],
                state_root: empty_root,
                transactions_root: [0u8; 32],
                receipts_root: [0u8; 32],
                proposer: ids[idx],
                timestamp_ms: 6000,
                proposer_signature: vec![],
            };
            let bh = block_hash(&h);
            h.proposer_signature = kps[idx].sign(&bh).to_vec();
            h
        };
        let block_a = make_block(0);
        let block_b = make_block(1);
        let hash_a = block_hash(&block_a);
        let hash_b = block_hash(&block_b);
        assert!(engine.accept_block(&block_a, &[]));
        assert!(engine.accept_block(&block_b, &[]));

        // 3 votes for A — quorum is 4, so no finalize.
        engine.accept_attestation(ids[1], next_h, hash_a);
        engine.accept_attestation(ids[2], next_h, hash_a);
        engine.accept_attestation(ids[3], next_h, hash_a);
        assert_eq!(engine.height(), next_h - 1, "3/5 < quorum");
        // Duplicate vote from v1 must not inflate to 4.
        engine.accept_attestation(ids[1], next_h, hash_a);
        assert_eq!(
            engine.height(),
            next_h - 1,
            "duplicate vote does not inflate quorum"
        );
        // v1 changes to B: A drops to {v2,v3}=2, still no quorum.
        engine.accept_attestation(ids[1], next_h, hash_b);
        assert_eq!(engine.height(), next_h - 1, "vote-change moves, not adds");
        // v1 back to A + v4 to A -> A has {v1,v2,v3,v4}=4 -> finalize.
        engine.accept_attestation(ids[1], next_h, hash_a);
        engine.accept_attestation(ids[4], next_h, hash_a);
        assert_eq!(engine.height(), next_h, "four distinct votes finalize");
        assert_eq!(
            block_hash(&engine.get_block(next_h).unwrap().header),
            hash_a
        );
    }

    /// Security (H-09): under v2 a (validator-signed) attestation for an
    /// out-of-window height must be rejected and NEVER stored. Without the
    /// look-ahead bound, one Byzantine validator can sign votes for arbitrary
    /// far-future heights, each creating a v2_votes entry that GC never reclaims
    /// (finalize/clear_stale only prune h <= current) -> unbounded memory DoS.
    #[test]
    fn v2_rejects_out_of_window_attestation_heights() {
        let kps: Vec<Keypair> = (0..5).map(|_| Keypair::generate()).collect();
        let ids: Vec<[u8; 32]> = kps.iter().map(|k| k.public_key()).collect();
        let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
        let store = MemoryStore::new();
        let mempool = Mempool::new(1000);
        // Observer engine (validator_id not in set) so it never self-votes and we
        // control exactly which heights enter v2_votes.
        let config = EngineConfig {
            validator_id: [99u8; 32],
            fork_choice_v2_height: 0,
            ..Default::default()
        };
        let engine = ConsensusEngine::with_validators(config, Box::new(store), mempool, vs);

        let current = engine.height();
        let some_hash = [7u8; 32];

        // Far-future height (the DoS vector): rejected, nothing recorded.
        assert!(!engine.accept_attestation(ids[1], u64::MAX, some_hash));
        // Just past the look-ahead window (64): rejected.
        assert!(!engine.accept_attestation(ids[1], current + 65, some_hash));
        // Already-finalized / current height: rejected (no point recording).
        assert!(!engine.accept_attestation(ids[1], current, some_hash));
        assert!(
            engine.v2_votes.read().unwrap().is_empty(),
            "no out-of-window height may be stored in v2_votes"
        );

        // In-window heights are still recorded (next height + within the window).
        engine.accept_attestation(ids[1], current + 1, some_hash);
        engine.accept_attestation(ids[1], current + 64, some_hash);
        let votes = engine.v2_votes.read().unwrap();
        assert!(votes.contains_key(&(current + 1)), "next height recorded");
        assert!(votes.contains_key(&(current + 64)), "window edge recorded");
        assert_eq!(votes.len(), 2, "only the two in-window heights stored");
    }

    /// Security (C-03): header_is_validator_signed must accept a header signed by
    /// an active validator and reject every forgery class, so an unauthenticated
    /// far-future NewBlock cannot poison network_height.
    #[test]
    fn header_is_validator_signed_distinguishes_real_from_forged() {
        let kps: Vec<Keypair> = (0..3).map(|_| Keypair::generate()).collect();
        let ids: Vec<[u8; 32]> = kps.iter().map(|k| k.public_key()).collect();
        let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
        let store = MemoryStore::new();
        let mempool = Mempool::new(1000);
        let config = EngineConfig {
            validator_id: ids[0],
            ..Default::default()
        };
        let engine = ConsensusEngine::with_validators(config, Box::new(store), mempool, vs);

        let mk = |proposer: [u8; 32], signer: Option<&Keypair>| -> BlockHeader {
            let mut h = BlockHeader {
                height: 999_999, // far-future — the poisoning case
                epoch: 0,
                parent_hash: [0u8; 32],
                state_root: [0u8; 32],
                transactions_root: [0u8; 32],
                receipts_root: [0u8; 32],
                proposer,
                timestamp_ms: 0,
                proposer_signature: vec![],
            };
            if let Some(kp) = signer {
                let bh = block_hash(&h);
                h.proposer_signature = kp.sign(&bh).to_vec();
            }
            h
        };

        // Real: signed by an active validator over its own block hash.
        assert!(engine.header_is_validator_signed(&mk(ids[1], Some(&kps[1]))));
        // Forged: no signature.
        assert!(!engine.header_is_validator_signed(&mk(ids[1], None)));
        // Forged: proposer not in the validator set (even if self-signed).
        let outsider = Keypair::generate();
        assert!(!engine.header_is_validator_signed(&mk(outsider.public_key(), Some(&outsider))));
        // Forged: signed by a different key than the claimed proposer.
        let mut wrong = mk(ids[1], None);
        let bh = block_hash(&wrong);
        wrong.proposer_signature = kps[2].sign(&bh).to_vec();
        assert!(!engine.header_is_validator_signed(&wrong));
    }

    /// Security (C-02): with authenticate_sync_blocks on, replay_synced_block
    /// rejects a synced block not signed by an active validator (forgery /
    /// fake-chain injection) while still applying a legitimately validator-signed
    /// block. With the flag off, the legacy unauthenticated behavior is preserved
    /// (which is exactly the vulnerability).
    #[test]
    fn replay_synced_block_authenticates_when_enabled() {
        let kps: Vec<Keypair> = (0..3).map(|_| Keypair::generate()).collect();
        let ids: Vec<[u8; 32]> = kps.iter().map(|k| k.public_key()).collect();
        let make_engine = |auth: bool| {
            let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
            let store = MemoryStore::new();
            let mempool = Mempool::new(1000);
            let config = EngineConfig {
                validator_id: ids[0],
                authenticate_sync_blocks: auth,
                ..Default::default()
            };
            ConsensusEngine::with_validators(config, Box::new(store), mempool, vs)
        };
        // Empty block at height 1 whose state_root matches the unchanged store, so
        // the only thing under test is the auth gate (not the root check).
        let build = |engine: &ConsensusEngine,
                     signer: Option<&Keypair>,
                     proposer: [u8; 32]|
         -> BlockHeader {
            let empty_root = engine.store().read().unwrap().state_root();
            let mut h = BlockHeader {
                height: 1,
                epoch: 0,
                parent_hash: [0u8; 32],
                state_root: empty_root,
                transactions_root: [0u8; 32],
                receipts_root: [0u8; 32],
                proposer,
                timestamp_ms: 6000,
                proposer_signature: vec![],
            };
            if let Some(kp) = signer {
                let bh = block_hash(&h);
                h.proposer_signature = kp.sign(&bh).to_vec();
            }
            h
        };

        // Flag ON: unsigned forgery rejected (chain does not advance); a
        // validator-signed block applies.
        let eng = make_engine(true);
        let forged = build(&eng, None, ids[1]); // valid proposer id, NO signature
        assert!(
            !eng.replay_synced_block(&forged, &[], vec![]),
            "unsigned synced block must be rejected"
        );
        assert_eq!(
            eng.height(),
            0,
            "rejected forgery must not advance the chain"
        );
        let signed = build(&eng, Some(&kps[1]), ids[1]);
        assert!(
            eng.replay_synced_block(&signed, &[], vec![]),
            "validator-signed synced block must apply"
        );
        assert_eq!(eng.height(), 1);

        // Flag ON: a block signed by a NON-validator key is rejected.
        let eng2 = make_engine(true);
        let outsider = Keypair::generate();
        let foreign = build(&eng2, Some(&outsider), outsider.public_key());
        assert!(
            !eng2.replay_synced_block(&foreign, &[], vec![]),
            "non-validator-signed synced block must be rejected"
        );
        assert_eq!(eng2.height(), 0);

        // Flag OFF: legacy behavior — the unsigned block is accepted (the bug).
        let eng3 = make_engine(false);
        let forged3 = build(&eng3, None, ids[1]);
        assert!(
            eng3.replay_synced_block(&forged3, &[], vec![]),
            "with flag off, legacy path accepts unauthenticated block"
        );
        assert_eq!(eng3.height(), 1);
    }

    #[test]
    fn process_slashing_does_not_locally_mutate_consensus_set() {
        // Determinism guarantee: queuing downtime evidence must NOT change the
        // offender's stake or active status in the in-memory consensus set —
        // only the deterministic on-chain slash (applied identically by every
        // node) may. Otherwise nodes diverge on proposer selection / quorum and
        // emit competing blocks (the 2026-06-24 fork cascade).
        let engine = setup_multi_validator_engine();
        let offender = [2u8; 32];
        let evidence = crate::slashing::SlashingEvidence {
            offender,
            reason: crate::slashing::SlashingReason::Downtime { missed_blocks: 100 },
        };

        let (stake_before, active_before) = {
            let vs = engine.validator_set();
            let vs = vs.read().unwrap();
            let v = vs.all().iter().find(|v| v.id == offender).unwrap();
            (v.stake, v.is_active())
        };
        assert!(active_before);

        engine.process_slashing(&evidence);

        let vs = engine.validator_set();
        let vs = vs.read().unwrap();
        let v = vs.all().iter().find(|v| v.id == offender).unwrap();
        assert_eq!(v.stake, stake_before, "stake must not change locally");
        assert!(
            v.is_active(),
            "status must remain Active locally (jail only via finalized on-chain slash)"
        );
    }

    #[test]
    fn rollback_rejects_target_outside_journal_range() {
        let (engine, kp, alice, bob) = setup_engine();
        let executor = BlockExecutor::new();
        let mut op = UserOperation {
            sender: alice,
            nonce: 0,
            actions: vec![Action::Transfer {
                to: bob,
                amount: 100,
            }],
            max_fee: 1000,
            signature: vec![],
        };
        op.signature = kp.sign(&executor.operation_signing_message(&op)).to_vec();
        engine.mempool().submit(op);
        engine.produce_block();
        let h = engine.height();
        let root = engine.get_block(h).unwrap().header.state_root;

        // target >= tip is rejected without mutating.
        assert!(!engine.rollback_to_height(h, &root));
        assert!(!engine.rollback_to_height(h + 5, &root));
        assert_eq!(
            engine.height(),
            h,
            "height unchanged after rejected rollback"
        );
    }

    /// Accept-path atomic restore (validator5 wedge fix, 2026-08-06): a CANONICAL
    /// block received from a peer must still finalize when our store is polluted
    /// by a leaked eager-commit at that height. finalize_pending_block must
    /// restore the store to the finalized tip BEFORE executing the peer's block —
    /// otherwise the canonical block executes on drifted state, mismatches, and is
    /// wrongly rejected, wedging the node.
    #[test]
    fn accept_path_restores_before_executing_canonical_peer_block() {
        let kps: Vec<Keypair> = (0..5).map(|_| Keypair::generate()).collect();
        let ids: Vec<[u8; 32]> = kps.iter().map(|k| k.public_key()).collect();
        let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
        let config = EngineConfig {
            validator_id: ids[0],
            fork_choice_v2_height: 0, // v2 active
            ..Default::default()
        };
        let engine = ConsensusEngine::with_validators(
            config,
            Box::new(MemoryStore::new()),
            Mempool::new(1000),
            vs,
        );

        // Establish a real finalized tip: produce block 1 and finalize it (4/5),
        // so `chain` has a tip for restore_store_to_finalized_tip to target.
        let p1 = engine.produce_block();
        let hash1 = block_hash(&p1.header);
        engine.accept_attestation(ids[1], 1, hash1);
        engine.accept_attestation(ids[2], 1, hash1);
        engine.accept_attestation(ids[3], 1, hash1);
        engine.accept_attestation(ids[4], 1, hash1);
        assert_eq!(engine.height(), 1, "finalized tip at height 1");

        let head_h = engine.height();
        let next_h = head_h + 1;
        let (parent_hash, parent_ts) = engine
            .get_block(head_h)
            .map(|b| (block_hash(&b.header), b.header.timestamp_ms))
            .unwrap_or(([0u8; 32], 0));
        let tip_root = engine.store().read().unwrap().state_root();

        // Simulate OUR leaked eager-commit at height `next_h`: pollute the store
        // and stash the matching candidate (already-executed, with its revert) in
        // v2_blocks so restore_store_to_finalized_tip can undo it.
        let phantom = b"acc/phantom-leak".to_vec();
        {
            let store_arc = engine.store();
            let mut store = store_arc.write().unwrap();
            store.put(&phantom, b"leaked").unwrap();
            store.commit_root();
        }
        let drifted_root = engine.store().read().unwrap().state_root();
        assert_ne!(
            drifted_root, tip_root,
            "store drifted (leaked eager-commit)"
        );
        let mut leaked_header = BlockHeader {
            height: next_h,
            epoch: next_h / crate::epoch::EPOCH_LENGTH,
            parent_hash,
            state_root: drifted_root,
            transactions_root: [0u8; 32],
            receipts_root: [0u8; 32],
            proposer: ids[0], // us — the superseded produce
            timestamp_ms: parent_ts + 6000,
            proposer_signature: vec![],
        };
        let leaked_bh = block_hash(&leaked_header);
        leaked_header.proposer_signature = kps[0].sign(&leaked_bh).to_vec();
        let mut revert: solen_execution::executor::BlockRevert = std::collections::HashMap::new();
        revert.insert(phantom.clone(), None); // undo = delete (no prior value)
        engine
            .v2_blocks
            .write()
            .unwrap()
            .entry(next_h)
            .or_default()
            .insert(
                leaked_bh,
                PendingBlock {
                    header: leaked_header,
                    operations: vec![],
                    proposed_at: std::time::Instant::now(),
                    already_executed: true,
                    result: None,
                    revert: Some(revert),
                    mismatch_count: 0,
                },
            );

        // A peer (ids[1]) proposes the CANONICAL block at next_h: empty block, so
        // its true root equals the clean tip root.
        let mut canonical = BlockHeader {
            height: next_h,
            epoch: next_h / crate::epoch::EPOCH_LENGTH,
            parent_hash,
            state_root: tip_root,
            transactions_root: [0u8; 32],
            receipts_root: [0u8; 32],
            proposer: ids[1],
            timestamp_ms: parent_ts + 6000,
            proposer_signature: vec![],
        };
        let canonical_hash = block_hash(&canonical);
        canonical.proposer_signature = kps[1].sign(&canonical_hash).to_vec();
        assert_ne!(canonical_hash, leaked_bh);
        assert!(
            engine.accept_block(&canonical, &[]),
            "canonical peer block accepted as candidate"
        );

        // 4/5 vote for the canonical hash (2/3 quorum) → v2 finalizes it via
        // finalize_pending_block.
        engine.accept_attestation(ids[1], next_h, canonical_hash);
        engine.accept_attestation(ids[2], next_h, canonical_hash);
        engine.accept_attestation(ids[3], next_h, canonical_hash);
        engine.accept_attestation(ids[4], next_h, canonical_hash);

        // With the accept-path restore, the canonical block finalized despite the
        // leaked eager-commit; without it, the node would be stuck at head_h.
        assert_eq!(
            engine.height(),
            next_h,
            "canonical peer block finalized — accept path restored the store first (no wedge)"
        );
        assert_eq!(
            block_hash(&engine.get_block(next_h).unwrap().header),
            canonical_hash,
            "the finalized block is the canonical one"
        );
        assert!(
            !engine.resync_requested(),
            "must NOT latch into resync — the store was locally repairable"
        );
        assert!(
            engine
                .store()
                .read()
                .unwrap()
                .get(&phantom)
                .unwrap()
                .is_none(),
            "the leaked eager write was undone by the accept-path restore"
        );
    }

    /// Regression (mainnet 2026-08-08, validator9/11 hang): after a snapshot
    /// resync, `reset_to_height` installs a marker block for the restored height.
    /// Its `result.state_root` MUST carry the real restored root, because the
    /// accept-path guard `restore_store_to_finalized_tip()` reads exactly that
    /// field as the finalized-tip target. If the marker leaves it `[0;32]` while
    /// the store holds the real root, the guard sees false "drift", finds nothing
    /// to undo, and latches `needs_resync` — which re-triggers snapshot resync in
    /// an infinite loop that hangs the node (RPC-dead, CPU-spinning). The boot
    /// path (with_validators) already sets the real root here — which is why a
    /// restart clears the hang; this keeps the live-resync path consistent. The
    /// v5 soak never snapshot-resynced, so this path had zero coverage.
    #[test]
    fn resync_marker_result_root_prevents_false_drift_resync_loop() {
        let ids: Vec<[u8; 32]> = (0..3).map(|_| Keypair::generate().public_key()).collect();
        let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
        let config = EngineConfig {
            validator_id: ids[0],
            fork_choice_v2_height: 0,
            ..Default::default()
        };

        // A store with real, non-empty state so the restored root is not [0;32].
        let mut store = MemoryStore::new();
        let seed_acct = {
            let mut id = [0u8; 32];
            id[..4].copy_from_slice(b"seed");
            id
        };
        apply_genesis(
            &mut store,
            vec![GenesisAccount {
                id: seed_acct,
                balance: 1_000_000,
                auth_methods: vec![],
            }],
        )
        .unwrap();
        let engine =
            ConsensusEngine::with_validators(config, Box::new(store), Mempool::new(1000), vs);
        let real_root = engine.store().read().unwrap().state_root();
        assert_ne!(real_root, [0u8; 32], "store holds real genesis state");

        // Simulate a completed snapshot resync at some restored height: the store
        // already holds `real_root`; reset_to_height installs the marker block.
        engine.reset_to_height(5, 0);

        // (1) The marker's result.state_root must equal the real restored root —
        // the exact field restore_store_to_finalized_tip targets. With the old
        // [0;32] marker this assertion fails.
        let marker_result_root = engine
            .chain
            .read()
            .unwrap()
            .last()
            .unwrap()
            .result
            .state_root;
        assert_eq!(
            marker_result_root, real_root,
            "resync marker result.state_root must carry the real restored root"
        );

        // (2) Behavioral proof: the accept-path guard must see NO drift on a
        // freshly restored store and must NOT latch a resync. With the old marker
        // it latches needs_resync → the infinite resync loop / hang.
        assert!(
            !engine.resync_requested(),
            "precondition: no resync pending"
        );
        engine.restore_store_to_finalized_tip();
        assert!(
            !engine.resync_requested(),
            "a freshly snapshot-restored store must not be seen as drifted (no false resync loop)"
        );
        assert_eq!(
            engine.store().read().unwrap().state_root(),
            real_root,
            "restore must not mutate a clean restored store"
        );
    }

    /// Regression (mainnet halt 2026-08-29, block 1513793): v2_leader must be a
    /// TOTAL order. Two candidates at one height from the SAME proposer (same rank)
    /// with EQUAL stake — the empty vs. non-empty variant a proposer emits when a
    /// gapped-nonce tx is included then dropped — were previously ordered only by
    /// HashMap iteration order (randomized per process), so different nodes elected
    /// different leaders and consensus livelocked. The hash tiebreak makes the
    /// winner deterministic (min hash) fleet-wide. Rebuilding the tally HashMap each
    /// iteration varies its order; the elected leader must be identical every time.
    #[test]
    fn v2_leader_is_deterministic_on_equal_stake_same_rank_ties() {
        let kps: Vec<Keypair> = (0..5).map(|_| Keypair::generate()).collect();
        let ids: Vec<[u8; 32]> = kps.iter().map(|k| k.public_key()).collect();
        let vs = ValidatorSet::new(ids.iter().map(|id| ValidatorInfo::new(*id, 100)).collect());
        let config = EngineConfig {
            validator_id: ids[0],
            fork_choice_v2_height: 0,
            ..Default::default()
        };
        let engine = ConsensusEngine::with_validators(
            config,
            Box::new(MemoryStore::new()),
            Mempool::new(1000),
            vs,
        );

        // Two variants of height 1 from the SAME proposer (ids[0]) -> same rank,
        // different transactions_root -> different block hashes (A0 vs A1).
        let h = 1u64;
        let mk = |troot: [u8; 32]| BlockHeader {
            height: h,
            epoch: 0,
            parent_hash: [0u8; 32],
            state_root: [7u8; 32],
            transactions_root: troot,
            receipts_root: [0u8; 32],
            proposer: ids[0],
            timestamp_ms: 1000,
            proposer_signature: vec![],
        };
        let hdr_a = mk([0u8; 32]);
        let hdr_b = mk([1u8; 32]);
        let ha = block_hash(&hdr_a);
        let hb = block_hash(&hdr_b);
        assert_ne!(ha, hb);
        let pb = |header: BlockHeader| PendingBlock {
            header,
            operations: vec![],
            proposed_at: std::time::Instant::now(),
            already_executed: false,
            result: None,
            revert: None,
            mismatch_count: 0,
        };
        {
            let mut v2 = engine.v2_blocks.write().unwrap();
            let e = v2.entry(h).or_default();
            e.insert(ha, pb(hdr_a));
            e.insert(hb, pb(hdr_b));
        }
        let expected = std::cmp::min(ha, hb);

        // Equal stake (2 distinct voters each = 200 vs 200), same rank (same
        // proposer). Only the hash tiebreak can decide -> always the min hash.
        for _ in 0..64 {
            let mut tally: HashMap<Hash, Vec<ValidatorId>> = HashMap::new();
            tally.insert(ha, vec![ids[1], ids[2]]);
            tally.insert(hb, vec![ids[3], ids[4]]);
            assert_eq!(
                engine.v2_leader(h, &tally),
                Some(expected),
                "v2_leader must deterministically elect the min-hash candidate on an equal-stake, same-rank tie"
            );
        }
    }
}

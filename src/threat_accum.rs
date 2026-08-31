//! Threat accumulator stack — a standard lazy NNUE accumulator-stack (push a
//! frame per move, materialise perspectives on demand).
//!
//! Separate from the PSQ accumulator. Each ply has:
//! - Per-perspective i16 accumulator values (aligned)
//! - Per-perspective accuracy flags
//! - Threat deltas (ArrayVec, no heap allocation)
//! - Move info for king mirror detection
//!
//! The stack is pre-allocated for MAX_PLY entries. Push/pop is trivial.
//! BoardObserver callbacks during make_move push deltas directly.
//! Evaluate walks back to find an accurate ancestor and replays forward.

use crate::threats::{RawThreatDelta, MAX_THREAT_DELTAS};
use crate::types::*;

/// Experiment predicate (env CODA_THREAT_REFRESH_ALWAYS, read once): replace
/// the delta-generation + walkback-replay pipeline with full re-enumeration
/// at every materialization. search.rs also consults this to skip
/// `generate_threat_deltas`, so generation and replay stay consistent.
#[inline]
pub fn refresh_always() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("CODA_THREAT_REFRESH_ALWAYS").is_ok())
}

/// Generate threat deltas eagerly inside `make_move` (env
/// `CODA_EAGER_THREAT_DELTAS`, read once) instead of on first replay.
///
/// Lazy is the default: 54.7% of eagerly generated deltas were never consumed,
/// because most children are cut or pruned before anything asks for their
/// accumulator. The two modes must produce byte-identical accumulators, so this
/// exists to A/B them — bench node counts have to match exactly either way, and
/// a difference is a bug in the lazy path rather than a tuning question.
#[inline]
pub fn eager_generation() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("CODA_EAGER_THREAT_DELTAS").is_ok())
}

/// Per-search threat-REFRESH mode: no per-move delta generation, accumulator
/// re-enumerates instead of replaying. Set once at search setup from the root
/// piece count (see `THREAT_REFRESH_PIECE_MAX`) and read by both sides of the
/// contract — the generator (`board.generate_threat_deltas`) and the consumer
/// (`ensure_computed`). They MUST agree: replaying from deltas that were never
/// generated would silently produce a wrong accumulator, so this is decided
/// once per search rather than per node.
pub static REFRESH_MODE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// True when this search should refresh rather than replay.
pub fn refresh_mode() -> bool {
    refresh_always() || REFRESH_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Pre-allocated depth of the threat accumulator stack.
///
/// Derived from the search's own ply cap rather than hardcoded: `search::MAX_PLY`
/// bounds the search stack, so entries beyond it were unreachable. The old 256
/// was 96 entries of pure waste — and not cheap waste: `ThreatEntry` is ~4.6 KB
/// (2 x FT i16 accumulators + a 128-entry DeltaVec), so the stack was ~1158 KB,
/// which EXCEEDS the ~1 MB per-process share of a 16 MB L3 at conc=16. Every
/// make/unmake writes into this structure, so it competes directly with the NNUE
/// weight rows for the same cache — and weight eviction is the measured cause of
/// our contended-NPS deficit (misses/Kinstr 0.100 -> 2.525 from conc1 to conc16).
///
/// `+8` is slack for QS chains that recurse past the nominal cap. Under-sizing is
/// safe regardless: `push()` grows the Vec on demand, so exceeding this costs one
/// reallocation, never correctness.
const MAX_PLY: usize = crate::search::MAX_PLY + 8;

/// Maximum FT (threat-accumulator) hidden size we support inline.
/// Production v9 uses 768; FT=1024 architecture probes need 1024.
/// Sized as a power of two to keep SIMD chunk paths well-tiled.
pub const MAX_FT_SIZE: usize = 1024;
/// Max active threat features collected by `refresh` (and the runtime verifier)
/// in one position. Measured max over bench is ~70 (avg ~7), so 1024 is ~14x
/// headroom — the truncation guard below is effectively unreachable. Both the
/// refresh collector AND the verifier scratch use this SAME bound so the
/// verifier can never be blind to a refresh truncation (a prior 256 cap on
/// both silently agreed on a truncated accumulator).
pub const MAX_ACTIVE_THREAT_FEATURES: usize = 1024;

/// Per-ply pawn-pair deltas. A separate list from `DeltaVec` because
/// `RawThreatDelta` is attacker/victim-shaped; see pawn_pair.rs.
#[derive(Clone, Copy)]
pub struct PPDeltaVec {
    data: [crate::pawn_pair::PawnPairDelta; crate::pawn_pair::MAX_PAWN_PAIR_DELTAS],
    len: usize,
}

impl Default for PPDeltaVec {
    fn default() -> Self { Self::new() }
}

impl PPDeltaVec {
    pub const fn new() -> Self {
        Self {
            data: [crate::pawn_pair::PawnPairDelta::ZERO;
                   crate::pawn_pair::MAX_PAWN_PAIR_DELTAS],
            len: 0,
        }
    }
    #[inline]
    pub fn copy_from_slice(&mut self, src: &[crate::pawn_pair::PawnPairDelta]) {
        // MAX_PAWN_PAIR_DELTAS is a proven bound (see pawn_pair.rs), not a
        // heuristic, so a breach is a logic error rather than a rare position.
        debug_assert!(src.len() <= self.data.len(),
            "pawn-pair delta overflow: {} > {}", src.len(), self.data.len());
        let n = src.len().min(self.data.len());
        self.data[..n].copy_from_slice(&src[..n]);
        self.len = n;
    }
    #[inline] pub fn as_slice(&self) -> &[crate::pawn_pair::PawnPairDelta] { &self.data[..self.len] }
    #[inline] pub fn clear(&mut self) { self.len = 0; }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }
}

/// Fixed-capacity array (no heap, like ArrayVec but simpler).
/// Tracks overflow so callers can force full recompute instead of
/// silently using incomplete deltas.
#[derive(Clone)]
pub struct DeltaVec {
    data: [RawThreatDelta; MAX_THREAT_DELTAS],
    len: usize,
    overflowed: bool,
}

impl Default for DeltaVec {
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaVec {
    pub const fn new() -> Self {
        Self {
            data: [RawThreatDelta::ZERO; MAX_THREAT_DELTAS],
            len: 0,
            overflowed: false,
        }
    }

    #[inline]
    pub fn clear(&mut self) { self.len = 0; self.overflowed = false; }

    #[inline]
    pub fn push(&mut self, d: RawThreatDelta) {
        if self.len < MAX_THREAT_DELTAS {
            self.data[self.len] = d;
            self.len += 1;
        } else {
            self.overflowed = true;
        }
    }

    #[inline]
    pub fn copy_from_slice(&mut self, src: &[RawThreatDelta]) {
        let n = src.len().min(MAX_THREAT_DELTAS);
        self.len = n;
        self.overflowed = src.len() > MAX_THREAT_DELTAS;
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.data.as_mut_ptr(), n);
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[RawThreatDelta] { &self.data[..self.len] }

    #[inline]
    pub fn len(&self) -> usize { self.len }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    #[inline]
    pub fn overflowed(&self) -> bool { self.overflowed }
}

/// Single threat accumulator entry (one ply).
#[repr(C, align(64))]
pub struct ThreatEntry {
    /// Per-perspective accumulator values: [WHITE][..h], [BLACK][..h]
    pub values: [[i16; MAX_FT_SIZE]; 2], // sized for v9 (768) + FT=1024 probes
    /// Per-perspective accuracy flags
    pub accurate: [bool; 2],
    /// Threat deltas for the move that produced this ply
    pub delta: DeltaVec,
    /// Pawn-pair deltas for the same move. Shares `deltas_valid` with `delta`:
    /// both are produced by the same eager/lazy machinery in one step.
    pub pp_delta: PPDeltaVec,
    /// The move that produced this ply (for king mirror check)
    pub mv: Move,
    /// Piece type that moved (for king mirror detection)
    pub moved_pt: u8,
    /// Color that moved (for per-perspective king mirror check)
    pub moved_color: u8,
    /// Piece type captured by `mv` (NO_PIECE_TYPE if none). Recorded so the
    /// stack can walk its own piece state backwards without consulting
    /// `Board::undo_stack` — that keeps the walk-back independent of how the
    /// search's undo stack is indexed, and makes null moves fall out for free
    /// (they carry `mv == NO_MOVE` and are simply skipped).
    pub captured: u8,
    /// Whether `delta` holds this ply's deltas yet. Distinct from "empty":
    /// under lazy generation an entry starts with no deltas and acquires them
    /// on first replay. Caching them here is what makes lazy generation a win —
    /// regenerating per replay rather than per move would save ~11% of the
    /// generation work instead of ~55%.
    pub deltas_valid: bool,
    /// Diagnostic (profile-threats only): whether this generation instance's
    /// deltas were ever replayed. Sized into struct padding; unused in prod.
    pub consumed: bool,
}

impl Default for ThreatEntry {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreatEntry {
    pub const fn new() -> Self {
        Self {
            values: [[0i16; MAX_FT_SIZE]; 2],
            accurate: [false; 2],
            delta: DeltaVec::new(),
            pp_delta: PPDeltaVec::new(),
            mv: NO_MOVE,
            moved_pt: NO_PIECE_TYPE,
            moved_color: WHITE,
            captured: NO_PIECE_TYPE,
            deltas_valid: false,
            consumed: false,
        }
    }
}

/// The threat accumulator stack.
pub struct ThreatStack {
    stack: Vec<ThreatEntry>,
    index: usize,
    hidden_size: usize,
    /// Reusable buffer for lazily regenerated deltas — avoids an allocation
    /// per materialisation.
    scratch: Vec<RawThreatDelta>,
    /// Whether threat features are active (net has threats)
    pub active: bool,
    /// Pawn-pair feature count (0 = net has no pawn-pair block). Carried here
    /// rather than threaded through every refresh/update signature; the
    /// pawn-pair block lives at offset `num_threat_features` in the same
    /// weight array, so `num_features` doubles as its base.
    pub pp_features: usize,
    /// Reusable scratch for lazily regenerated pawn-pair deltas.
    pp_scratch: Vec<crate::pawn_pair::PawnPairDelta>,
}

impl ThreatStack {
    pub fn new(hidden_size: usize) -> Self {
        let mut stack = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            stack.push(ThreatEntry::new());
        }
        Self { stack, index: 0, hidden_size, active: false, pp_features: 0,
               scratch: Vec::with_capacity(MAX_THREAT_DELTAS),
               pp_scratch: Vec::with_capacity(crate::pawn_pair::MAX_PAWN_PAIR_DELTAS) }
    }

    #[inline]
    pub fn index(&self) -> usize { self.index }

    #[inline]
    pub fn current(&self) -> &ThreatEntry { &self.stack[self.index] }

    #[inline]
    pub fn current_mut(&mut self) -> &mut ThreatEntry { &mut self.stack[self.index] }

    /// Copy `Board::threat_deltas` into the current entry after a successful
    /// `make_move`, and record the move metadata needed by mirror checks.
    #[inline]
    pub fn absorb_deltas(&mut self, board: &crate::board::Board) {
        #[cfg(feature = "profile-threats")]
        crate::threats::apply_stats::record_generated(board.threat_deltas.len());
        let eager = board.generate_threat_deltas;
        let entry = self.current_mut();

        // Move metadata is recorded either way: under lazy generation it is
        // what lets `materialize_deltas` walk the piece state back to this ply.
        if let Some(undo) = board.undo_stack.last() {
            entry.mv = undo.mv;
            entry.captured = undo.captured;
            if undo.mv != NO_MOVE {
                entry.moved_pt = board.mailbox[move_to(undo.mv) as usize];
                entry.moved_color = flip_color(board.side_to_move);
            }
        }

        if eager {
            entry.delta.copy_from_slice(&board.threat_deltas);
            entry.pp_delta.copy_from_slice(&board.pawn_pair_deltas);
            entry.deltas_valid = true;
        } else {
            entry.deltas_valid = false;
        }
    }

    /// Give every entry in `from_ply..=self.index` its deltas, regenerating any
    /// that lazy generation left absent.
    ///
    /// Returns false if a regenerated entry overflowed its delta capacity, in
    /// which case the caller must refresh instead of replaying — the same
    /// contract `can_update`'s overflow check enforces for eager deltas.
    ///
    /// Walks the piece state backwards from the LIVE board (which sits at
    /// `self.index`) to just before `from_ply`, then forward again re-emitting.
    /// The scratch state is a 128-byte copy, so the live board — which the NNUE
    /// accumulator stack and the Zobrist keys both alias — is never touched.
    fn materialize_deltas(&mut self, board: &crate::board::Board, from_ply: usize) -> bool {
        let to_ply = self.index;
        if from_ply > to_ply {
            return true;
        }

        // Fast path: the span is already covered, either because generation was
        // eager or because an earlier replay over these plies already paid for
        // it. This is the cache that makes lazy generation worth doing.
        let mut needed = false;
        for p in from_ply..=to_ply {
            let e = &self.stack[p];
            if e.mv != NO_MOVE && !e.deltas_valid {
                needed = true;
                break;
            }
        }
        if !needed {
            return true;
        }

        let mut st = crate::threats::PieceState::from_board(board);
        for p in (from_ply..=to_ply).rev() {
            let e = &self.stack[p];
            if e.mv != NO_MOVE {
                crate::threats::undo_move_state(&mut st, e.moved_color, e.mv, e.captured);
            }
        }

        let mut scratch = std::mem::take(&mut self.scratch);
        let mut ok = true;
        for p in from_ply..=to_ply {
            let (mv, captured, color, valid) = {
                let e = &self.stack[p];
                (e.mv, e.captured, e.moved_color, e.deltas_valid)
            };
            if mv == NO_MOVE {
                continue;
            }
            // Always replayed, because even an entry whose deltas are already
            // cached has to advance `st` past its move. Regenerating a cached
            // entry's deltas and discarding them only happens on a partially
            // covered span, which is rare and at most a ply or two wide.
            // Pawn-pair deltas come from the state BEFORE the move, so they
            // must be generated ahead of replay_move_deltas, which advances
            // `st`. Same generator as the eager path, so the two agree by
            // construction.
            if self.pp_features > 0 && !valid {
                let mut pp = std::mem::take(&mut self.pp_scratch);
                crate::pawn_pair::push_pawn_pair_deltas(
                    &mut pp, st.pieces[crate::types::PAWN as usize], &st.colors,
                    color, mv, captured,
                    st.mailbox[crate::types::move_from(mv) as usize]);
                self.stack[p].pp_delta.copy_from_slice(&pp);
                self.pp_scratch = pp;
            }
            crate::threats::replay_move_deltas(&mut st, color, mv, captured, &mut scratch);
            if !valid {
                let e = &mut self.stack[p];
                e.delta.copy_from_slice(&scratch);
                e.deltas_valid = true;
                if e.delta.overflowed() {
                    ok = false;
                }
            }
        }
        self.scratch = scratch;
        ok
    }

    /// Push: increment index, reset flags, clear deltas.
    /// Called BEFORE make_move (standard accumulator-stack push).
    pub fn push(&mut self, mv: Move, moved_pt: u8) {
        self.index += 1;
        if self.index >= self.stack.len() {
            self.stack.push(ThreatEntry::new());
        }
        let entry = &mut self.stack[self.index];
        entry.accurate = [false; 2];
        entry.delta.clear();
        entry.pp_delta.clear();
        entry.deltas_valid = false;
        entry.mv = mv;
        entry.moved_pt = moved_pt;
        #[cfg(feature = "profile-threats")]
        { entry.consumed = false; }
    }

    /// Pop: decrement index. Saturates at 0 — if push/pop balance is
    /// off, a stray pop stays put rather than wrapping to usize::MAX
    /// and crashing on the next slice access. The debug_assert still
    /// catches the bug in dev builds; release silently no-ops at the
    /// boundary.
    pub fn pop(&mut self) {
        debug_assert!(self.index > 0);
        self.index = self.index.saturating_sub(1);
    }

    /// Reset: for new positions (between bench positions, new game).
    pub fn reset(&mut self) {
        self.index = 0;
        self.stack[0].accurate = [false; 2];
    }

    /// Mark both perspectives stale so the next `ensure_computed` does a full
    /// refresh. Used by the eval-bench microbench to isolate threat-refresh
    /// cost, and by the UCI `eval` command, which evaluates an arbitrary
    /// position that the stack was never walked to.
    pub fn invalidate(&mut self) {
        self.stack[self.index].accurate = [false; 2];
    }

    /// Full refresh for one perspective: zero + enumerate all threats.
    /// Collects feature indices first, then applies with SIMD.
    pub fn refresh(&mut self, net_weights: &[i8], num_features: usize,
                   board: &crate::board::Board, pov: Color) {
        let h = self.hidden_size;
        let entry = &mut self.stack[self.index];
        let p = pov as usize;
        entry.values[p][..h].fill(0);

        let occ = board.colors[0] | board.colors[1];
        let king_sq = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
        let mirrored = (king_sq % 8) >= 4;

        // Collect feature indices, then apply with SIMD. MaybeUninit skips the
        // 2 KB zero-init per refresh — `indices[..n_indices]` is fully written
        // by enumerate_threats below; consumers only read that prefix.
        let mut indices_storage =
            std::mem::MaybeUninit::<[usize; MAX_ACTIVE_THREAT_FEATURES]>::uninit();
        let indices_ptr = scratch_ptr!(indices_storage, usize);
        let mut n_indices = 0usize;
        // Track whether the enumerator produced more
        // features than the buffer can hold. Excess features would be dropped
        // and only `accurate[p]=false` (so children re-refresh) — but THIS
        // node's eval would still consume the truncated accumulator. With the
        // buffer at MAX_ACTIVE_THREAT_FEATURES (1024, ~14x the measured max of
        // 70) this is unreachable; the guard remains as defense-in-depth and
        // the debug_assert below makes any future breach loud instead of
        // silently wrong. Note: a full refresh does NOT self-correct a
        // truncation — the "next full path" is this same refresh — so the cap
        // must simply be large enough never to trip.
        let mut overflowed = false;

        crate::threats::enumerate_threats(
            &board.pieces, &board.colors, &board.mailbox,
            occ, pov, mirrored,
            |feat_idx| {
                if feat_idx < num_features {
                    if n_indices < MAX_ACTIVE_THREAT_FEATURES {
                        unsafe { indices_ptr.add(n_indices).write(feat_idx); }
                        n_indices += 1;
                    } else {
                        overflowed = true;
                    }
                }
            },
        );
        debug_assert!(
            !overflowed,
            "threat refresh exceeded MAX_ACTIVE_THREAT_FEATURES ({}) — raise the cap",
            MAX_ACTIVE_THREAT_FEATURES
        );

        #[cfg(feature = "profile-threats")]
        crate::threats::refresh_stats::record(n_indices, overflowed);

        // Pawn-pair features occupy the shared feature space above the threat
        // block, so their rows live in the SAME weight array and can simply be
        // appended to the index list -- no second weight pass, and pack time
        // stays untouched.
        if self.pp_features > 0 {
            crate::pawn_pair::enumerate_pawn_pairs(
                board.pieces[crate::types::PAWN as usize], &board.colors, pov, mirrored,
                |pp_idx| {
                    if n_indices < MAX_ACTIVE_THREAT_FEATURES {
                        unsafe { indices_ptr.add(n_indices).write(num_features + pp_idx); }
                        n_indices += 1;
                    } else {
                        overflowed = true;
                    }
                },
            );
        }

        // Apply all weight rows with SIMD
        let indices = scratch_slice!(indices_ptr, n_indices);
        crate::threats::add_weight_rows(
            &mut entry.values[p][..h], net_weights, h, indices,
        );

        entry.accurate[p] = !overflowed;
    }

    /// Check if we can incrementally update this perspective by walking back.
    /// Returns Some(ancestor_index) or None (need full refresh).
    /// The condition for whether threats can be refreshed incrementally from an
    /// ancestor rather than fully recomputed.
    #[inline]
    pub fn can_update(&self, pov: Color) -> Option<usize> {
        for i in (0..self.index).rev() {
            // Validate the move that produced entry[i+1] BEFORE accepting
            // entry[i] as an ancestor. If the move at i+1 is a king crossing
            // (changes mirror for this perspective) or has overflowed deltas,
            // we cannot replay from any ancestor at or below i — the stored
            // deltas would apply with the wrong mirror or be incomplete.
            //
            // Do NOT return Some(i) on `accurate[i]` before doing this check:
            // that lets a king-file crossing at the current ply slip through
            // whenever the prior ply was accurate — the common case.
            let entry = &self.stack[i + 1];
            if entry.mv != NO_MOVE {
                // Under lazy generation overflow is unknown until the deltas
                // are built, so an absent entry is treated as fine here and
                // re-checked by `materialize_deltas`.
                if entry.deltas_valid && entry.delta.overflowed() {
                    #[cfg(feature = "profile-threats")]
                    crate::threats::apply_stats::record_refresh_cause(2);
                    return None;
                }
                if entry.moved_pt == KING && entry.moved_color == pov {
                    let from = move_from(entry.mv);
                    let to = move_to(entry.mv);
                    if (from % 8 >= 4) != (to % 8 >= 4) {
                        // This perspective's king crossed e-file — mirroring changed.
                        #[cfg(feature = "profile-threats")]
                        crate::threats::apply_stats::record_refresh_cause(0);
                        return None;
                    }
                }
            }

            if self.stack[i].accurate[pov as usize] {
                return Some(i);
            }
        }
        #[cfg(feature = "profile-threats")]
        crate::threats::apply_stats::record_refresh_cause(1);
        None
    }

    /// Incremental update: replay from ancestor to current index for one perspective.
    /// Uses SIMD apply_threat_deltas for the inner loop (AVX2 register tiling).
    pub fn update(&mut self, ancestor: usize, net_weights: &[i8], num_features: usize,
                  board: &crate::board::Board, pov: Color) {
        let h = self.hidden_size;
        let p = pov as usize;
        let king_sq = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
        let mirrored = (king_sq % 8) >= 4;

        #[cfg(feature = "profile-threats")]
        crate::threats::apply_stats::record_replay_gap(self.index - ancestor);

        // Cross-ply cancellation measurement (double_inc_update upside #1): for
        // gap>=2 replays, gather all valid delta indices across the span and
        // count net-zero add/sub pairs over the COMBINED multiset. Compared to
        // the per-ply 3.8%, the excess is the cross-ply cancellation double_inc
        // would capture. Profile-only.
        #[cfg(feature = "profile-threats")]
        if self.index - ancestor >= 2 {
            let mut adds: Vec<usize> = Vec::new();
            let mut subs: Vec<usize> = Vec::new();
            for ply in (ancestor + 1)..=self.index {
                for d in &self.stack[ply].delta.data[..self.stack[ply].delta.len] {
                    let idx = crate::threats::threat_index(
                        d.attacker_cp() as usize, d.from_sq() as u32,
                        d.victim_cp() as usize, d.to_sq() as u32, mirrored, pov);
                    if idx < 0 || (idx as usize) >= num_features { continue; }
                    if d.add() { adds.push(idx as usize); } else { subs.push(idx as usize); }
                }
            }
            crate::threats::apply_stats::record_crossply(&adds, &subs);
        }

        for ply in (ancestor + 1)..=self.index {
            let entry_mv = self.stack[ply].mv;

            #[cfg(feature = "profile-threats")]
            if entry_mv != NO_MOVE && !self.stack[ply].consumed {
                self.stack[ply].consumed = true;
                crate::threats::apply_stats::record_first_consume();
            }

            if entry_mv == NO_MOVE || self.stack[ply].delta.is_empty() {
                // Null move or no threat deltas: copy from previous. Pawn-pair
                // deltas are applied below regardless -- a move can change the
                // pawn structure while producing no threat delta.
                let (prev, curr) = self.stack.split_at_mut(ply);
                curr[0].values[p][..h].copy_from_slice(&prev[ply - 1].values[p][..h]);
            } else {
                // Use SIMD apply_threat_deltas (copies src + applies adds/subs)
                let (prev, curr) = self.stack.split_at_mut(ply);
                let entry = &mut curr[0];
                let local_deltas = entry.delta.as_slice();
                unsafe {
                    crate::threats::apply_threat_deltas(
                        &mut entry.values[p][..h],
                        &prev[ply - 1].values[p][..h],
                        local_deltas,
                        net_weights, h, num_features,
                        pov, mirrored,
                    );
                }
            }

            if self.pp_features > 0 && !self.stack[ply].pp_delta.is_empty() {
                let entry = &mut self.stack[ply];
                let pp = entry.pp_delta;
                crate::pawn_pair::apply_pawn_pair_deltas(
                    &mut entry.values[p][..h], pp.as_slice(),
                    net_weights, h, num_features, pov, mirrored);
            }

            self.stack[ply].accurate[p] = true;
        }
    }

    /// Incremental update for both perspectives when they share an ancestor.
    ///
    /// The replay validity checks are still performed by `can_update` before
    /// this is called. This path only removes the duplicate raw-delta walk; the
    /// SIMD accumulator apply remains per perspective.
    pub fn update_dual(&mut self, ancestor: usize, net_weights: &[i8], num_features: usize,
                       board: &crate::board::Board) {
        let h = self.hidden_size;
        let white_king_sq = (board.pieces[KING as usize] & board.colors[WHITE as usize]).trailing_zeros();
        let black_king_sq = (board.pieces[KING as usize] & board.colors[BLACK as usize]).trailing_zeros();
        let mirrored_w = (white_king_sq % 8) >= 4;
        let mirrored_b = (black_king_sq % 8) >= 4;
        let pp = self.pp_features;

        #[cfg(feature = "profile-threats")]
        {
            crate::threats::apply_stats::record_replay_gap(self.index - ancestor);
            crate::threats::apply_stats::record_replay_gap(self.index - ancestor);
        }

        for ply in (ancestor + 1)..=self.index {
            let entry_mv = self.stack[ply].mv;

            #[cfg(feature = "profile-threats")]
            if entry_mv != NO_MOVE && !self.stack[ply].consumed {
                self.stack[ply].consumed = true;
                crate::threats::apply_stats::record_first_consume();
            }

            let (prev, curr) = self.stack.split_at_mut(ply);
            let prev_entry = &prev[ply - 1];
            let entry = &mut curr[0];

            if entry_mv == NO_MOVE || entry.delta.is_empty() {
                entry.values[WHITE as usize][..h]
                    .copy_from_slice(&prev_entry.values[WHITE as usize][..h]);
                entry.values[BLACK as usize][..h]
                    .copy_from_slice(&prev_entry.values[BLACK as usize][..h]);
            } else {
                let local_deltas = entry.delta.as_slice();
                let (dst_w, dst_b) = {
                    let (w, b) = entry.values.split_at_mut(1);
                    (&mut w[0][..h], &mut b[0][..h])
                };
                unsafe {
                    crate::threats::apply_threat_deltas_dual(
                        dst_w,
                        &prev_entry.values[WHITE as usize][..h],
                        dst_b,
                        &prev_entry.values[BLACK as usize][..h],
                        local_deltas,
                        net_weights, h, num_features,
                        mirrored_w, mirrored_b,
                    );
                }
            }

            if pp > 0 && !entry.pp_delta.is_empty() {
                let ppd = entry.pp_delta;
                crate::pawn_pair::apply_pawn_pair_deltas(
                    &mut entry.values[WHITE as usize][..h], ppd.as_slice(),
                    net_weights, h, num_features, WHITE, mirrored_w);
                crate::pawn_pair::apply_pawn_pair_deltas(
                    &mut entry.values[BLACK as usize][..h], ppd.as_slice(),
                    net_weights, h, num_features, BLACK, mirrored_b);
            }

            entry.accurate = [true, true];
        }
    }

    /// Get the accumulator values for a perspective.
    #[inline]
    pub fn values(&self, pov: Color) -> &[i16] {
        &self.stack[self.index].values[pov as usize][..self.hidden_size]
    }

    /// Ensure both perspectives are computed for the current position.
    /// Standard lazy evaluate: ensure both perspectives are materialised.
    #[inline]
    /// `num_pp` is the net's pawn-pair feature count (0 if it has none). It is
    /// a PARAMETER rather than a field set at construction because a missed
    /// assignment would be silent -- the accumulator would simply omit the
    /// pawn-pair contribution and evaluate a slightly wrong position on every
    /// node. Passing it here makes the compiler check every production site.
    pub fn ensure_computed(&mut self, net_weights: &[i8], num_features: usize,
                          num_pp: usize, board: &crate::board::Board) {
        if !self.active { return; }
        self.pp_features = num_pp;

        // Experiment (CODA_THREAT_REFRESH_ALWAYS): bypass the walkback/replay
        // machinery and re-enumerate from the board every time. Paired with
        // skipping delta generation in make_move (search.rs reads the same
        // predicate), this measures whether the replay path earns its keep —
        // avg active features/position (~6.8) is close to avg delta rows per
        // replayed edge (~7.4), so refresh may cost about the same as a
        // single-edge replay while deleting all generation work.
        if refresh_mode() {
            for pov in [WHITE, BLACK] {
                if !self.stack[self.index].accurate[pov as usize] {
                    self.refresh(net_weights, num_features, board, pov);
                }
            }
            return;
        }

        let idx = self.index;
        if !self.stack[idx].accurate[WHITE as usize] && !self.stack[idx].accurate[BLACK as usize] {
            let white_ancestor = self.can_update(WHITE);
            let black_ancestor = self.can_update(BLACK);
            if let (Some(w), Some(b)) = (white_ancestor, black_ancestor) {
                // On overflow fall through to the per-perspective loop rather
                // than replaying. `materialize_deltas` leaves the offending
                // entry marked valid-and-overflowed, so the `can_update` below
                // now returns None for it and both perspectives refresh.
                if w == b && self.materialize_deltas(board, w + 1) {
                    self.update_dual(w, net_weights, num_features, board);
                    return;
                }
            }
        }

        for pov in [WHITE, BLACK] {
            if self.stack[idx].accurate[pov as usize] {
                continue;
            }

            match self.can_update(pov) {
                Some(ancestor) => {
                    if self.materialize_deltas(board, ancestor + 1) {
                        self.update(ancestor, net_weights, num_features, board, pov);
                    } else {
                        self.refresh(net_weights, num_features, board, pov);
                    }
                }
                None => self.refresh(net_weights, num_features, board, pov),
            }
        }
    }
}

#[cfg(test)]
mod incremental_tests {
    //! Regression tests: the incremental threat update path must produce
    //! the same per-perspective accumulator as a full re-enumeration
    //! after every move. Drives two ThreatStacks side-by-side along the
    //! same move sequence — one always refreshes, the other relies on
    //! the incremental deltas path (ensure_computed → can_update → update).
    //!
    //! Failure mode this targets: capture moves that remove a blocker
    //! between a slider and a piece behind it should register a new
    //! x-ray feature, but the incremental path may miss it.
    //!
    //! Each FEN → moves sequence is a self-contained scenario. The
    //! deterministic weight pattern makes any single feature-level
    //! divergence show up as an element-wise vector diff.
    use super::*;
    use crate::board::Board;
    use crate::movegen::generate_legal_moves;
    use crate::threats::{num_threat_features, RawThreatDelta};

    const H: usize = 768;

    /// Deterministic weights: each (feature, channel) gets a distinct i8.
    /// Ensures a single-feature multiset divergence produces a visible
    /// element-wise delta in the accumulator.
    fn make_weights(num_features: usize) -> Vec<i8> {
        let mut w = vec![0i8; num_features * H];
        for idx in 0..num_features {
            for j in 0..H {
                // Mix idx and channel with primes. Mod 251 keeps values
                // in i8 range while staying well-distributed.
                let v = ((idx.wrapping_mul(7919)).wrapping_add(j.wrapping_mul(31)) % 251) as i32 - 125;
                w[idx * H + j] = v as i8;
            }
        }
        w
    }

    fn parse_uci(board: &Board, s: &str) -> Move {
        let bytes = s.as_bytes();
        assert!(bytes.len() >= 4, "bad uci: {}", s);
        let from_file = bytes[0] - b'a';
        let from_rank = bytes[1] - b'1';
        let to_file = bytes[2] - b'a';
        let to_rank = bytes[3] - b'1';
        let from = crate::types::square(from_file, from_rank);
        let to = crate::types::square(to_file, to_rank);
        let promo_flag = if bytes.len() > 4 {
            match bytes[4] {
                b'q' => Some(FLAG_PROMOTE_Q),
                b'r' => Some(FLAG_PROMOTE_R),
                b'b' => Some(FLAG_PROMOTE_B),
                b'n' => Some(FLAG_PROMOTE_N),
                _ => None,
            }
        } else { None };
        let legal = generate_legal_moves(board);
        for i in 0..legal.len {
            let mv = legal.get(i);
            if move_from(mv) == from && move_to(mv) == to {
                if let Some(pf) = promo_flag {
                    if move_flags(mv) == pf { return mv; }
                } else if !is_promotion(mv) {
                    return mv;
                }
            }
        }
        panic!("no legal move {} in position", s);
    }

    fn absorb_deltas(ts: &mut ThreatStack, board: &mut Board) {
        ts.absorb_deltas(board);
    }

    /// Run the scenario: play each UCI move, verifying after every ply
    /// that incremental == full-refresh for both perspectives.
    /// Same scenario, but with the pawn-pair block active. Pawn-pair features
    /// share the threat weight array above the threat block, so the only
    /// changes are a longer weight vector and `pp_features` set on both stacks.
    ///
    /// Run BOTH generation modes. Eager builds the deltas in `make_move`; lazy
    /// leaves them absent and makes `materialize_deltas` reconstruct them by
    /// walking the piece state backwards. Those are two separate pieces of
    /// code that must agree, and only the eager one is exercised by default.
    fn run_scenario_pp(name: &str, fen: &str, moves: &[&str]) {
        let pp = crate::pawn_pair::PAWN_PAIR_FEATURES;
        run_scenario_inner(name, fen, moves, pp, true);
        run_scenario_inner(name, fen, moves, pp, false);
    }

    fn run_scenario(name2: &str, fen: &str, moves: &[&str], pp: usize) {
        run_scenario_inner(name2, fen, moves, pp, true);
    }

    fn run_scenario_inner(name: &str, fen: &str, moves: &[&str], pp: usize, eager: bool) {
        crate::init();
        let _space = crate::threats::FEATURE_SPACE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let nf = num_threat_features();
        let weights = make_weights(nf + pp);

        let mut board = Board::new();
        board.set_fen(fen);
        board.generate_threat_deltas = eager;
        board.generate_pawn_pair_deltas = pp > 0 && eager;

        let mut incr = ThreatStack::new(H);
        incr.pp_features = pp;
        incr.active = true;
        incr.refresh(&weights, nf, &board, WHITE);
        incr.refresh(&weights, nf, &board, BLACK);

        let mut refs = ThreatStack::new(H);
        refs.pp_features = pp;
        refs.active = true;
        refs.refresh(&weights, nf, &board, WHITE);
        refs.refresh(&weights, nf, &board, BLACK);

        // Sanity: both start identical.
        assert_eq!(incr.values(WHITE), refs.values(WHITE), "{}: baseline W mismatch", name);
        assert_eq!(incr.values(BLACK), refs.values(BLACK), "{}: baseline B mismatch", name);

        for (ply, uci) in moves.iter().enumerate() {
            let mv = parse_uci(&board, uci);

            // Incremental side: push before make, absorb deltas after.
            incr.push(NO_MOVE, NO_PIECE_TYPE);
            // Reference side: push too so indices line up; we'll overwrite
            // with a refresh (no delta replay).
            refs.push(NO_MOVE, NO_PIECE_TYPE);

            let ok = board.make_move(mv);
            assert!(ok, "{}: move {} illegal at ply {}", name, uci, ply);

            absorb_deltas(&mut incr, &mut board);
            incr.ensure_computed(&weights, nf, pp, &board);

            refs.refresh(&weights, nf, &board, WHITE);
            refs.refresh(&weights, nf, &board, BLACK);

            // Compare element-wise and surface first divergence.
            for pov in [WHITE, BLACK] {
                let a = incr.values(pov);
                let b = refs.values(pov);
                if a != b {
                    let mut first = None;
                    for j in 0..H {
                        if a[j] != b[j] { first = Some((j, a[j], b[j])); break; }
                    }
                    let (j, av, bv) = first.unwrap();
                    panic!(
                        "{}: ply={} move={} pov={} first diff at channel {} incr={} refresh={} (delta_count={})",
                        name, ply, uci,
                        if pov == WHITE { "W" } else { "B" },
                        j, av, bv,
                        incr.current().delta.len(),
                    );
                }
            }
        }
    }

    /// The curated scenario corpus — one table so path-forced sweeps
    /// (`incremental_suite_forced_avx2`) can't drift from the individual
    /// tests below, which each run one entry by name.
    const SCENARIOS: &[(&str, &str, &[&str])] = &[
        ("startpos_quiet",
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         &["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]),
        ("simple_captures",
         "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
         &["b8c6", "f1b5", "a7a6", "b5c6", "d7c6"]),
        ("rook_xray_capture",
         "r3k3/8/8/8/p7/8/8/R3K3 w Q - 0 1",
         &["a1a4"]),
        ("bishop_xray_diagonal",
         "7r/8/8/8/3p4/8/8/B3K2k w - - 0 1",
         &["a1d4"]),
        ("queen_xray",
         "3k4/8/8/8/3p4/8/8/3QK3 w - - 0 1",
         &["d1d4"]),
        ("unrelated_capture",
         "k1b5/pp6/8/8/4n3/2B5/PP6/K1R5 w - - 0 1",
         &["c3e5"]),
        ("rook_captures_blocker_with_xray_behind",
         "k7/n7/8/8/p7/8/8/R3K3 w Q - 0 1",
         &["a1a4"]),
        ("capture_then_retreat",
         "k7/n5p1/8/8/p7/8/8/R3K3 w Q - 0 1",
         &["a1a4", "g7g6", "a4a1"]),
        ("kiwipete",
         "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
         &["e5g6", "f7g6", "e2f1", "c7c6", "d5c6"]),
        ("ep_capture",
         "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
         &["e5f6"]),
        ("castle_ks",
         "r3k2r/pppqppbp/2np1np1/4P3/2B5/2NQ1N2/PPP2PPP/R1B1K2R w KQkq - 0 1",
         &["e1g1"]),
        ("castle_qs_phantom",
         "rQr5/p2pkp1n/1n2p1p1/7q/b1P1P3/Np2NB1P/PP3P1P/R3K2R w KQ - 3 15",
         &["e1c1"]),
        ("slider_move_reveals_xray",
         "4k3/8/8/3r4/8/8/3R4/3QK3 w - - 0 1",
         &["d2d4"]),
        ("chain_captures",
         "4k3/8/3p4/4p3/3P4/2N5/8/4K3 w - - 0 1",
         &["d4e5", "d6e5", "c3e4", "e8d8"]),
        ("promotion_capture",
         "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
         &["a7a8q"]),
    ];

    /// Pawn-structure-heavy scenarios, run with the pawn-pair block active.
    /// These are the moves where the pawn SET changes shape: a capture that
    /// removes a pawn, a promotion that removes one without adding one, and en
    /// passant, where the captured pawn is not on the destination square.
    const PP_SCENARIOS: &[(&str, &str, &[&str])] = &[
        ("pp_pawn_storm",
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         &["e2e4", "d7d5", "e4d5", "c7c6", "d5c6", "b8c6", "d2d4", "e7e5", "d4e5"]),
        ("pp_en_passant",
         "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
         &["e5f6", "e7f6", "d2d4", "c7c5", "d4c5"]),
        ("pp_promotion",
         "8/P6P/4k3/8/8/4K3/p6p/8 w - - 0 1",
         &["a7a8q", "a2a1q", "h7h8n", "h2h1n"]),
        // Forces the SINGLE-perspective `update` path, which the other
        // scenarios never reach: they all take the dual fast path because both
        // perspectives share an ancestor. Here the white king CAPTURES a pawn
        // while crossing the d/e file boundary, so white's mirror flips (white
        // refreshes) while black replays -- and the move changes the pawn set,
        // so black's replay carries a non-empty pawn-pair delta list.
        ("pp_king_crosses_mirror_taking_pawn",
         "4k3/8/8/8/8/3P4/PPP1p3/3K4 w - - 0 1",
         &["d1e2"]),
        ("pp_doubled_and_phalanx",
         "4k3/8/8/8/8/2PPP3/2P1P3/4K3 w - - 0 1",
         &["c3c4", "e8d8", "d3d4", "d8c8", "e3e4"]),
    ];

    fn pp_scenario(name: &str) {
        let (n, fen, moves) =
            PP_SCENARIOS.iter().find(|s| s.0 == name).expect("unknown pp scenario");
        run_scenario_pp(n, fen, moves);
    }

    /// With the pawn-pair block active, incremental replay must still equal a
    /// full refresh at every ply. This is the end-to-end check over the whole
    /// path -- delta generation in make_move, lazy regeneration in
    /// materialize_deltas, and both the single and dual replay routines.
    #[test]
    fn pawn_pair_incremental_matches_refresh() {
        for (n, _, _) in PP_SCENARIOS { pp_scenario(n); }
    }

    /// The same, with every existing threat scenario re-run with the block on:
    /// pawn-pair must not disturb the threat path.
    #[test]
    fn pawn_pair_does_not_disturb_threat_scenarios() {
        for (n, fen, moves) in SCENARIOS { run_scenario_pp(n, fen, moves); }
    }

    fn scenario(name: &str) {
        let (n, fen, moves) =
            SCENARIOS.iter().find(|s| s.0 == name).expect("unknown scenario name");
        run_scenario(n, fen, moves, 0);
    }

    #[test]
    fn startpos_quiet_moves() {
        // Sanity: no captures, no x-rays activated.
        scenario("startpos_quiet");
    }

    #[test]
    fn simple_captures_no_xray() {
        // Knight captures with no slider x-ray behind the captured square.
        scenario("simple_captures");
    }

    #[test]
    fn rook_captures_pawn_revealing_xray() {
        // White rook on a1, white pawn gone, black pawn on a7, black king a8.
        // Rook takes pawn on a7 — now rook attacks king (already attacked via a-file).
        // More interesting: set up a rook blocked by enemy piece with enemy piece behind.
        // Position: white rook a1, black pawn a4 (blocker), black rook a8 (x-ray target).
        // Wa1 captures a4 pawn directly; before capture a1→a4 direct, a1→a8 x-ray.
        // After capture (Ra1xa4), rook now on a4; direct Ra4→a8 on a-file.
        scenario("rook_xray_capture");
    }

    #[test]
    fn bishop_xray_through_pawn_captured() {
        // White bishop a1, black pawn d4 (blocker), black rook h8 (x-ray target).
        // Before: Ba1 directly attacks pawn d4; x-ray through d4 reveals... nothing
        // (h8 is on the a1-h8 diagonal, pawn d4 is also on it, rook h8 behind).
        // Capture Bxd4 changes geometry.
        scenario("bishop_xray_diagonal");
    }

    #[test]
    fn queen_xray_orthogonal_and_diagonal() {
        // Queen on d1 with pawn on d4 (blocker) and king on d8 (x-ray target).
        // Capture reveals queen → king x-ray on d-file.
        scenario("queen_xray");
    }

    #[test]
    fn capture_that_opens_third_party_xray() {
        // The tricky case: a capture that doesn't involve the slider at all
        // but removes a piece that was blocking a slider from seeing behind.
        //
        // Setup:
        //   White rook a1, white pawn a4 (its own blocker),
        //   black pawn a5 (gets captured-ish scenario),
        //   black king a8.
        //
        // Better — capture by a different piece:
        //   White rook h1, white bishop c3 (irrelevant),
        //   black rook h8 on open h-file with black pawn h4 blocking (direct & x-ray slot),
        //   black knight g5 that white's bishop will capture.
        //
        // After Bc3xg5 (unrelated capture), the h-file situation is unchanged,
        // so this is a quiet-for-h-file capture. Good negative test.
        //
        // A real third-party x-ray: white rook a1 blocked by white pawn a2
        // from seeing black pawn a7 → black king a8. When something else
        // captures elsewhere, nothing should change on the a-file.
        // (unrelated knight takes would be c3xe4; we move bishop c3-e5)
        scenario("unrelated_capture");
    }

    #[test]
    fn capture_blocker_between_slider_and_third_piece() {
        // Core x-ray-on-capture bug scenario.
        // White rook a1, black pawn a4 (blocker), black knight a7 (x-ray target behind).
        // Before Rxa4: direct a1→a4 pawn; x-ray a1→a7 knight (through pawn a4).
        // After Rxa4: rook now on a4; direct a4→a7 knight.
        // Incremental path must net out the x-ray-through-pawn feature loss.
        scenario("rook_captures_blocker_with_xray_behind");
    }

    #[test]
    fn slider_captures_then_moves_away() {
        // Multi-move scenario: capture then retreat. Exercises back-to-back
        // delta application. The BN on a7 is pinned against BK on a8 once
        // the rook lands on a4, so use a black pawn move between the two
        // white moves to test incremental survival across a black move.
        scenario("capture_then_retreat");
    }

    #[test]
    fn kiwipete_tactical_sequence() {
        // Rich middlegame with many sliders and captures.
        // e5g6: Nxg6 (captures BP). f7g6: black pawn recapture.
        scenario("kiwipete");
    }

    #[test]
    fn en_passant_capture() {
        // EP captures remove a piece from a square other than the move's `to`.
        // Tests push_threats_on_change for EP cap_sq + push_threats_on_move
        // for the pawn (exf6 en passant).
        scenario("ep_capture");
    }

    #[test]
    fn castling_kingside() {
        // Castle moves both king and rook — tests back-to-back deltas
        // plus per-perspective king-file-mirror change (O-O).
        scenario("castle_ks");
    }

    #[test]
    fn castling_queenside_phantom_xray_regression() {
        // Regression for the 2b slider-iteration rewrite: during the rook
        // leg of O-O-O (a1→d1) the moved rook is in pieces_bb at d1 but
        // occ_transit has d1 cleared. Without `& occ` on section-2b
        // candidates, d1 is iterated as a phantom x-ray candidate from
        // sq=a1 (with king@c1 between d1 and a1 → exactly one blocker)
        // and a spurious (rook@d1, wrook, a1) delta is emitted. Caught
        // by fuzz_random_games seed 0xdebd0132 ply 28. (O-O-O)
        scenario("castle_qs_phantom");
    }

    #[test]
    fn slider_move_reveals_x_ray_for_other_slider() {
        // WQ on d1, WR on d2 blocking. WR moves to d5 — WQ's rank/file view
        // shifts: gains direct d-file targets, loses the blocker.
        scenario("slider_move_reveals_xray");
    }

    #[test]
    fn chain_of_captures() {
        // Back-to-back captures — pawn trades leaving the incremental
        // state to absorb multiple small deltas in sequence.
        scenario("chain_captures");
    }

    #[test]
    fn promotion_with_capture() {
        // Pawn captures and promotes — double state change.
        scenario("promotion_capture");
    }

    /// Deterministic fuzzer: plays random legal moves from several
    /// starting positions and asserts incremental == refresh after
    /// every move. Covers move-type combinations that curated scenarios
    /// can miss (EP in slider ray, promotion into pin, castling through
    /// x-ray target, etc.).
    ///
    /// This is the counterpart to the curated scenario list above —
    /// those pin down specific failure modes; this one finds whatever
    /// we haven't thought of. If it fires on a new pattern, add a
    /// curated scenario for that pattern and keep the fuzzer for
    /// regression.
    #[test]
    fn fuzz_random_games() {
        run_fuzz_random_games(true);
    }

    /// Same fuzz, driving the LAZY generation path that production now uses by
    /// default: `make_move` emits nothing and the deltas are rebuilt on first
    /// replay from the stack's own move metadata. Without this the fuzz would
    /// only ever prove the eager path, which is no longer the one that ships.
    #[test]
    fn fuzz_random_games_lazy_generation() {
        run_fuzz_random_games(false);
    }

    fn run_fuzz_random_games(eager: bool) {
        crate::init();
        let _space = crate::threats::FEATURE_SPACE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let nf = num_threat_features();
        let weights = make_weights(nf);

        // Several varied starting positions — opening, kiwipete middle-game,
        // tactical midgame with heavy slider activity, and an endgame.
        const START_FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1", // promotion testbed
        ];

        // Deterministic xorshift32 PRNG — reproducible failures.
        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            x
        }

        const MAX_PLIES_PER_GAME: usize = 120;
        const GAMES_PER_FEN: usize = 20;

        for (fen_idx, fen) in START_FENS.iter().enumerate() {
            for game in 0..GAMES_PER_FEN {
                let seed: u32 = 0xDEADBEEFu32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((game as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };

                let mut board = Board::new();
                board.set_fen(fen);
                board.generate_threat_deltas = eager;

                let mut incr = ThreatStack::new(H);
                incr.active = true;
                incr.refresh(&weights, nf, &board, WHITE);
                incr.refresh(&weights, nf, &board, BLACK);

                let mut refs = ThreatStack::new(H);
                refs.active = true;
                refs.refresh(&weights, nf, &board, WHITE);
                refs.refresh(&weights, nf, &board, BLACK);

                for ply in 0..MAX_PLIES_PER_GAME {
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 {
                        break; // stalemate or checkmate
                    }
                    let idx = (next_u32(&mut rng) as usize) % legal.len;
                    let mv = legal.get(idx);

                    incr.push(NO_MOVE, NO_PIECE_TYPE);
                    refs.push(NO_MOVE, NO_PIECE_TYPE);

                    let ok = board.make_move(mv);
                    assert!(ok, "fuzz {} game {} ply {}: move {} illegal?",
                        fen_idx, game, ply, crate::types::move_to_uci(mv));

                    // Use the production absorb rather than reimplementing it.
                    // This block used to hand-roll the same field assignments,
                    // and when `captured` and `deltas_valid` were added for lazy
                    // generation the copy silently went stale — the walk-back
                    // then mis-inverted every capture. A duplicated absorb is
                    // exactly the drift this fuzz is supposed to catch, not
                    // contain.
                    incr.absorb_deltas(&board);

                    incr.ensure_computed(&weights, nf, 0, &board);
                    refs.refresh(&weights, nf, &board, WHITE);
                    refs.refresh(&weights, nf, &board, BLACK);

                    for pov in [WHITE, BLACK] {
                        let a = incr.values(pov);
                        let b = refs.values(pov);
                        if a != b {
                            // Find first divergent channel for a useful panic message.
                            let mut first = None;
                            for j in 0..H {
                                if a[j] != b[j] {
                                    first = Some((j, a[j], b[j]));
                                    break;
                                }
                            }
                            let (j, av, bv) = first.unwrap();
                            panic!(
                                "fuzz divergence: fen_idx={} game={} ply={} move={} pov={} \
                                 channel={} incr={} refresh={} seed={:#x}",
                                fen_idx, game, ply,
                                crate::types::move_to_uci(mv),
                                if pov == WHITE { "W" } else { "B" },
                                j, av, bv, seed,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Gap fuzzer: the fuzzer above is forward-only
    /// (never pops/unmakes) and calls ensure_computed after EVERY move, so
    /// every replay has gap == 1 and the pop/re-push and gap >= 2 replay
    /// paths (update/update_dual spanning multiple plies, ancestors found
    /// below stale popped entries) were never exercised. This one does a
    /// random DFS walk: push+make+absorb, pop+unmake, and only verifies
    /// (ensure_computed + scratch-enumeration compare) on a random subset
    /// of steps, so gaps of 2..8+ plies and post-pop stale-entry reuse
    /// occur constantly.
    #[test]
    fn fuzz_random_walk_with_pops_and_lazy_gaps() {
        run_fuzz_walk_with_pops(true);
    }

    /// The pops/gaps walk against LAZY generation. This is the sharpest test of
    /// the lazy path: `materialize_deltas` walks the piece state back from the
    /// live board across a replay span, so pushes, pops and multi-ply gaps are
    /// precisely where a mis-stepped walk-back would show up.
    #[test]
    fn fuzz_random_walk_with_pops_and_lazy_gaps_lazy_generation() {
        run_fuzz_walk_with_pops(false);
    }

    fn run_fuzz_walk_with_pops(eager: bool) {
        crate::init();
        let _space = crate::threats::FEATURE_SPACE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let nf = num_threat_features();
        let weights = make_weights(nf);

        const START_FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1",
            // castling-heavy, kings near the e-file mirror boundary
            "r3k2r/pppq1ppp/2n1pn2/3p4/3P4/2N1PN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
        ];

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            x
        }

        // Scratch check: enumerate from scratch into a local buffer, compare.
        fn verify(incr: &mut ThreatStack, board: &Board, weights: &[i8], nf: usize,
                  tag: &str, seed: u32, step: usize, gap_hist: &mut [u64; 32]) {
            // Record the replay gap this ensure_computed will take (per pov,
            // record the max) BEFORE it runs, for coverage reporting.
            for pov in [WHITE, BLACK] {
                if !incr.current().accurate[pov as usize] {
                    if let Some(anc) = incr.can_update(pov) {
                        let g = (incr.index() - anc).min(31);
                        gap_hist[g] += 1;
                    } else {
                        gap_hist[0] += 1; // full refresh bucket
                    }
                }
            }
            incr.ensure_computed(weights, nf, 0, board);
            let occ = board.colors[0] | board.colors[1];
            for pov in [WHITE, BLACK] {
                let ksq = (board.pieces[KING as usize] & board.colors[pov as usize])
                    .trailing_zeros();
                let mirrored = (ksq % 8) >= 4;
                let mut check = vec![0i16; H];
                crate::threats::enumerate_threats(
                    &board.pieces, &board.colors, &board.mailbox,
                    occ, pov, mirrored,
                    |idx| {
                        if idx < nf {
                            let w = idx * H;
                            for j in 0..H { check[j] += weights[w + j] as i16; }
                        }
                    },
                );
                let live = incr.values(pov);
                if &check[..] != live {
                    let mut first = None;
                    for j in 0..H {
                        if check[j] != live[j] { first = Some((j, live[j], check[j])); break; }
                    }
                    let (j, av, bv) = first.unwrap();
                    panic!(
                        "gap-fuzz divergence: {} seed={:#x} step={} pov={} fen=\"{}\" \
                         ch{} incr={} scratch={} (stack index {})",
                        tag, seed, step,
                        if pov == WHITE { "W" } else { "B" },
                        board.to_fen(), j, av, bv, incr.index(),
                    );
                }
            }
        }

        const STEPS: usize = 400;
        const WALKS_PER_FEN: usize = 8;
        const MAX_DEPTH: usize = 24;
        let mut gap_hist = [0u64; 32];

        for (fen_idx, fen) in START_FENS.iter().enumerate() {
            for walk in 0..WALKS_PER_FEN {
                let seed: u32 = 0xC0DA_C1u32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((walk as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };

                let mut board = Board::new();
                board.set_fen(fen);
                board.generate_threat_deltas = eager;

                let mut incr = ThreatStack::new(H);
                incr.active = true;
                incr.refresh(&weights, nf, &board, WHITE);
                incr.refresh(&weights, nf, &board, BLACK);

                // Track which made plies were null so we call the right unmake.
                let mut null_stack: Vec<bool> = Vec::new();
                let unmake_one = |board: &mut Board, incr: &mut ThreatStack,
                                      null_stack: &mut Vec<bool>| {
                    incr.pop();
                    if null_stack.pop().unwrap() {
                        board.unmake_null_move();
                    } else {
                        board.unmake_move();
                    }
                };
                for step in 0..STEPS {
                    let action = next_u32(&mut rng) % 100;
                    let depth = null_stack.len();
                    if action < 35 && depth > 0 {
                        // ~35%: pop + unmake
                        unmake_one(&mut board, &mut incr, &mut null_stack);
                    } else if action >= 90 && depth < MAX_DEPTH && !board.in_check()
                        && !null_stack.last().copied().unwrap_or(false)
                    {
                        // ~10%: null-move ply (search does this in NMP / RFP audit)
                        incr.push(NO_MOVE, NO_PIECE_TYPE);
                        board.make_null_move();
                        null_stack.push(true);
                    } else if depth < MAX_DEPTH {
                        // ~55%: push + make + absorb
                        let legal = generate_legal_moves(&board);
                        if legal.len == 0 {
                            if depth == 0 { break; }
                            unmake_one(&mut board, &mut incr, &mut null_stack);
                            continue;
                        }
                        let mv = legal.get((next_u32(&mut rng) as usize) % legal.len);
                        incr.push(NO_MOVE, NO_PIECE_TYPE);
                        let ok = board.make_move(mv);
                        assert!(ok, "gap-fuzz: legal move rejected");
                        incr.absorb_deltas(&board);
                        null_stack.push(false);
                    } else {
                        unmake_one(&mut board, &mut incr, &mut null_stack);
                    }
                    // Verify only on ~1/4 of steps so replay gaps build up.
                    if next_u32(&mut rng) % 4 == 0 {
                        verify(&mut incr, &board, &weights, nf,
                               &format!("fen{} walk{}", fen_idx, walk), seed, step,
                               &mut gap_hist);
                    }
                }
                // Final verification at wherever the walk ended.
                verify(&mut incr, &board, &weights, nf,
                       &format!("fen{} walk{} (final)", fen_idx, walk), seed, STEPS,
                       &mut gap_hist);
            }
        }
        // Coverage: prove gaps >= 2 were actually exercised.
        eprintln!("gap-fuzz replay-gap histogram (0 = full refresh): {:?}", &gap_hist[..12]);
        let multi: u64 = gap_hist[2..].iter().sum();
        assert!(multi > 100,
            "gap-fuzz did not exercise replay gaps >= 2 (histogram {:?})", &gap_hist[..12]);
    }

    /// Sanity: manual i16 value comparison across all 256 channels
    /// proves the weight pattern actually differs between features.
    #[test]
    fn weights_distinguish_features() {
        crate::init();
        let _space = crate::threats::FEATURE_SPACE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let nf = num_threat_features();
        let w = make_weights(nf.min(4));
        // Feature 0's row should differ from feature 1's row.
        let row0 = &w[0..H];
        let row1 = &w[H..2 * H];
        assert_ne!(row0, row1, "weight pattern collides between features");
    }

    /// Diagnostic: print the multiset symmetric diff between the
    /// incremental-applied feature indices and the refresh-enumerated
    /// indices for a single move. Not a pass/fail test — run with
    /// `cargo test dump_diff -- --nocapture` to inspect.
    #[test]
    #[ignore]
    fn dump_diff_b8c6_quiet_knight() {
        crate::init();
        let _space = crate::threats::FEATURE_SPACE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let nf = num_threat_features();
        let mut board = Board::new();
        // Position after 1.e4 e5 — white to move, then black plays Nc6 on next ply.
        // Set up mid-game FEN so it's black-to-move playing b8c6 immediately.
        board.set_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
        board.generate_threat_deltas = true;

        // Multiset diff, same structure as dump_diff_rook_capture_with_xray but parametric on pov.
        for pov in [WHITE, BLACK] {
            let mut pre: Vec<usize> = Vec::new();
            {
                let occ = board.colors[0] | board.colors[1];
                let k = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
                crate::threats::enumerate_threats(
                    &board.pieces, &board.colors, &board.mailbox,
                    occ, pov, (k % 8) >= 4,
                    |idx| if idx < nf { pre.push(idx) },
                );
            }
            eprintln!("[pov={}] pre count={}", if pov == WHITE { "W" } else { "B" }, pre.len());
        }

        let mv = parse_uci(&board, "b8c6");
        board.make_move(mv);

        for pov in [WHITE, BLACK] {
            let mut post: Vec<usize> = Vec::new();
            {
                let occ = board.colors[0] | board.colors[1];
                let k = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
                crate::threats::enumerate_threats(
                    &board.pieces, &board.colors, &board.mailbox,
                    occ, pov, (k % 8) >= 4,
                    |idx| if idx < nf { post.push(idx) },
                );
            }

            let k = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
            let mirrored = (k % 8) >= 4;

            let mut actual_adds: Vec<usize> = Vec::new();
            let mut actual_subs: Vec<usize> = Vec::new();
            for d in board.threat_deltas.iter() {
                let idx = crate::threats::threat_index(
                    d.attacker_cp() as usize, d.from_sq() as u32,
                    d.victim_cp() as usize, d.to_sq() as u32,
                    mirrored, pov,
                );
                if idx < 0 || (idx as usize) >= nf { continue; }
                if d.add() { actual_adds.push(idx as usize); }
                else { actual_subs.push(idx as usize); }
            }

            // Rebuild pre from the delta'd post to expose divergence.
            // expected_pre = post - adds + subs (treat as multiset)
            // For a matching path: expected_pre == pre.
            let mut reconstructed = post.clone();
            for a in &actual_adds {
                if let Some(p) = reconstructed.iter().position(|x| x == a) {
                    reconstructed.swap_remove(p);
                }
            }
            reconstructed.extend(actual_subs.iter());

            // Sort both for comparison.
            let mut pre: Vec<usize> = Vec::new();
            {
                // recompute pre via a pre-state snapshot via UNDO — but our board
                // already applied the move. Recompute using full pre-move FEN.
                let mut pre_board = Board::new();
                pre_board.set_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
                let occ = pre_board.colors[0] | pre_board.colors[1];
                let k0 = (pre_board.pieces[KING as usize] & pre_board.colors[pov as usize]).trailing_zeros();
                crate::threats::enumerate_threats(
                    &pre_board.pieces, &pre_board.colors, &pre_board.mailbox,
                    occ, pov, (k0 % 8) >= 4,
                    |idx| if idx < nf { pre.push(idx) },
                );
            }

            let mut pre_sorted = pre.clone(); pre_sorted.sort();
            let mut recon_sorted = reconstructed.clone(); recon_sorted.sort();

            let missing: Vec<usize> = pre_sorted.iter().filter(|x| !recon_sorted.contains(x)).cloned().collect();
            let extra: Vec<usize> = recon_sorted.iter().filter(|x| !pre_sorted.contains(x)).cloned().collect();
            eprintln!("[pov={}] pre={} post={} adds={} subs={} missing_from_recon={:?} extra_in_recon={:?}",
                if pov == WHITE { "W" } else { "B" }, pre.len(), post.len(), actual_adds.len(), actual_subs.len(), missing, extra);
            if !missing.is_empty() || !extra.is_empty() {
                // Enumerate post-state tuples that map to the extra index.
                let extra_set = extra.clone();
                eprintln!("  tuples in post that hit each 'extra' index:");
                let occ_post = board.colors[0] | board.colors[1];
                let mailbox_post = &board.mailbox;
                let white_bb = board.colors[0];
                let pieces_bb = &board.pieces;
                for extra_idx in &extra_set {
                    for color in [0u8, 1u8] {
                        for pt in 0..6u8 {
                            let mut bb = pieces_bb[pt as usize] & board.colors[color as usize];
                            while bb != 0 {
                                let sq = bb.trailing_zeros();
                                bb &= bb - 1;
                                let cp_a = crate::threats::colored_piece(color, pt);
                                let atts = crate::threats::piece_attacks_occ(pt, color, sq, occ_post) & occ_post;
                                let mut t = atts;
                                while t != 0 {
                                    let tsq = t.trailing_zeros();
                                    t &= t - 1;
                                    let vpt = mailbox_post[tsq as usize];
                                    if vpt >= 6 { continue; }
                                    let vcol = if white_bb & (1u64 << tsq) != 0 { 0 } else { 1 };
                                    let cp_v = crate::threats::colored_piece(vcol, vpt);
                                    let i = crate::threats::threat_index(
                                        cp_a, sq, cp_v, tsq as u32, mirrored, pov,
                                    );
                                    if i as usize == *extra_idx {
                                        eprintln!("    idx={} direct {}@{} -> {}@{}", i, cp_a, sq, cp_v, tsq);
                                    }
                                }
                                // x-ray
                                if pt == BISHOP || pt == ROOK || pt == QUEEN {
                                    let mut dt = atts;
                                    while dt != 0 {
                                        let bsq = dt.trailing_zeros();
                                        dt &= dt - 1;
                                        let ow = occ_post & !(1u64 << bsq);
                                        let through = crate::threats::piece_attacks_occ(pt, color, sq, ow);
                                        let revealed = through & !atts & ow;
                                        if revealed == 0 { continue; }
                                        let xsq = if sq < bsq {
                                            // Mirror of threats.rs blocker-bounds fix
                                            // (commit 97b805f): guard `1u64 << (bsq+1)`
                                            // against UB at bsq=63.
                                            let above_mask = if bsq + 1 < 64 {
                                                !((1u64 << (bsq + 1)) - 1)
                                            } else { 0 };
                                            let a = revealed & above_mask;
                                            if a != 0 { a.trailing_zeros() } else { 64 }
                                        } else {
                                            let b = revealed & ((1u64 << bsq) - 1);
                                            if b != 0 { 63 - b.leading_zeros() } else { 64 }
                                        };
                                        if xsq < 64 {
                                            let xpt = mailbox_post[xsq as usize];
                                            if xpt < 6 {
                                                let xcol = if white_bb & (1u64 << xsq) != 0 { 0 } else { 1 };
                                                let cp_x = crate::threats::colored_piece(xcol, xpt);
                                                let i = crate::threats::threat_index(
                                                    cp_a, sq, cp_x, xsq, mirrored, pov,
                                                );
                                                if i as usize == *extra_idx {
                                                    eprintln!("    idx={} xray  {}@{} -> {}@{}", i, cp_a, sq, cp_x, xsq);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                eprintln!("  raw deltas (idx = threat_index for this pov):");
                for d in board.threat_deltas.iter() {
                    let idx = crate::threats::threat_index(
                        d.attacker_cp() as usize, d.from_sq() as u32,
                        d.victim_cp() as usize, d.to_sq() as u32,
                        mirrored, pov,
                    );
                    eprintln!("    a_cp={} from={} v_cp={} to={} add={} idx={}",
                        d.attacker_cp(), d.from_sq(), d.victim_cp(), d.to_sq(), d.add(), idx);
                }
                // Dump pre and post multisets too (sorted).
                let mut pre_sorted2 = pre.clone(); pre_sorted2.sort();
                let mut post_sorted = post.clone(); post_sorted.sort();
                eprintln!("  pre : {:?}", pre_sorted2);
                eprintln!("  post: {:?}", post_sorted);
            }
        }
    }

    #[test]
    #[ignore]
    fn dump_diff_rook_capture_with_xray() {
        crate::init();
        let _space = crate::threats::FEATURE_SPACE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let nf = num_threat_features();
        let mut board = Board::new();
        board.set_fen("k7/n7/8/8/p7/8/8/R3K3 w Q - 0 1");
        board.generate_threat_deltas = true;

        // Baseline multiset: enumerate_threats at position BEFORE move.
        let mut pre_indices_w: Vec<usize> = Vec::new();
        {
            let occ = board.colors[0] | board.colors[1];
            let wk = (board.pieces[KING as usize] & board.colors[0]).trailing_zeros();
            crate::threats::enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, WHITE, (wk % 8) >= 4,
                |idx| if idx < nf { pre_indices_w.push(idx) },
            );
        }

        let mv = parse_uci(&board, "a1a4");
        board.make_move(mv);

        // Refresh multiset at post-move position.
        let mut post_indices_w: Vec<usize> = Vec::new();
        {
            let occ = board.colors[0] | board.colors[1];
            let wk = (board.pieces[KING as usize] & board.colors[0]).trailing_zeros();
            crate::threats::enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, WHITE, (wk % 8) >= 4,
                |idx| if idx < nf { post_indices_w.push(idx) },
            );
        }

        // Expected delta multiset: (post - pre)_add, (pre - post)_sub.
        let mut expected_adds = post_indices_w.clone();
        let mut expected_subs = pre_indices_w.clone();
        // Remove intersection to get the symmetric difference (net change).
        for i in (0..expected_adds.len()).rev() {
            if let Some(p) = expected_subs.iter().position(|&x| x == expected_adds[i]) {
                expected_subs.swap_remove(p);
                expected_adds.swap_remove(i);
            }
        }
        expected_adds.sort();
        expected_subs.sort();

        // Incremental multiset: walk board.threat_deltas through threat_index.
        let wk = (board.pieces[KING as usize] & board.colors[0]).trailing_zeros();
        let mirrored = (wk % 8) >= 4;
        let mut actual_adds: Vec<usize> = Vec::new();
        let mut actual_subs: Vec<usize> = Vec::new();
        for d in board.threat_deltas.iter() {
            let idx = crate::threats::threat_index(
                d.attacker_cp() as usize, d.from_sq() as u32,
                d.victim_cp() as usize, d.to_sq() as u32,
                mirrored, WHITE,
            );
            if idx < 0 || (idx as usize) >= nf { continue; }
            if d.add() { actual_adds.push(idx as usize); }
            else { actual_subs.push(idx as usize); }
        }
        // Net out the actual delta too (incremental deltas can include
        // add+sub pairs for the same feature that cancel out).
        for i in (0..actual_adds.len()).rev() {
            if let Some(p) = actual_subs.iter().position(|&x| x == actual_adds[i]) {
                actual_subs.swap_remove(p);
                actual_adds.swap_remove(i);
            }
        }
        actual_adds.sort();
        actual_subs.sort();

        eprintln!("=== Rxa4 (a1→a4 capture, knight on a7 behind) WHITE pov ===");
        eprintln!("expected_adds (post - pre): {:?}", expected_adds);
        eprintln!("expected_subs (pre - post): {:?}", expected_subs);
        eprintln!("actual_adds   (from deltas): {:?}", actual_adds);
        eprintln!("actual_subs   (from deltas): {:?}", actual_subs);

        let missing_adds: Vec<_> = expected_adds.iter().filter(|i| !actual_adds.contains(i)).collect();
        let extra_adds: Vec<_> = actual_adds.iter().filter(|i| !expected_adds.contains(i)).collect();
        let missing_subs: Vec<_> = expected_subs.iter().filter(|i| !actual_subs.contains(i)).collect();
        let extra_subs: Vec<_> = actual_subs.iter().filter(|i| !expected_subs.contains(i)).collect();
        eprintln!("missing adds (expected but not emitted): {:?}", missing_adds);
        eprintln!("extra   adds (emitted but not expected): {:?}", extra_adds);
        eprintln!("missing subs (expected but not emitted): {:?}", missing_subs);
        eprintln!("extra   subs (emitted but not expected): {:?}", extra_subs);

        // Print raw deltas too.
        eprintln!("raw deltas:");
        for d in board.threat_deltas.iter() {
            eprintln!("  attacker_cp={} from={} victim_cp={} to={} add={}",
                d.attacker_cp(), d.from_sq(), d.victim_cp(), d.to_sq(), d.add());
        }
    }

    /// Reproduce the fuzz_random_games failure seed 0xdebd0132 at ply 28 (e1c1).
    /// Prints the pre-move FEN, raw deltas, and per-channel threat_index diff
    /// for the BLACK pov where the failure is observed. Ignored by default —
    /// run with `cargo test --release diag_fuzz_ply28_e1c1 -- --nocapture --ignored`.
    #[test]
    #[ignore]
    fn diag_fuzz_ply28_e1c1() {
        crate::init();
        let _space = crate::threats::FEATURE_SPACE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let nf = num_threat_features();

        // Kiwipete + same xorshift32 PRNG as fuzz_random_games (fen_idx=1, game=0).
        const KIWIPETE: &str =
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let seed: u32 = 0xdebd0132;
        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            x
        }

        let mut rng = seed;
        let mut board = Board::new();
        board.set_fen(KIWIPETE);
        board.generate_threat_deltas = true;

        for ply in 0..28 {
            let legal = generate_legal_moves(&board);
            assert!(legal.len > 0, "no legal at ply {}", ply);
            let idx = (next_u32(&mut rng) as usize) % legal.len;
            let mv = legal.get(idx);
            assert!(board.make_move(mv), "move illegal at ply {}", ply);
        }

        // Ply 28: pick the move, expect e1c1 (white castling queenside).
        let legal = generate_legal_moves(&board);
        let idx = (next_u32(&mut rng) as usize) % legal.len;
        let mv = legal.get(idx);
        let uci = crate::types::move_to_uci(mv);
        let pre_fen = board.to_fen();
        eprintln!("=== ply 28 pre-FEN: {}", pre_fen);
        eprintln!("=== ply 28 move: {}", uci);
        assert_eq!(uci, "e1c1", "expected e1c1, got {}", uci);

        // Pre-move enumeration for BLACK pov.
        let pov = BLACK;
        let mirrored_pre = {
            let occ = board.colors[0] | board.colors[1];
            let _ = occ;
            let k = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
            (k % 8) >= 4
        };
        let mut pre_indices: Vec<usize> = Vec::new();
        {
            let occ = board.colors[0] | board.colors[1];
            crate::threats::enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, pov, mirrored_pre,
                |i| if i < nf { pre_indices.push(i) },
            );
        }

        // Make the move.
        assert!(board.make_move(mv));

        // Post-move enumeration for BLACK.
        let mirrored_post = {
            let k = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
            (k % 8) >= 4
        };
        let mut post_indices: Vec<usize> = Vec::new();
        {
            let occ = board.colors[0] | board.colors[1];
            crate::threats::enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, pov, mirrored_post,
                |i| if i < nf { post_indices.push(i) },
            );
        }

        // If mirror flipped between pre and post, per-feature index
        // comparison isn't meaningful. Skip directly in that case.
        eprintln!("mirrored_pre={} mirrored_post={}", mirrored_pre, mirrored_post);

        // Expected net change: (post - pre)_add, (pre - post)_sub.
        let mut expected_adds = post_indices.clone();
        let mut expected_subs = pre_indices.clone();
        for i in (0..expected_adds.len()).rev() {
            if let Some(p) = expected_subs.iter().position(|&x| x == expected_adds[i]) {
                expected_subs.swap_remove(p);
                expected_adds.swap_remove(i);
            }
        }
        expected_adds.sort(); expected_subs.sort();

        // Actual net: walk raw deltas through threat_index with the POST-move mirror.
        let mut actual_adds: Vec<usize> = Vec::new();
        let mut actual_subs: Vec<usize> = Vec::new();
        for d in board.threat_deltas.iter() {
            let idx = crate::threats::threat_index(
                d.attacker_cp() as usize, d.from_sq() as u32,
                d.victim_cp() as usize, d.to_sq() as u32,
                mirrored_post, pov,
            );
            if idx < 0 || (idx as usize) >= nf { continue; }
            if d.add() { actual_adds.push(idx as usize); }
            else { actual_subs.push(idx as usize); }
        }
        // Net out same-feature add+sub cancellations.
        for i in (0..actual_adds.len()).rev() {
            if let Some(p) = actual_subs.iter().position(|&x| x == actual_adds[i]) {
                actual_subs.swap_remove(p);
                actual_adds.swap_remove(i);
            }
        }
        actual_adds.sort(); actual_subs.sort();

        let missing_adds: Vec<_> = expected_adds.iter().filter(|i| !actual_adds.contains(i)).cloned().collect();
        let extra_adds:   Vec<_> = actual_adds.iter().filter(|i| !expected_adds.contains(i)).cloned().collect();
        let missing_subs: Vec<_> = expected_subs.iter().filter(|i| !actual_subs.contains(i)).cloned().collect();
        let extra_subs:   Vec<_> = actual_subs.iter().filter(|i| !expected_subs.contains(i)).cloned().collect();
        eprintln!("pre count={} post count={}", pre_indices.len(), post_indices.len());
        eprintln!("expected_adds: {:?}", expected_adds);
        eprintln!("expected_subs: {:?}", expected_subs);
        eprintln!("actual_adds:   {:?}", actual_adds);
        eprintln!("actual_subs:   {:?}", actual_subs);
        eprintln!("missing adds (expected but not emitted): {:?}", missing_adds);
        eprintln!("extra   adds (emitted but not expected): {:?}", extra_adds);
        eprintln!("missing subs (expected but not emitted): {:?}", missing_subs);
        eprintln!("extra   subs (emitted but not expected): {:?}", extra_subs);

        eprintln!("raw deltas (ply28 e1c1):");
        for d in board.threat_deltas.iter() {
            let idx = crate::threats::threat_index(
                d.attacker_cp() as usize, d.from_sq() as u32,
                d.victim_cp() as usize, d.to_sq() as u32,
                mirrored_post, pov,
            );
            eprintln!("  a_cp={} from={} v_cp={} to={} add={}  idx(B)={}",
                d.attacker_cp(), d.from_sq(), d.victim_cp(), d.to_sq(), d.add(), idx);
        }
    }

    /// Ensure RawThreatDelta round-trips — if this breaks the whole
    /// Sparsity measurement: feature activation frequency across many
    /// self-play positions. Ignored by default — run with
    /// `cargo test --release measure_feature_sparsity -- --nocapture --ignored`.
    ///
    /// Purpose: inform the "drop cold features" optimization target. If
    /// the bottom X% of features fire <0.001% of positions, dropping
    /// them shrinks the 50 MB weight matrix proportionally with near-
    /// zero Elo cost, improving cache residency on memory-constrained
    /// hardware.
    #[test]
    #[ignore]
    fn measure_feature_sparsity() {
        crate::init();
        let _space = crate::threats::FEATURE_SPACE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let nf = num_threat_features();
        eprintln!("Measuring activation frequency across {} threat features", nf);

        // Deterministic self-play positions from 5 starting FENs * 30 games * 80 plies
        // ≈ 12000 positions. Wider than fuzz_random_games, focused on distribution.
        const START_FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1",
        ];
        const GAMES_PER_FEN: usize = 30;
        const MAX_PLIES: usize = 80;

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            x
        }

        // Histograms: total activations per feature + positions observed
        let mut feature_hits: Vec<u32> = vec![0u32; nf];
        let mut positions = 0u64;
        let mut total_activations = 0u64;

        for (fi, fen) in START_FENS.iter().enumerate() {
            for g in 0..GAMES_PER_FEN {
                let seed: u32 = 0x12345u32
                    .wrapping_add((fi as u32).wrapping_mul(1_000_003))
                    .wrapping_add((g as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };

                let mut board = Board::new();
                board.set_fen(fen);

                for _ply in 0..MAX_PLIES {
                    // Record feature activations from both POVs at this position.
                    for pov in [WHITE, BLACK] {
                        let occ = board.colors[0] | board.colors[1];
                        let k = (board.pieces[KING as usize] & board.colors[pov as usize])
                            .trailing_zeros();
                        let mirrored = (k % 8) >= 4;
                        crate::threats::enumerate_threats(
                            &board.pieces, &board.colors, &board.mailbox,
                            occ, pov, mirrored,
                            |idx| {
                                if idx < nf {
                                    feature_hits[idx] = feature_hits[idx].saturating_add(1);
                                    total_activations += 1;
                                }
                            },
                        );
                    }
                    positions += 1;

                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    let mv_idx = (next_u32(&mut rng) as usize) % legal.len;
                    let mv = legal.get(mv_idx);
                    if !board.make_move(mv) { break; }
                }
            }
        }

        // Distribution buckets
        let mut bucket_0     = 0u64; // never activated
        let mut bucket_1_9   = 0u64;
        let mut bucket_10_99 = 0u64;
        let mut bucket_100_999 = 0u64;
        let mut bucket_1k_plus = 0u64;
        let mut max_hits = 0u32;
        let mut max_idx  = 0usize;
        for (i, &h) in feature_hits.iter().enumerate() {
            if h == 0 { bucket_0 += 1; }
            else if h < 10 { bucket_1_9 += 1; }
            else if h < 100 { bucket_10_99 += 1; }
            else if h < 1000 { bucket_100_999 += 1; }
            else { bucket_1k_plus += 1; }
            if h > max_hits { max_hits = h; max_idx = i; }
        }

        // Top-K features
        let mut indexed: Vec<(usize, u32)> = feature_hits.iter().enumerate()
            .map(|(i, &h)| (i, h)).collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));

        // Coverage: cumulative % of activations captured by top-K features
        let mut cumulative = 0u64;
        let mut features_for_99 = 0usize;
        let mut features_for_90 = 0usize;
        let mut features_for_50 = 0usize;
        for (i, (_idx, h)) in indexed.iter().enumerate() {
            cumulative += *h as u64;
            let pct = cumulative as f64 / total_activations as f64 * 100.0;
            if pct >= 50.0 && features_for_50 == 0 { features_for_50 = i + 1; }
            if pct >= 90.0 && features_for_90 == 0 { features_for_90 = i + 1; }
            if pct >= 99.0 && features_for_99 == 0 { features_for_99 = i + 1; break; }
        }

        eprintln!("\n=== Threat feature sparsity measurement ===");
        eprintln!("Total positions sampled:   {}", positions);
        eprintln!("Total activations recorded: {}", total_activations);
        eprintln!("Avg features active per pov-position: {:.1}", total_activations as f64 / (positions * 2) as f64);
        eprintln!("\n--- Feature activation distribution ---");
        eprintln!("  0 hits    (never fired):    {:>7} features ({:.1}%)", bucket_0, bucket_0 as f64 / nf as f64 * 100.0);
        eprintln!("  1-9 hits  (very rare):      {:>7} features ({:.1}%)", bucket_1_9, bucket_1_9 as f64 / nf as f64 * 100.0);
        eprintln!("  10-99     (uncommon):       {:>7} features ({:.1}%)", bucket_10_99, bucket_10_99 as f64 / nf as f64 * 100.0);
        eprintln!("  100-999   (common):         {:>7} features ({:.1}%)", bucket_100_999, bucket_100_999 as f64 / nf as f64 * 100.0);
        eprintln!("  1000+     (hot):            {:>7} features ({:.1}%)", bucket_1k_plus, bucket_1k_plus as f64 / nf as f64 * 100.0);
        eprintln!("\n--- Coverage ---");
        eprintln!("  Top {} features capture 50% of activations", features_for_50);
        eprintln!("  Top {} features capture 90% of activations", features_for_90);
        eprintln!("  Top {} features capture 99% of activations", features_for_99);
        eprintln!("  Max activations on single feature: {} (idx={})", max_hits, max_idx);

        // Memory implications
        let row_bytes = 768;  // v9 accumulator size, i8 weights
        let total_bytes = (nf as u64) * row_bytes;
        let dead_bytes  = bucket_0 * row_bytes;
        let cold_bytes  = (bucket_0 + bucket_1_9) * row_bytes;
        eprintln!("\n--- Memory impact (768-byte rows, i8 weights) ---");
        eprintln!("  Total weight matrix: {:.1} MB", total_bytes as f64 / 1_048_576.0);
        eprintln!("  Dropping dead features (0 hits): save {:.1} MB ({:.1}%)", dead_bytes as f64 / 1_048_576.0, bucket_0 as f64 / nf as f64 * 100.0);
        eprintln!("  Dropping dead+rare (<10 hits):   save {:.1} MB ({:.1}%)", cold_bytes as f64 / 1_048_576.0, (bucket_0 + bucket_1_9) as f64 / nf as f64 * 100.0);
    }

    /// incremental path will silently misapply deltas.
    #[test]
    fn raw_delta_roundtrip() {
        let d = RawThreatDelta::new(5, 28, 11, 63, true);
        assert_eq!(d.attacker_cp(), 5);
        assert_eq!(d.from_sq(), 28);
        assert_eq!(d.victim_cp(), 11);
        assert_eq!(d.to_sq(), 63);
        assert!(d.add());
        let d2 = RawThreatDelta::new(0, 0, 0, 0, false);
        assert!(!d2.add());
    }
}

/// Threat accumulator stack (Reckless pattern).
///
/// Separate from the PSQ accumulator. Each ply has:
/// - Per-perspective i16 accumulator values (aligned)
/// - Per-perspective accuracy flags
/// - Threat deltas (ArrayVec, no heap allocation)
/// - Move info for king mirror detection
///
/// The stack is pre-allocated for MAX_PLY entries. Push/pop is trivial.
/// BoardObserver callbacks during make_move push deltas directly.
/// Evaluate walks back to find an accurate ancestor and replays forward.

use crate::threats::{RawThreatDelta, MAX_THREAT_DELTAS};
use crate::types::*;

const MAX_PLY: usize = 256;

/// Fixed-capacity array (no heap, like ArrayVec but simpler).
/// Tracks overflow so callers can force full recompute instead of
/// silently using incomplete deltas.
#[derive(Clone)]
pub struct DeltaVec {
    data: [RawThreatDelta; MAX_THREAT_DELTAS],
    len: usize,
    overflowed: bool,
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
    pub values: [[i16; 768]; 2], // max accumulator size for v9
    /// Per-perspective accuracy flags
    pub accurate: [bool; 2],
    /// Threat deltas for the move that produced this ply
    pub delta: DeltaVec,
    /// The move that produced this ply (for king mirror check)
    pub mv: Move,
    /// Piece type that moved (for king mirror detection)
    pub moved_pt: u8,
    /// Color that moved (for per-perspective king mirror check)
    pub moved_color: u8,
}

impl ThreatEntry {
    pub const fn new() -> Self {
        Self {
            values: [[0i16; 768]; 2],
            accurate: [false; 2],
            delta: DeltaVec::new(),
            mv: NO_MOVE,
            moved_pt: NO_PIECE_TYPE,
            moved_color: WHITE,
        }
    }
}

/// The threat accumulator stack.
pub struct ThreatStack {
    stack: Vec<ThreatEntry>,
    index: usize,
    hidden_size: usize,
    /// Whether threat features are active (net has threats)
    pub active: bool,
}

impl ThreatStack {
    pub fn new(hidden_size: usize) -> Self {
        let mut stack = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            stack.push(ThreatEntry::new());
        }
        Self { stack, index: 0, hidden_size, active: false }
    }

    #[inline]
    pub fn index(&self) -> usize { self.index }

    #[inline]
    pub fn current(&self) -> &ThreatEntry { &self.stack[self.index] }

    #[inline]
    pub fn current_mut(&mut self) -> &mut ThreatEntry { &mut self.stack[self.index] }

    /// Push: increment index, reset flags, clear deltas.
    /// Called BEFORE make_move (mirrors Reckless's Network::push).
    pub fn push(&mut self, mv: Move, moved_pt: u8) {
        self.index += 1;
        if self.index >= self.stack.len() {
            self.stack.push(ThreatEntry::new());
        }
        let entry = &mut self.stack[self.index];
        entry.accurate = [false; 2];
        entry.delta.clear();
        entry.mv = mv;
        entry.moved_pt = moved_pt;
    }

    /// Pop: decrement index.
    pub fn pop(&mut self) {
        debug_assert!(self.index > 0);
        self.index -= 1;
    }

    /// Reset: for new positions (between bench positions, new game).
    pub fn reset(&mut self) {
        self.index = 0;
        self.stack[0].accurate = [false; 2];
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

        // Collect feature indices, then apply with SIMD
        let mut indices = [0usize; 256]; // max active threat features per position
        let mut n_indices = 0usize;

        crate::threats::enumerate_threats(
            &board.pieces, &board.colors, &board.mailbox,
            occ, pov, mirrored,
            |feat_idx| {
                if feat_idx < num_features && n_indices < indices.len() {
                    indices[n_indices] = feat_idx;
                    n_indices += 1;
                }
            },
        );

        // Apply all weight rows with SIMD
        crate::threats::add_weight_rows(
            &mut entry.values[p][..h], net_weights, h, &indices[..n_indices],
        );

        entry.accurate[p] = true;
    }

    /// Check if we can incrementally update this perspective by walking back.
    /// Returns Some(ancestor_index) or None (need full refresh).
    /// Matches Reckless's can_update_threats.
    #[inline]
    pub fn can_update(&self, pov: Color) -> Option<usize> {
        for i in (0..self.index).rev() {
            if self.stack[i].accurate[pov as usize] {
                return Some(i);
            }

            // Check the entry at i+1 for king crossing e-file
            let entry = &self.stack[i + 1];
            if entry.mv == NO_MOVE {
                continue; // null move, safe
            }
            if entry.delta.overflowed() {
                return None; // overflow means incomplete deltas, can't replay
            }
            if entry.moved_pt == KING && entry.moved_color == pov {
                let from = move_from(entry.mv);
                let to = move_to(entry.mv);
                if (from % 8 >= 4) != (to % 8 >= 4) {
                    // This perspective's king crossed e-file — mirroring changed
                    return None;
                }
            }
        }
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

        for ply in (ancestor + 1)..=self.index {
            let entry_mv = self.stack[ply].mv;

            if entry_mv == NO_MOVE || self.stack[ply].delta.is_empty() {
                // Null move or no deltas: copy from previous
                let (prev, curr) = self.stack.split_at_mut(ply);
                curr[0].values[p][..h].copy_from_slice(&prev[ply - 1].values[p][..h]);
            } else {
                // Copy deltas to local buffer to avoid borrow conflict with split_at_mut
                let n_deltas = self.stack[ply].delta.len;
                let mut local_deltas = [crate::threats::RawThreatDelta::ZERO; 128];
                local_deltas[..n_deltas].copy_from_slice(&self.stack[ply].delta.data[..n_deltas]);
                // Use SIMD apply_threat_deltas (copies src + applies adds/subs)
                let (prev, curr) = self.stack.split_at_mut(ply);
                crate::threats::apply_threat_deltas(
                    &mut curr[0].values[p][..h],
                    &prev[ply - 1].values[p][..h],
                    &local_deltas[..n_deltas],
                    net_weights, h, num_features,
                    pov, mirrored,
                );
            }

            self.stack[ply].accurate[p] = true;
        }
    }

    /// Get the accumulator values for a perspective.
    #[inline]
    pub fn values(&self, pov: Color) -> &[i16] {
        &self.stack[self.index].values[pov as usize][..self.hidden_size]
    }

    /// Ensure both perspectives are computed for the current position.
    /// Matches Reckless's evaluate() pattern.
    #[inline]
    pub fn ensure_computed(&mut self, net_weights: &[i8], num_features: usize,
                          board: &crate::board::Board) {
        if !self.active { return; }

        let idx = self.index;
        for pov in [WHITE, BLACK] {
            if self.stack[idx].accurate[pov as usize] {
                continue;
            }

            match self.can_update(pov) {
                Some(ancestor) => self.update(ancestor, net_weights, num_features, board, pov),
                None => self.refresh(net_weights, num_features, board, pov),
            }
        }
    }
}

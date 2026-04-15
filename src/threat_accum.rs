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
            data: [RawThreatDelta { attacker_cp: 0, from_sq: 0, victim_cp: 0, to_sq: 0, add: false }; MAX_THREAT_DELTAS],
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
}

impl ThreatEntry {
    pub const fn new() -> Self {
        Self {
            values: [[0i16; 768]; 2],
            accurate: [false; 2],
            delta: DeltaVec::new(),
            mv: NO_MOVE,
            moved_pt: NO_PIECE_TYPE,
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
    pub fn refresh(&mut self, net_weights: &[i8], num_features: usize,
                   board: &crate::board::Board, pov: Color) {
        let h = self.hidden_size;
        let entry = &mut self.stack[self.index];
        let p = pov as usize;
        entry.values[p][..h].fill(0);

        let occ = board.colors[0] | board.colors[1];
        let king_sq = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
        let mirrored = (king_sq % 8) >= 4;

        crate::threats::enumerate_threats(
            &board.pieces, &board.colors, &board.mailbox,
            occ, pov, mirrored,
            |feat_idx| {
                if feat_idx < num_features {
                    let w_off = feat_idx * h;
                    for j in 0..h {
                        entry.values[p][j] += net_weights[w_off + j] as i16;
                    }
                }
            },
        );

        entry.accurate[p] = true;
    }

    /// Check if we can incrementally update this perspective by walking back.
    /// Returns Some(ancestor_index) or None (need full refresh).
    /// Matches Reckless's can_update_threats.
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
            if entry.delta.is_empty() || entry.delta.overflowed() {
                return None; // real move without deltas or overflow, can't pass
            }
            if entry.moved_pt == KING {
                let from = move_from(entry.mv);
                let to = move_to(entry.mv);
                if (from % 8 >= 4) != (to % 8 >= 4) {
                    // King crossed e-file — mirroring changed for this perspective
                    // Only invalidates if it's THIS perspective's king
                    // For simplicity, invalidate either perspective's king crossing
                    return None;
                }
            }
        }
        None
    }

    /// Incremental update: replay from ancestor to current index for one perspective.
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
                // Apply deltas
                let (prev, curr) = self.stack.split_at_mut(ply);
                curr[0].values[p][..h].copy_from_slice(&prev[ply - 1].values[p][..h]);

                for delta in curr[0].delta.data[..curr[0].delta.len].iter() {
                    let idx = crate::threats::threat_index(
                        delta.attacker_cp as usize, delta.from_sq as u32,
                        delta.victim_cp as usize, delta.to_sq as u32,
                        mirrored, pov,
                    );
                    if idx < 0 || (idx as usize) >= num_features { continue; }
                    let w_off = idx as usize * h;
                    if delta.add {
                        for j in 0..h {
                            curr[0].values[p][j] += net_weights[w_off + j] as i16;
                        }
                    } else {
                        for j in 0..h {
                            curr[0].values[p][j] -= net_weights[w_off + j] as i16;
                        }
                    }
                }
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
    pub fn ensure_computed(&mut self, net_weights: &[i8], num_features: usize,
                          board: &crate::board::Board) {
        if !self.active { return; }

        for pov in [WHITE, BLACK] {
            if self.stack[self.index].accurate[pov as usize] {
                continue;
            }

            match self.can_update(pov) {
                Some(ancestor) => self.update(ancestor, net_weights, num_features, board, pov),
                None => self.refresh(net_weights, num_features, board, pov),
            }
        }
    }
}

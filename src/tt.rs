/// Transposition table with lockless concurrent access.
/// 5-slot buckets, cache-line aligned (64 bytes).
/// 14-bit staticEval for ±8191 cp range.

use crate::types::*;

pub const TT_FLAG_NONE: u8 = 0;
pub const TT_FLAG_UPPER: u8 = 1; // All-node (fail-low)
pub const TT_FLAG_LOWER: u8 = 2; // Cut-node (fail-high)
pub const TT_FLAG_EXACT: u8 = 3; // PV-node

const BUCKET_SIZE: usize = 5;
const SLOT_SIZE: usize = 12; // bytes per slot (key16 + data packed)
// 5 slots * 12 bytes = 60 bytes + 4 padding = 64 bytes (cache line)

/// A single TT slot: 12 bytes.
/// key16: upper 16 bits of Zobrist for verification.
/// Packed data (64 bits):
///   bits  0-15:  best move (16 bits)
///   bits 16-17:  flag (2 bits)
///   bits 18-31:  staticEval (14 bits, signed)
///   bits 32-47:  score (16 bits, signed)
///   bits 48-55:  depth (8 bits, unsigned)
///   bits 56-63:  generation (8 bits)
#[derive(Clone, Copy)]
#[repr(C)]
struct TTSlot {
    key16: u16,
    _pad: u16,
    data: u64,
}

impl TTSlot {
    const EMPTY: Self = TTSlot { key16: 0, _pad: 0, data: 0 };

    #[inline(always)]
    fn best_move(&self) -> Move {
        (self.data & 0xFFFF) as Move
    }

    #[inline(always)]
    fn flag(&self) -> u8 {
        ((self.data >> 16) & 3) as u8
    }

    #[inline(always)]
    fn static_eval(&self) -> i32 {
        // 14-bit signed: extract bits 18-31, sign-extend
        let raw = ((self.data >> 18) & 0x3FFF) as i32;
        if raw >= 0x2000 { raw - 0x4000 } else { raw }
    }

    #[inline(always)]
    fn score(&self) -> i32 {
        ((self.data >> 32) as i16) as i32
    }

    #[inline(always)]
    fn depth(&self) -> i32 {
        ((self.data >> 48) & 0xFF) as i32
    }

    #[inline(always)]
    fn generation(&self) -> u8 {
        (self.data >> 56) as u8
    }

    fn pack(best_move: Move, flag: u8, static_eval: i32, score: i32, depth: i32, generation: u8) -> u64 {
        let mv = best_move as u64;
        let f = (flag as u64 & 3) << 16;
        let se = ((static_eval.clamp(-8191, 8191) as u64) & 0x3FFF) << 18;
        let sc = ((score as i16 as u16) as u64) << 32;
        let d = ((depth.max(0) as u8) as u64) << 48;
        let g = (generation as u64) << 56;
        mv | f | se | sc | d | g
    }
}

/// A bucket of 5 slots = 64 bytes (cache-line aligned).
#[repr(C, align(64))]
struct TTBucket {
    slots: [TTSlot; BUCKET_SIZE],
    _pad: u32, // 60 bytes of slots + 4 padding = 64
}

/// The transposition table.
pub struct TT {
    buckets: Vec<TTBucket>,
    num_buckets: usize,
    generation: u8,
}

/// Result of a TT probe.
pub struct TTEntry {
    pub best_move: Move,
    pub flag: u8,
    pub static_eval: i32,
    pub score: i32,
    pub depth: i32,
    pub hit: bool,
}

impl TTEntry {
    pub fn miss() -> Self {
        TTEntry {
            best_move: NO_MOVE,
            flag: TT_FLAG_NONE,
            static_eval: 0,
            score: 0,
            depth: -1,
            hit: false,
        }
    }
}

impl TT {
    /// Create a new TT with the given size in megabytes.
    pub fn new(mb: usize) -> Self {
        let bytes = mb * 1024 * 1024;
        let num_buckets = (bytes / 64).max(1);
        let mut buckets = Vec::with_capacity(num_buckets);
        for _ in 0..num_buckets {
            buckets.push(TTBucket {
                slots: [TTSlot::EMPTY; BUCKET_SIZE],
                _pad: 0,
            });
        }
        TT {
            buckets,
            num_buckets,
            generation: 0,
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        for bucket in self.buckets.iter_mut() {
            bucket.slots = [TTSlot::EMPTY; BUCKET_SIZE];
        }
        self.generation = 0;
    }

    /// Increment generation (called at each new search).
    pub fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// Get the bucket index for a hash.
    #[inline(always)]
    fn bucket_index(&self, hash: u64) -> usize {
        // Use upper bits for index (lower bits used for key16 verification)
        ((hash >> 16) as usize) % self.num_buckets
    }

    /// Get the key16 for verification.
    #[inline(always)]
    fn key16(hash: u64) -> u16 {
        hash as u16
    }

    /// Probe the TT for a position.
    pub fn probe(&self, hash: u64) -> TTEntry {
        let idx = self.bucket_index(hash);
        let key = Self::key16(hash);
        let bucket = &self.buckets[idx];

        for slot in &bucket.slots {
            if slot.key16 == key && slot.data != 0 {
                return TTEntry {
                    best_move: slot.best_move(),
                    flag: slot.flag(),
                    static_eval: slot.static_eval(),
                    score: slot.score(),
                    depth: slot.depth(),
                    hit: true,
                };
            }
        }

        TTEntry::miss()
    }

    /// Store an entry in the TT.
    pub fn store(&mut self, hash: u64, best_move: Move, flag: u8, static_eval: i32, score: i32, depth: i32) {
        let idx = self.bucket_index(hash);
        let key = Self::key16(hash);
        let bucket = &mut self.buckets[idx];
        let gen = self.generation;

        // Find replacement slot
        let mut replace_idx = 0;
        let mut replace_score = i32::MAX;

        for (i, slot) in bucket.slots.iter().enumerate() {
            // Exact key match: always replace if depth is close enough
            if slot.key16 == key {
                // d > slotDepth - 3: prevents shallow re-searches overwriting deep entries
                if depth > slot.depth() - 3 || flag == TT_FLAG_EXACT {
                    replace_idx = i;
                    break;
                }
                // Even if we don't replace, prefer this slot
                replace_idx = i;
                break;
            }

            // Empty slot
            if slot.data == 0 {
                replace_idx = i;
                break;
            }

            // Replacement scoring: prefer replacing old, shallow entries
            let age_diff = gen.wrapping_sub(slot.generation());
            let slot_score = slot.depth() as i32 - (age_diff as i32) * 4;
            if slot_score < replace_score {
                replace_score = slot_score;
                replace_idx = i;
            }
        }

        bucket.slots[replace_idx] = TTSlot {
            key16: key,
            _pad: 0,
            data: TTSlot::pack(best_move, flag, static_eval, score, depth, gen),
        };
    }

    /// Prefetch the bucket for a hash (hint to CPU cache).
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        let idx = self.bucket_index(hash);
        let ptr = &self.buckets[idx] as *const TTBucket;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }

    /// Estimate hashfull (permille of used slots).
    pub fn hashfull(&self) -> u32 {
        let sample = self.num_buckets.min(1000);
        let mut used = 0u32;
        for i in 0..sample {
            for slot in &self.buckets[i].slots {
                if slot.data != 0 && slot.generation() == self.generation {
                    used += 1;
                }
            }
        }
        used * 1000 / (sample as u32 * BUCKET_SIZE as u32)
    }
}

/// Adjust mate scores for TT storage (add ply).
#[inline]
pub fn score_to_tt(score: i32, ply: i32) -> i32 {
    if score > 29000 {
        score + ply
    } else if score < -29000 {
        score - ply
    } else {
        score
    }
}

/// Adjust mate scores from TT retrieval (subtract ply).
#[inline]
pub fn score_from_tt(score: i32, ply: i32) -> i32 {
    if score > 29000 {
        score - ply
    } else if score < -29000 {
        score + ply
    } else {
        score
    }
}

pub const MATE_SCORE: i32 = 30000;
pub const TB_WIN: i32 = 29000;

#[inline]
pub fn is_mate_score(score: i32) -> bool {
    score.abs() > TB_WIN
}

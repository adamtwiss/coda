/// Transposition table with lockless concurrent access.
/// 5-slot buckets, cache-line aligned (64 bytes).
/// Matches GoChess: parallel arrays, 32-bit key verification, power-of-2 indexing.

use crate::types::*;

pub const TT_FLAG_NONE: u8 = 0;
pub const TT_FLAG_UPPER: u8 = 1; // All-node (fail-low, score <= alpha)
pub const TT_FLAG_LOWER: u8 = 2; // Cut-node (fail-high, score >= beta)
pub const TT_FLAG_EXACT: u8 = 3; // PV-node (exact score)

const BUCKET_SIZE: usize = 5;

/// Packed data layout (64 bits) — same as GoChess:
///   bits  0-15:  best move (16 bits)
///   bits 16-17:  flag (2 bits)
///   bits 18-31:  staticEval (14 bits, signed, biased)
///   bits 32-47:  score (16 bits, signed)
///   bits 48-55:  depth (8 bits, unsigned)
///   bits 56-63:  generation (8 bits)

#[inline(always)]
fn pack_data(best_move: Move, flag: u8, static_eval: i32, score: i32, depth: i32, generation: u8) -> u64 {
    let mv = best_move as u64;
    let f = (flag as u64 & 3) << 16;
    // Bias static_eval to unsigned 14-bit: clamp ±8191, add 8192, mask
    let se_clamped = static_eval.clamp(-8191, 8191) as i16;
    let se14 = ((se_clamped as u16).wrapping_add(8192)) & 0x3FFF;
    let se = (se14 as u64) << 18;
    let sc = ((score as i16 as u16) as u64) << 32;
    let d = ((depth.max(0) as u8) as u64) << 48;
    let g = (generation as u64) << 56;
    mv | f | se | sc | d | g
}

#[inline(always)]
fn unpack_move(data: u64) -> Move {
    (data & 0xFFFF) as Move
}

#[inline(always)]
fn unpack_flag(data: u64) -> u8 {
    ((data >> 16) & 3) as u8
}

#[inline(always)]
fn unpack_static_eval(data: u64) -> i32 {
    let se14 = ((data >> 18) & 0x3FFF) as u16;
    (se14 as i16 - 8192) as i32
}

#[inline(always)]
fn unpack_score(data: u64) -> i32 {
    ((data >> 32) as i16) as i32
}

#[inline(always)]
fn unpack_depth(data: u64) -> i32 {
    ((data >> 48) & 0xFF) as i32
}

#[inline(always)]
fn unpack_generation(data: u64) -> u8 {
    (data >> 56) as u8
}

/// A bucket of 5 slots using parallel arrays (matches GoChess layout).
/// data[5] = 40 bytes, keys[5] = 20 bytes, _pad = 4 bytes → 64 bytes.
#[repr(C, align(64))]
struct TTBucket {
    data: [u64; BUCKET_SIZE],    // 40 bytes — packed entry data
    keys: [u32; BUCKET_SIZE],    // 20 bytes — upper32(hash) XOR lower32(data)
    _pad: [u8; 4],               // 4 bytes padding to 64
}

impl TTBucket {
    const EMPTY: Self = TTBucket {
        data: [0; BUCKET_SIZE],
        keys: [0; BUCKET_SIZE],
        _pad: [0; 4],
    };
}

/// The transposition table.
pub struct TT {
    buckets: Vec<TTBucket>,
    mask: usize,  // num_buckets - 1 (power of 2)
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
        let num_buckets_raw = bytes / 64;
        // Round down to power of 2
        let mut size = 1usize;
        while size * 2 <= num_buckets_raw {
            size *= 2;
        }
        let size = size.max(1);
        let mut buckets = Vec::with_capacity(size);
        for _ in 0..size {
            buckets.push(TTBucket::EMPTY);
        }
        TT {
            buckets,
            mask: size - 1,
            generation: 0,
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        for bucket in self.buckets.iter_mut() {
            *bucket = TTBucket::EMPTY;
        }
        self.generation = 0;
    }

    /// Increment generation (called at each new search).
    pub fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// Get the bucket index for a hash (power-of-2 masking, matches GoChess).
    #[inline(always)]
    fn bucket_index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    /// Probe the TT for a position.
    pub fn probe(&self, hash: u64) -> TTEntry {
        let idx = self.bucket_index(hash);
        let bucket = &self.buckets[idx];
        let key_upper = (hash >> 32) as u32;

        for i in 0..BUCKET_SIZE {
            let data = bucket.data[i];
            let stored_key = bucket.keys[i];

            // 32-bit verification: stored_key XOR lower32(data) == upper32(hash)
            if stored_key ^ (data as u32) != key_upper {
                continue;
            }

            let flag = unpack_flag(data);
            if flag == TT_FLAG_NONE {
                continue;
            }

            return TTEntry {
                best_move: unpack_move(data),
                flag,
                static_eval: unpack_static_eval(data),
                score: unpack_score(data),
                depth: unpack_depth(data),
                hit: true,
            };
        }

        TTEntry::miss()
    }

    /// Store an entry in the TT.
    pub fn store(&mut self, hash: u64, best_move: Move, flag: u8, static_eval: i32, score: i32, depth: i32) {
        let idx = self.bucket_index(hash);
        let bucket = &mut self.buckets[idx];
        let gen = self.generation;
        let key_upper = (hash >> 32) as u32;

        let new_data = pack_data(best_move, flag, static_eval, score, depth, gen);
        let new_key = key_upper ^ (new_data as u32);

        // Scan all 5 slots: key match, empty, or worst-scoring
        let mut replace_idx = 0;
        let mut replace_score = i32::MAX;

        for i in 0..BUCKET_SIZE {
            let slot_data = bucket.data[i];
            let slot_key = bucket.keys[i];
            let recovered_upper = slot_key ^ (slot_data as u32);

            let slot_flag = unpack_flag(slot_data);
            let slot_depth = unpack_depth(slot_data);
            let slot_gen = unpack_generation(slot_data);

            // Empty slot: use immediately
            if slot_flag == TT_FLAG_NONE {
                bucket.data[i] = new_data;
                bucket.keys[i] = new_key;
                return;
            }

            // Key match: update if newer generation or sufficiently deep
            if recovered_upper == key_upper {
                if depth > slot_depth - 3 || gen != slot_gen {
                    bucket.data[i] = new_data;
                    bucket.keys[i] = new_key;
                }
                return;
            }

            // Track worst slot for replacement: depth - 4*age
            let age = gen.wrapping_sub(slot_gen) as i32;
            let slot_score = slot_depth - age * 4;
            if slot_score < replace_score {
                replace_score = slot_score;
                replace_idx = i;
            }
        }

        // No key match and no empty slot: replace worst-scoring slot
        bucket.data[replace_idx] = new_data;
        bucket.keys[replace_idx] = new_key;
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
        let sample = (self.mask + 1).min(1000);
        let mut used = 0u32;
        for i in 0..sample {
            for j in 0..BUCKET_SIZE {
                let flag = unpack_flag(self.buckets[i].data[j]);
                if flag != TT_FLAG_NONE {
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

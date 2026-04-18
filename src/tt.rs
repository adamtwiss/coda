/// Transposition table with lockless concurrent access.
/// 5-slot buckets, cache-line aligned (64 bytes).
/// Parallel arrays, 32-bit XOR key verification, power-of-2 indexing.

use crate::types::*;
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};

pub const TT_FLAG_NONE: u8 = 0;
pub const TT_FLAG_EXACT: u8 = 1; // PV-node (exact score)
pub const TT_FLAG_LOWER: u8 = 2; // Cut-node (fail-high, score >= beta)
pub const TT_FLAG_UPPER: u8 = 3; // All-node (fail-low, score <= alpha)

const BUCKET_SIZE: usize = 5;

/// Packed data layout (64 bits):
///   bits  0-15:  best move (16 bits)
///   bits 16-17:  flag (2 bits)
///   bit  18:     tt_pv (1 bit) — was this position a PV node?
///   bits 19-31:  staticEval (13 bits, signed, biased ±4095)
///   bits 32-47:  score (16 bits, signed)
///   bits 48-55:  depth (8 bits, unsigned)
///   bits 56-63:  generation (8 bits)

#[inline(always)]
fn pack_data(best_move: Move, flag: u8, static_eval: i32, score: i32, depth: i32, generation: u8, tt_pv: bool) -> u64 {
    let mv = best_move as u64;
    let f = (flag as u64 & 3) << 16;
    let pv = (tt_pv as u64) << 18;
    // Bias static_eval to unsigned 13-bit: clamp ±4095, add 4096, mask
    let se_clamped = static_eval.clamp(-4095, 4095) as i16;
    let se13 = ((se_clamped as u16).wrapping_add(4096)) & 0x1FFF;
    let se = (se13 as u64) << 19;
    let sc = ((score as i16 as u16) as u64) << 32;
    let d = ((depth as i8 as u8) as u64) << 48;
    let g = (generation as u64) << 56;
    mv | f | pv | se | sc | d | g
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
fn unpack_tt_pv(data: u64) -> bool {
    (data >> 18) & 1 != 0
}

#[inline(always)]
fn unpack_static_eval(data: u64) -> i32 {
    let se13 = ((data >> 19) & 0x1FFF) as u16;
    (se13 as i16 - 4096) as i32
}

#[inline(always)]
fn unpack_score(data: u64) -> i32 {
    ((data >> 32) as i16) as i32
}

#[inline(always)]
fn unpack_depth(data: u64) -> i32 {
    (((data >> 48) & 0xFF) as u8 as i8) as i32
}

#[inline(always)]
fn unpack_generation(data: u64) -> u8 {
    (data >> 56) as u8
}

/// A bucket of 5 slots using parallel arrays.
/// data[5] = 40 bytes, keys[5] = 20 bytes, _pad = 4 bytes → 64 bytes.
/// Uses atomics for lockless Lazy SMP. On x86-64, Relaxed atomics = plain MOV.
#[repr(C, align(64))]
struct TTBucket {
    data: [AtomicU64; BUCKET_SIZE],  // 40 bytes — packed entry data
    keys: [AtomicU32; BUCKET_SIZE],  // 20 bytes — upper32(hash) XOR lower32(data)
    _pad: [u8; 4],                   // 4 bytes padding to 64
}

impl TTBucket {
    fn new_empty() -> Self {
        TTBucket {
            data: [
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0),
            ],
            keys: [
                AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0),
                AtomicU32::new(0), AtomicU32::new(0),
            ],
            _pad: [0; 4],
        }
    }

    fn clear(&self) {
        for i in 0..BUCKET_SIZE {
            self.data[i].store(0, Ordering::Relaxed);
            self.keys[i].store(0, Ordering::Relaxed);
        }
    }
}

/// The transposition table. Thread-safe for Lazy SMP.
pub struct TT {
    buckets: Vec<TTBucket>,
    mask: usize,  // num_buckets - 1 (power of 2)
    generation: std::sync::atomic::AtomicU8,
}

// TT is safe to share: all fields use atomics or are immutable after construction.
unsafe impl Sync for TT {}
unsafe impl Send for TT {}

/// Result of a TT probe.
pub struct TTEntry {
    pub best_move: Move,
    pub flag: u8,
    pub static_eval: i32,
    pub score: i32,
    pub depth: i32,
    pub tt_pv: bool,
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
            tt_pv: false,
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
            buckets.push(TTBucket::new_empty());
        }
        // Hint to use huge pages for the TT allocation (Linux transparent huge pages)
        #[cfg(target_os = "linux")]
        unsafe {
            let ptr = buckets.as_ptr() as *mut libc::c_void;
            let len = size * std::mem::size_of::<TTBucket>();
            libc::madvise(ptr, len, libc::MADV_HUGEPAGE);
        }
        TT {
            buckets,
            mask: size - 1,
            generation: std::sync::atomic::AtomicU8::new(0),
        }
    }

    /// Clear all entries.
    pub fn clear(&self) {
        for bucket in self.buckets.iter() {
            bucket.clear();
        }
    }

    /// Increment generation (called at each new search).
    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the bucket index for a hash (power-of-2 masking).
    #[inline(always)]
    fn bucket_index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    /// Probe the TT for a position. Lock-free via atomic loads.
    pub fn probe(&self, hash: u64) -> TTEntry {
        let idx = self.bucket_index(hash);
        let bucket = &self.buckets[idx];
        let key_upper = (hash >> 32) as u32;

        for i in 0..BUCKET_SIZE {
            let data = bucket.data[i].load(Ordering::Relaxed);
            let stored_key = bucket.keys[i].load(Ordering::Relaxed);

            // 32-bit XOR verification: detects torn reads from concurrent writes
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
                tt_pv: unpack_tt_pv(data),
                hit: true,
            };
        }

        TTEntry::miss()
    }

    /// Store an entry in the TT. Lock-free via atomic stores.
    /// Store an entry: (key, depth, score, flag, move, staticEval, isPV)
    pub fn store(&self, hash: u64, depth: i32, score: i32, flag: u8, best_move: Move, static_eval: i32, is_pv: bool) {
        let idx = self.bucket_index(hash);
        let bucket = &self.buckets[idx];
        let gen = self.generation.load(Ordering::Relaxed);
        let key_upper = (hash >> 32) as u32;

        let new_data = pack_data(best_move, flag, static_eval, score, depth, gen, is_pv);
        let new_key = key_upper ^ (new_data as u32);

        // Scan all 5 slots: key match, empty, or worst-scoring
        let mut replace_idx = 0;
        let mut replace_score = i32::MAX;

        for i in 0..BUCKET_SIZE {
            let slot_data = bucket.data[i].load(Ordering::Relaxed);
            let slot_key = bucket.keys[i].load(Ordering::Relaxed);
            let recovered_upper = slot_key ^ (slot_data as u32);

            let slot_flag = unpack_flag(slot_data);
            let slot_depth = unpack_depth(slot_data);
            let slot_gen = unpack_generation(slot_data);

            // Empty slot: use immediately
            if slot_flag == TT_FLAG_NONE {
                bucket.data[i].store(new_data, Ordering::Relaxed);
                bucket.keys[i].store(new_key, Ordering::Relaxed);
                return;
            }

            // Key match: update if newer generation or sufficiently deep
            if recovered_upper == key_upper {
                if depth > slot_depth - 3 || gen != slot_gen {
                    bucket.data[i].store(new_data, Ordering::Relaxed);
                    bucket.keys[i].store(new_key, Ordering::Relaxed);
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
        bucket.data[replace_idx].store(new_data, Ordering::Relaxed);
        bucket.keys[replace_idx].store(new_key, Ordering::Relaxed);
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
                let flag = unpack_flag(self.buckets[i].data[j].load(Ordering::Relaxed));
                if flag != TT_FLAG_NONE {
                    used += 1;
                }
            }
        }
        used * 1000 / (sample as u32 * BUCKET_SIZE as u32)
    }
}

/// Adjust mate scores for TT storage (add ply).
/// Threshold: MateScore - 100 = 28900.
#[inline]
pub fn score_to_tt(score: i32, ply: i32) -> i32 {
    if score > MATE_SCORE - 100 {
        score + ply
    } else if score < -(MATE_SCORE - 100) {
        score - ply
    } else {
        score
    }
}

/// Adjust mate scores from TT retrieval (subtract ply).
#[inline]
pub fn score_from_tt(score: i32, ply: i32) -> i32 {
    if score > MATE_SCORE - 100 {
        score - ply
    } else if score < -(MATE_SCORE - 100) {
        score + ply
    } else {
        score
    }
}

pub const MATE_SCORE: i32 = 29000; // distinct from Infinity = 30000
pub const TB_WIN: i32 = 28800;

#[inline]
pub fn is_mate_score(score: i32) -> bool {
    score.abs() > MATE_SCORE - 100
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let test_cases = vec![
            //  (move, flag, staticEval, score, depth, gen, tt_pv)
            (1234u16, 3u8, 150i32, 300i32, 12i32, 5u8, true),     // normal PV
            (0u16, 1u8, -500i32, -200i32, 0i32, 0u8, false),       // negative eval/score
            (5000u16, 2u8, 4095i32, 30000i32, 30i32, 255u8, true), // max values
            (100u16, 3u8, -4095i32, -30000i32, -1i32, 128u8, false), // min values
            (0u16, 0u8, 0i32, 0i32, -1i32, 0u8, false),             // QS depth -1
        ];

        for (mv, flag, se, score, depth, gen, pv) in &test_cases {
            let data = pack_data(*mv as Move, *flag, *se, *score, *depth, *gen, *pv);
            let got_mv = unpack_move(data);
            let got_flag = unpack_flag(data);
            let got_pv = unpack_tt_pv(data);
            let got_se = unpack_static_eval(data);
            let got_score = unpack_score(data);
            let got_depth = unpack_depth(data);
            let got_gen = unpack_generation(data);

            assert_eq!(got_mv, *mv as Move, "move mismatch for {:?}", (mv, flag, se, score, depth, gen));
            assert_eq!(got_flag, *flag, "flag mismatch");
            assert_eq!(got_pv, *pv, "tt_pv mismatch");
            assert_eq!(got_se, (*se).clamp(-4095, 4095), "static_eval mismatch: packed {} got {}", se, got_se);
            assert_eq!(got_score, *score as i16 as i32, "score mismatch: packed {} got {}", score, got_score);
            assert_eq!(got_depth, *depth, "depth mismatch: packed {} got {}", depth, got_depth);
            assert_eq!(got_gen, *gen, "gen mismatch");
        }
    }
    
    #[test]
    fn test_store_probe_roundtrip() {
        crate::init();
        let tt = TT::new(1);
        
        // Store and probe back
        let hash = 0x123456789ABCDEF0u64;
        tt.store(hash, 12, 300, TT_FLAG_EXACT, 1234, 150, true);
        let entry = tt.probe(hash);

        assert!(entry.hit, "should hit");
        assert_eq!(entry.best_move, 1234);
        assert_eq!(entry.flag, TT_FLAG_EXACT);
        assert_eq!(entry.static_eval, 150);
        assert_eq!(entry.score, 300);
        assert_eq!(entry.depth, 12);
        assert!(entry.tt_pv, "should be PV");

        // Store at depth -1 (QS) and probe back
        let hash2 = 0xFEDCBA9876543210u64;
        tt.store(hash2, -1, -50, TT_FLAG_UPPER, 100, -200, false);
        let entry2 = tt.probe(hash2);

        assert!(entry2.hit, "QS entry should hit");
        assert_eq!(entry2.depth, -1, "QS depth should be -1, got {}", entry2.depth);
        assert_eq!(entry2.score, -50);
        assert_eq!(entry2.static_eval, -200);
    }

    // ──────────────────────────────────────────────────────────────────
    // Bucket replacement + XOR-key verification audit tests.
    // ──────────────────────────────────────────────────────────────────

    fn mk_hash(i: u64) -> u64 { 0xDEAD_BEEF_C0DEu64.wrapping_mul(i).wrapping_add(0x12345) }

    /// Basic roundtrip: store → probe returns same data.
    #[test]
    fn tt_store_probe_roundtrip() {
        let tt = TT::new(4); // 4 MB
        let h = mk_hash(1);
        tt.store(h, 5, 42, TT_FLAG_EXACT, 0x1234, -100, true);
        let e = tt.probe(h);
        assert!(e.hit);
        assert_eq!(e.depth, 5);
        assert_eq!(e.score, 42);
        assert_eq!(e.flag, TT_FLAG_EXACT);
        assert_eq!(e.best_move, 0x1234);
        assert_eq!(e.static_eval, -100);
        assert_eq!(e.tt_pv, true);
    }

    /// Five distinct hashes mapping to the same bucket must all coexist
    /// (the bucket has 5 slots). We force a collision by choosing
    /// hashes whose lower bits agree (picking 5 hashes that differ only
    /// in the upper 32 bits but share the bucket index).
    #[test]
    fn tt_five_distinct_hashes_all_fit() {
        let tt = TT::new(4);
        let base = 0x1u64;
        // Shift upper 32 bits so lower bits (bucket index) stay identical.
        let hashes: Vec<u64> = (0..5).map(|i| base | ((i as u64 + 1) << 32)).collect();

        for (i, &h) in hashes.iter().enumerate() {
            tt.store(h, (i + 1) as i32, 100 + i as i32,
                TT_FLAG_EXACT, (i + 10) as u16, 0, false);
        }

        // Verify all 5 probe successfully with correct depth.
        for (i, &h) in hashes.iter().enumerate() {
            let e = tt.probe(h);
            assert!(e.hit, "hash {} should hit", i);
            assert_eq!(e.depth, (i + 1) as i32, "hash {} depth", i);
            assert_eq!(e.score, 100 + i as i32, "hash {} score", i);
        }
    }

    /// XOR-key verification: a probe with a DIFFERENT hash that happens
    /// to collide with the same bucket must not return another entry's
    /// data (stored_key ^ data != key_upper → miss).
    #[test]
    fn tt_xor_verification_prevents_wrong_key_hit() {
        let tt = TT::new(4);
        let h1: u64 = 0x1u64 | (0xAAAA_BBBBu64 << 32);
        let h2: u64 = 0x1u64 | (0xCCCC_DDDDu64 << 32); // different upper, same lower (same bucket)

        tt.store(h1, 5, 42, TT_FLAG_EXACT, 0x100, 0, false);

        // Probing h2 must not return h1's entry.
        let e = tt.probe(h2);
        assert!(!e.hit, "different hash probing same bucket must miss");
    }

    /// Same-key re-store must UPDATE the existing slot, not allocate
    /// a new one. Property: after 10 rewrites of the same hash, the
    /// bucket only has 1 entry with that key.
    #[test]
    fn tt_same_key_updates_same_slot() {
        let tt = TT::new(4);
        let h = mk_hash(42);
        for i in 0..10 {
            tt.store(h, i + 1, i * 10, TT_FLAG_EXACT, (i + 1) as u16, 0, false);
        }
        let e = tt.probe(h);
        assert!(e.hit);
        assert_eq!(e.depth, 10, "last write should be visible");
        assert_eq!(e.score, 90);
    }

    /// Depth-gated replacement on same-key: new store with shallower
    /// depth (depth <= slot_depth - 3) and same generation MUST NOT
    /// overwrite. Protects the deep result from being replaced by a
    /// shallow one during search.
    #[test]
    fn tt_same_key_shallow_does_not_overwrite() {
        let tt = TT::new(4);
        let h = mk_hash(7);
        // Write deep
        tt.store(h, 20, 100, TT_FLAG_EXACT, 0x200, 0, false);
        // Same-gen shallow attempt: depth 16 < 20-3 = 17, should NOT overwrite.
        tt.store(h, 16, 500, TT_FLAG_LOWER, 0x300, 0, false);

        let e = tt.probe(h);
        assert!(e.hit);
        assert_eq!(e.depth, 20, "deep write must survive shallow same-gen overwrite");
        assert_eq!(e.score, 100);

        // Same-gen near-same-depth (20 > 17): overwrites.
        tt.store(h, 19, 999, TT_FLAG_UPPER, 0x400, 0, false);
        let e2 = tt.probe(h);
        // Depth 19 > 20-3 = 17, so overwrite allowed.
        assert_eq!(e2.depth, 19);
        assert_eq!(e2.score, 999);
    }

    /// Store returns early on empty slot; the store-loop scans all
    /// slots and doesn't re-enter a key-matched replacement path
    /// by mistake. Verify sequential fill of 5 slots all land on
    /// distinct slots (property: 5 distinct keys survive, test is
    /// essentially the same as tt_five_distinct_hashes_all_fit but
    /// with a 6th write triggering eviction of exactly one entry).
    #[test]
    fn tt_sixth_key_evicts_one_not_all() {
        let tt = TT::new(4);
        let base = 0x5u64;
        let hashes: Vec<u64> = (0..6).map(|i| base | ((i as u64 + 1) << 32)).collect();

        for (i, &h) in hashes.iter().enumerate() {
            // All same depth so eviction falls back on "worst slot" = first scanned.
            tt.store(h, 10, i as i32, TT_FLAG_EXACT, (i + 1) as u16, 0, false);
        }

        let mut hits = 0;
        for &h in &hashes {
            if tt.probe(h).hit { hits += 1; }
        }
        // With 5 slots and 6 distinct keys, exactly 5 should be present.
        assert_eq!(hits, 5, "exactly 5 of 6 keys survive in a 5-slot bucket");
    }
}

#[cfg(test)]
mod targeted_tests {
    use super::*;

    #[test]
    fn test_divergent_hash_probe() {
        crate::init();
        let tt = TT::new(64);

        let target_hash = 0x5cac71485b008015u64;
        // b8=1, d7=51: move = (51 << 6) | 1 = 3265
        let mv: Move = (51 << 6) | 1; // b8d7

        // Store: (hash, depth, score, flag, move, static_eval, is_pv)
        tt.store(target_hash, -1, -20, TT_FLAG_LOWER, mv, -100, false);

        let entry = tt.probe(target_hash);
        assert!(entry.hit, "Should find the stored entry!");
        assert_eq!(entry.best_move, mv);
        assert_eq!(entry.score, -20);
        assert_eq!(entry.depth, -1);
        assert_eq!(entry.flag, TT_FLAG_LOWER);
        println!("Direct store+probe: OK");

        // Store entries that map to the SAME bucket to test eviction
        let mask = (64 * 1024 * 1024 / 64) - 1;
        let bucket_idx = target_hash as usize & mask;

        // Store 10 entries to the same bucket
        for i in 0..10u64 {
            let h = ((i + 1) << 20) | (bucket_idx as u64);
            if h != target_hash {
                tt.store(h, (i as i32) + 1, 0, TT_FLAG_EXACT, 0, 0, false);
            }
        }

        let entry2 = tt.probe(target_hash);
        if entry2.hit {
            println!("After 10 bucket colliders: Still found! depth={}", entry2.depth);
        } else {
            println!("After 10 bucket colliders: EVICTED!");
        }
    }
}

impl TT {
    /// Dump all non-empty TT entries to a file (debug/diagnostic).
    /// Format: one line per entry: "bucket_idx slot_idx hash depth score flag move static_eval"
    pub fn dump_to_file(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        let num_buckets = self.mask + 1;
        for bi in 0..num_buckets {
            let bucket = &self.buckets[bi];
            for si in 0..BUCKET_SIZE {
                let data = bucket.data[si].load(Ordering::Relaxed);
                let key = bucket.keys[si].load(Ordering::Relaxed);
                let flag = unpack_flag(data);
                if flag == TT_FLAG_NONE { continue; }
                let upper32 = key ^ (data as u32);
                let hash = ((upper32 as u64) << 32) | (bi as u64); // approximate — lower bits are bucket index
                let depth = unpack_depth(data);
                let score = unpack_score(data);
                let mv = unpack_move(data);
                let se = unpack_static_eval(data);
                let gen = unpack_generation(data);
                writeln!(f, "{} {} {:016x} {} {} {} {} {} {}", bi, si, hash, depth, score, flag, mv, se, gen)?;
            }
        }
        Ok(())
    }
}

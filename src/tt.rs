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

/// 2 MB-aligned bucket array with explicit huge-page preference.
///
/// Three-tier allocation strategy, tried in order:
///
/// 1. **Explicit hugetlb pool** (`mmap(MAP_HUGETLB | MAP_ANONYMOUS)`) —
///    deterministic, bypasses THP/khugepaged. Requires the admin to
///    pre-allocate pages: `sudo sysctl -w vm.nr_hugepages=<N>` (one
///    page per 2 MB of TT to cover). When the pool is configured and
///    has enough free pages, the TT is guaranteed to land on 2 MB
///    physical pages — visible as `Private_Hugetlb` in `/proc/<pid>/smaps`.
///
/// 2. **Aligned heap + MADV_HUGEPAGE + MADV_COLLAPSE** — relies on the
///    kernel's transparent huge-page machinery. Works when THP is
///    functional; opportunistic.
///
/// 3. **Aligned heap + madvise hint only** — last resort; lets
///    khugepaged promote lazily over many scan intervals (or never, on
///    kernels where THP is silently broken — observed on Ubuntu HWE
///    6.8.0-110 with `full_scans` incrementing but `pages_collapsed=0`).
///
/// All three return 2 MB-aligned memory. The sanity test
/// `test_tt_allocation_is_2mb_aligned` pins the alignment invariant so
/// future allocator regressions break the build rather than silently
/// undoing any huge-page win.
///
/// The previous implementation used `Vec::with_capacity` → glibc malloc
/// → 16-byte-aligned pages, so `madvise(MADV_HUGEPAGE)` was a no-op and
/// Coda fell back to 4 KB pages on every Linux host. Verified on the
/// production lichess box: `AnonHugePages: 0` for live coda with a
/// 1 GB TT before this fix.
#[cfg(target_os = "linux")]
struct AlignedBuckets {
    ptr: std::ptr::NonNull<TTBucket>,
    len: usize,
    /// How the mapping was obtained — determines the Drop path.
    backing: AllocBacking,
}

#[cfg(target_os = "linux")]
enum AllocBacking {
    /// mmap(MAP_HUGETLB): released via munmap.
    Hugetlb { size: usize },
    /// std::alloc: released via dealloc.
    Heap { layout: std::alloc::Layout },
}

#[cfg(target_os = "linux")]
impl AlignedBuckets {
    const HUGE_PAGE: usize = 2 * 1024 * 1024; // 2 MB

    fn new(len: usize) -> Self {
        let bytes = len.checked_mul(std::mem::size_of::<TTBucket>()).expect("TT size overflow");
        // Layout/Hugetlb require size to be a multiple of 2 MB.
        let size = (bytes + Self::HUGE_PAGE - 1) & !(Self::HUGE_PAGE - 1);

        // Tier 1: explicit hugetlb mapping. Guaranteed real huge pages
        // when the pool is configured; fails cleanly otherwise.
        if let Some(ab) = Self::try_hugetlb(len, size) {
            return ab;
        }

        // Tier 2/3: 2 MB-aligned heap allocation + THP hints.
        Self::from_heap_with_thp(len, size)
    }

    fn try_hugetlb(len: usize, size: usize) -> Option<Self> {
        // SAFETY: `mmap` with these flags is well-defined; we check for
        // MAP_FAILED. The returned mapping is owned until munmap.
        unsafe {
            let raw = libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
                -1,
                0,
            );
            if raw == libc::MAP_FAILED {
                return None;
            }
            // MAP_ANONYMOUS pages are kernel-zeroed; no memset needed.
            let ptr = std::ptr::NonNull::new(raw as *mut TTBucket)?;
            Some(AlignedBuckets {
                ptr,
                len,
                backing: AllocBacking::Hugetlb { size },
            })
        }
    }

    fn from_heap_with_thp(len: usize, size: usize) -> Self {
        use std::alloc::{alloc_zeroed, Layout};

        let layout = Layout::from_size_align(size, Self::HUGE_PAGE).expect("TT layout");
        // SAFETY: size/align are valid (checked by Layout), allocation is
        // zero-init, we own the returned pointer until Drop.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = std::ptr::NonNull::new(raw as *mut TTBucket)
            .unwrap_or_else(|| std::alloc::handle_alloc_error(layout));

        // MADV_HUGEPAGE sets the `hg` VmFlag; MADV_COLLAPSE (kernel 6.1+)
        // synchronously promotes if the kernel is willing. Both return
        // benign errors on older / misbehaving kernels and we accept the
        // 4 KB fallback silently.
        unsafe {
            let raw_ptr = ptr.as_ptr() as *mut libc::c_void;
            libc::madvise(raw_ptr, size, libc::MADV_HUGEPAGE);

            // Force-populate every 4 KB page so MADV_COLLAPSE has
            // something to collapse (alloc_zeroed is logically zero but
            // physical pages don't fault in until written).
            let page = 4 * 1024;
            let mut off = 0usize;
            while off < size {
                std::ptr::write_volatile((raw_ptr as *mut u8).add(off), 0u8);
                off += page;
            }

            // MADV_COLLAPSE = 25 (Linux 6.1+); not in libc we pin.
            const MADV_COLLAPSE: libc::c_int = 25;
            libc::madvise(raw_ptr, size, MADV_COLLAPSE);
        }

        AlignedBuckets {
            ptr,
            len,
            backing: AllocBacking::Heap { layout },
        }
    }

    /// True if this instance is backed by an explicit hugetlb mapping.
    /// Used by the constructor to log which tier served the TT.
    fn is_hugetlb(&self) -> bool {
        matches!(self.backing, AllocBacking::Hugetlb { .. })
    }
}

#[cfg(target_os = "linux")]
impl Drop for AlignedBuckets {
    fn drop(&mut self) {
        // SAFETY: `backing` records exactly how `ptr` was obtained; we
        // match it with the correct deallocator. TTBucket is atomic-only
        // so has no meaningful Drop.
        unsafe {
            match self.backing {
                AllocBacking::Hugetlb { size } => {
                    libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, size);
                }
                AllocBacking::Heap { layout } => {
                    std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
impl std::ops::Deref for AlignedBuckets {
    type Target = [TTBucket];
    fn deref(&self) -> &[TTBucket] {
        // SAFETY: pointer is valid for `len` TTBuckets, zero-initialised
        // which is a valid bit pattern for all AtomicU64/AtomicU32 fields.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

/// The transposition table. Thread-safe for Lazy SMP.
pub struct TT {
    #[cfg(target_os = "linux")]
    buckets: AlignedBuckets,
    #[cfg(not(target_os = "linux"))]
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

        #[cfg(target_os = "linux")]
        let buckets = {
            let ab = AlignedBuckets::new(size);
            // Announce the path we took — helps confirm huge pages are
            // actually in play without digging through /proc/<pid>/smaps.
            let tier = if ab.is_hugetlb() { "explicit hugetlb (MAP_HUGETLB)" } else { "aligned heap + THP" };
            let mb_alloc = (size * std::mem::size_of::<TTBucket>()) >> 20;
            let _ = mb_alloc; // kept for potential future reporting
            println!("info string TT {} MB via {} ({} buckets)", mb, tier, size);
            ab
        };

        #[cfg(not(target_os = "linux"))]
        let buckets = {
            let mut v = Vec::with_capacity(size);
            for _ in 0..size {
                v.push(TTBucket::new_empty());
            }
            v
        };

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
    ///
    /// Memory ordering on aarch64: load `keys` first with Acquire, then `data`
    /// with Acquire. Pairs with the Release stores in `store()` so any reader
    /// observing a new key is guaranteed to see the matching new data.
    /// On x86 this is essentially free (Acquire/Release degenerate to Relaxed
    /// in the hardware ordering); on aarch64 it adds a `dmb ishld` barrier
    /// per load — necessary because aarch64's weaker model would otherwise
    /// allow the writer's two stores to be observed out of order, which
    /// could let the XOR check accept old data under a new key by
    /// coincidental 32-bit collision.
    pub fn probe(&self, hash: u64) -> TTEntry {
        let idx = self.bucket_index(hash);
        let bucket = &self.buckets[idx];
        let key_upper = (hash >> 32) as u32;

        for i in 0..BUCKET_SIZE {
            // Order matters: load key (Acquire) BEFORE data so the
            // Acquire-Release synchronization on the key store carries the
            // happens-before edge to the data load below.
            let stored_key = bucket.keys[i].load(Ordering::Acquire);
            let data = bucket.data[i].load(Ordering::Acquire);

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
            // Probe-equivalent loads: key first (Acquire) so we see the
            // matching data write on aarch64.
            let slot_key = bucket.keys[i].load(Ordering::Acquire);
            let slot_data = bucket.data[i].load(Ordering::Acquire);
            let recovered_upper = slot_key ^ (slot_data as u32);

            let slot_flag = unpack_flag(slot_data);
            let slot_depth = unpack_depth(slot_data);
            let slot_gen = unpack_generation(slot_data);

            // Empty slot: use immediately. Store data first (Release) then
            // key (Release). The key Release publishes the data write so any
            // probe seeing the new key on another core is guaranteed to see
            // the matching new data.
            if slot_flag == TT_FLAG_NONE {
                bucket.data[i].store(new_data, Ordering::Release);
                bucket.keys[i].store(new_key, Ordering::Release);
                return;
            }

            // Key match: update if EXACT bound, newer generation, or
            // sufficiently deep. EXACT-always-wins isolates the rare-fire
            // half of the SF replacement gate (SF tt.cpp:101 first clause)
            // while keeping Coda's existing depth threshold and ignoring
            // the +2*pv depth bonus and `-3 → -4` threshold change tested
            // separately in sibling branches.
            if recovered_upper == key_upper {
                if flag == TT_FLAG_EXACT || depth > slot_depth - 3 || gen != slot_gen {
                    bucket.data[i].store(new_data, Ordering::Release);
                    bucket.keys[i].store(new_key, Ordering::Release);
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
        bucket.data[replace_idx].store(new_data, Ordering::Release);
        bucket.keys[replace_idx].store(new_key, Ordering::Release);
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

/// Adjust mate and TB scores for TT storage (add ply).
///
/// C5 (2026-04-22 audit): interior TB probes return `TB_WIN - ply`,
/// which is 100cp below the original `MATE_SCORE - 100` threshold. That
/// meant TB scores passed through TT store/load unadjusted, so a TB
/// score stored at ply=10 and retrieved at ply=20 was 10cp too
/// optimistic. Threshold widened to `TB_WIN - 128` to cover the full
/// `[TB_WIN - MAX_PLY, MATE_SCORE]` range (MAX_PLY=64 in search).
/// Normal evals never reach this range.
#[inline]
pub fn score_to_tt(score: i32, ply: i32) -> i32 {
    if score > TB_WIN - 128 {
        score + ply
    } else if score < -(TB_WIN - 128) {
        score - ply
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

/// Adjust mate and TB scores from TT retrieval (subtract ply). See
/// `score_to_tt` for the threshold rationale.
#[inline]
pub fn score_from_tt(score: i32, ply: i32) -> i32 {
    if score > TB_WIN - 128 {
        score - ply
    } else if score < -(TB_WIN - 128) {
        score + ply
    } else {
        score
    }
}

/// P3 50mr-threatened mate downgrade (Reckless pattern). A stored
/// mate-in-N is only reachable if N <= 100 - halfmove; otherwise
/// the 50-move rule claims the draw first. Return a TB-level win
/// signal instead of a false mate.
///
/// **Apply only at cutoff return sites**, not at every TT-score
/// read. Many downstream checks (`< MATE_SCORE - 100`) filter out
/// mate scores specifically; pushing a downgraded-mate through them
/// changes the meaning of those filters and enables unintended
/// extensions / cutoffs / refinements.
#[inline]
pub fn downgrade_50mr_mate(adjusted_score: i32, ply: i32, halfmove: u16) -> i32 {
    let halfmove = halfmove as i32;
    if adjusted_score > MATE_SCORE - 100 && MATE_SCORE - adjusted_score > 100 - halfmove {
        return TB_WIN - ply;
    }
    if adjusted_score < -(MATE_SCORE - 100) && MATE_SCORE + adjusted_score > 100 - halfmove {
        return -TB_WIN + ply;
    }
    adjusted_score
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The huge-page fix relies on the TT's base pointer being 2 MB
    /// aligned — otherwise `madvise(MADV_HUGEPAGE)` silently falls back
    /// to 4 KB pages (the bug this commit is addressing). Regressing
    /// the allocator would silently undo the NPS win, so pin the
    /// invariant in a test.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_tt_allocation_is_2mb_aligned() {
        // Exercise sizes on both sides of the 2 MB rounding path.
        for mb in [1usize, 4, 16, 64, 256, 1024] {
            let tt = TT::new(mb);
            let ptr = tt.buckets.as_ptr() as usize;
            const HUGE: usize = 2 * 1024 * 1024;
            assert_eq!(
                ptr & (HUGE - 1), 0,
                "TT base pointer {:#x} (Hash={} MB) is not 2 MB aligned — \
                 MADV_HUGEPAGE will be a no-op on transparent_hugepage=madvise hosts",
                ptr, mb
            );
        }
    }

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

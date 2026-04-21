/// Syzygy probe-result cache, keyed by our native Zobrist hash.
///
/// A probe against shakmaty-syzygy involves a `Board -> shakmaty::Chess`
/// rebuild plus the tablebase's own decompression work (dictionary lookups,
/// permutation normalisation, several `.rtbw`/`.rtbz` reads even when warm
/// in the page cache). On an endgame-heavy search the WDL probe can dominate
/// NPS. This cache short-circuits repeats: a hit is one cache-line atomic
/// load.
///
/// Lockless, Lazy-SMP-safe: each slot holds `(key ^ value, value)` so a torn
/// read is detected by the XOR check and treated as a miss.
///
/// Stores only WDL probes (interior-node use). DTZ is root-only and rare.

use std::sync::atomic::{AtomicU64, Ordering};

#[repr(align(16))]
struct Slot {
    /// key XOR value — on read, `stored_xor ^ value == key` confirms no tear.
    key_xor_value: AtomicU64,
    /// encoded WDL: 0 = empty slot, otherwise `wdl + 3` so valid range 1..=5.
    value: AtomicU64,
}

/// shakmaty-syzygy WDL → Coda score has 5 discrete outputs:
/// -20000 (Loss), -1 (BlessedLoss/MaybeLoss), 0 (Draw), +1 (CursedWin/
/// MaybeWin), +20000 (Win). We encode 1..=5 so 0 = "empty slot".
#[inline(always)]
fn encode(wdl: i32) -> u64 {
    match wdl {
        -20000 => 1,
        -1 => 2,
        0 => 3,
        1 => 4,
        20000 => 5,
        _ => 0, // unknown → treat as empty
    }
}

#[inline(always)]
fn decode(v: u64) -> i32 {
    match v {
        1 => -20000,
        2 => -1,
        3 => 0,
        4 => 1,
        5 => 20000,
        _ => 0,
    }
}

pub struct TbCache {
    slots: Box<[Slot]>,
    mask: u64,
    enabled: bool,
}

impl TbCache {
    /// Create a cache of `mb` megabytes. `mb=0` creates a disabled cache.
    pub fn new(mb: usize) -> Self {
        if mb == 0 {
            return Self { slots: Box::new([]), mask: 0, enabled: false };
        }
        let slot_bytes = std::mem::size_of::<Slot>();
        let desired = (mb * 1024 * 1024) / slot_bytes;
        // Round down to a power of two so indexing is a mask.
        let mut n = 1usize;
        while n * 2 <= desired { n *= 2; }
        n = n.max(1024); // small lower bound to keep the cache useful at tiny MB
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(Slot {
                key_xor_value: AtomicU64::new(0),
                value: AtomicU64::new(0),
            });
        }
        Self {
            slots: v.into_boxed_slice(),
            mask: (n as u64) - 1,
            enabled: true,
        }
    }

    #[inline(always)]
    pub fn enabled(&self) -> bool { self.enabled }

    /// Probe by Zobrist key. Returns the cached WDL if the slot matches.
    #[inline]
    pub fn probe(&self, key: u64) -> Option<i32> {
        if !self.enabled { return None; }
        let slot = &self.slots[(key & self.mask) as usize];
        let kxv = slot.key_xor_value.load(Ordering::Relaxed);
        let val = slot.value.load(Ordering::Relaxed);
        if val != 0 && kxv ^ val == key {
            Some(decode(val))
        } else {
            None
        }
    }

    /// Store a probe result. No-op when disabled.
    #[inline]
    pub fn store(&self, key: u64, wdl: i32) {
        if !self.enabled { return; }
        let slot = &self.slots[(key & self.mask) as usize];
        let val = encode(wdl);
        // Write value first, then the XOR key. A concurrent reader that
        // observes an intermediate state (old key_xor_value, new value, or
        // vice-versa) fails the XOR check and falls through to shakmaty —
        // which is always safe because shakmaty probes are pure functions
        // of position.
        slot.value.store(val, Ordering::Relaxed);
        slot.key_xor_value.store(key ^ val, Ordering::Relaxed);
    }

    /// Clear all slots. Called on `ucinewgame` and when resized.
    pub fn clear(&self) {
        for slot in self.slots.iter() {
            slot.key_xor_value.store(0, Ordering::Relaxed);
            slot.value.store(0, Ordering::Relaxed);
        }
    }

    /// Current size in MB (for UCI introspection if we want it).
    pub fn size_mb(&self) -> usize {
        if !self.enabled { 0 } else {
            self.slots.len() * std::mem::size_of::<Slot>() / (1024 * 1024)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_cache_never_hits() {
        let c = TbCache::new(0);
        assert!(!c.enabled());
        assert_eq!(c.probe(0xDEADBEEF), None);
        c.store(0xDEADBEEF, 2); // no-op
        assert_eq!(c.probe(0xDEADBEEF), None);
    }

    #[test]
    fn store_probe_roundtrip() {
        let c = TbCache::new(1);
        for wdl in [-20000, -1, 0, 1, 20000] {
            let key = 0x1234_5678_9ABC_DEF0u64.wrapping_mul(wdl as u64 as u32 as u64 + 10);
            c.store(key, wdl);
            assert_eq!(c.probe(key), Some(wdl), "roundtrip failed for wdl={}", wdl);
        }
    }

    #[test]
    fn collision_different_key_misses() {
        let c = TbCache::new(1);
        c.store(0xAAAA, 1);
        // Different key colliding on the low bits: value won't validate.
        // Find one with same low bits:
        let mask = (c.slots.len() as u64) - 1;
        let colliding = 0xAAAA | ((0xBBBBu64) << 32);
        assert_eq!(colliding & mask, 0xAAAA & mask);
        assert_eq!(c.probe(colliding), None, "collision must not hit");
    }

    #[test]
    fn clear_removes_entries() {
        let c = TbCache::new(1);
        c.store(0x1234, -1);
        assert_eq!(c.probe(0x1234), Some(-1));
        c.clear();
        assert_eq!(c.probe(0x1234), None);
    }
}

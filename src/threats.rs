/// Threat feature computation for NNUE.
///
/// Encodes (attacker_piece, attacker_sq, victim_piece, victim_sq) relationships.
/// Each active threat on the board contributes one feature index into the threat
/// accumulator. Feature indices are perspective-relative with king-file mirroring.
///
/// ATTRIBUTION
///
/// Threat inputs are not Coda's idea and not any single engine's. The input
/// encoding was introduced in **Monty** by Viren6 and jw1912, who actively
/// encouraged A/B engines to try it. It then spread quickly through the top
/// engines, in this order:
///
///   Monty (AGPL)         origin, Viren6 / jw1912
///   PlentyChess (GPL-3)  2025-10-12
///   Stockfish (GPL-3)    2025-11-12, SFNNv10
///   Reckless (AGPL)      2025-12-03
///
/// and is now used by Viridithas, Stormphrax, Hobbes and others. See
/// PlentyChess PR #400 for the fullest public account of who contributed what.
///
/// Coda added threat inputs to its existing NNUE architecture. The interaction
/// map and target counts follow the same shape as Stockfish's tables; those are
/// functional facts about which pieces attack which, so every implementation
/// converges on them.
///
/// **Reckless was the implementation Coda referenced while writing this.** Our
/// 2026-07 licence audit found parts of the result too closely modelled on it,
/// and those parts were reimplemented in Coda's own expression (see
/// docs/license_analysis_2026-07-13.md). Rewriting the expression was the right
/// fix for the licence question, but it is not a reason to drop the credit, and
/// for a while we did drop it. This notice restores it.
///
/// Total features: ~66,864 (depends on piece-pair filtering).


/// Test-only serialization guard for the process-global `EMIT_XRAY`.
/// `cargo test` runs tests concurrently; any test that *mutates* `EMIT_XRAY`
/// (the x-ray-OFF fuzzers) and any test that *depends* on its default value
/// (threat enumeration/consistency tests) MUST hold this lock for their whole
/// body, or the mutator's window corrupts a concurrent reader (spurious
/// failures — see the x-ray-off test-isolation fix).
///

#[cfg(feature = "profile-threats")]
pub mod apply_stats {
    //! apply_threat_deltas delta-count histogram.
    //! Used to decide whether a long-tail of high-delta-count moves is
    //! worth capping/batching, or whether delta counts are uniform.
    //! High-end buckets are tightened to surface behavior near the
    //! MAX_THREAT_DELTAS=128 cap (cap-hits land in CAP_HIT).
    use std::sync::atomic::{AtomicU64, Ordering};

    static CALLS: AtomicU64 = AtomicU64::new(0);
    static TOTAL_DELTAS: AtomicU64 = AtomicU64::new(0);
    static MAX_OBSERVED: AtomicU64 = AtomicU64::new(0);
    static B0: AtomicU64 = AtomicU64::new(0);
    static B1_4: AtomicU64 = AtomicU64::new(0);
    static B5_8: AtomicU64 = AtomicU64::new(0);
    static B9_12: AtomicU64 = AtomicU64::new(0);
    static B13_16: AtomicU64 = AtomicU64::new(0);
    static B17_24: AtomicU64 = AtomicU64::new(0);
    static B25_32: AtomicU64 = AtomicU64::new(0);
    static B33_48: AtomicU64 = AtomicU64::new(0);
    static B49_64: AtomicU64 = AtomicU64::new(0);
    static B65_96: AtomicU64 = AtomicU64::new(0);
    static B97_127: AtomicU64 = AtomicU64::new(0);
    static CAP_HIT: AtomicU64 = AtomicU64::new(0); // n == MAX_THREAT_DELTAS (128)
    // Cancellation instrumentation: how many streamed weight rows (adds+subs)
    // are net-zero same-index add/sub pairs SF would cancel (FusedUpdateData)
    // but Coda currently streams twice. STREAMED = total rows applied;
    // CANCELLED = rows that net to zero (2 per matched add/sub index pair).
    static STREAMED_ROWS: AtomicU64 = AtomicU64::new(0);
    static CANCELLED_ROWS: AtomicU64 = AtomicU64::new(0);
    // Generation-side: deltas GENERATED per move (counted at absorb, once per
    // move). Caching/laziness-immune — unlike deltas/apply-call which is
    // inflated by lazy-replay depth + eval-cache materialization frequency.
    // This is the architecture-pure "threat-model density" number.
    static GEN_MOVES: AtomicU64 = AtomicU64::new(0);
    static GEN_DELTAS: AtomicU64 = AtomicU64::new(0);
    // Laziness sizing: how many generated entries are ever REPLAYED (unique —
    // first walkback visit only, dual counts once). generated - consumed =
    // delta-generation work that lazy generation could skip entirely.
    static GEN_CONSUMED: AtomicU64 = AtomicU64::new(0);

    /// Record per-move generated delta count (once per make_move, at absorb).
    #[inline(always)]
    pub fn record_generated(n: usize) {
        GEN_MOVES.fetch_add(1, Ordering::Relaxed);
        GEN_DELTAS.fetch_add(n as u64, Ordering::Relaxed);
    }

    /// Record first replay consumption of a generated entry (unique per
    /// push/absorb generation instance; update_dual marks once).
    #[inline(always)]
    pub fn record_first_consume() {
        GEN_CONSUMED.fetch_add(1, Ordering::Relaxed);
    }

    // Refresh-cause split (walkback/Finny scoping):
    // 0 = king mirror crossing, 1 = no accurate ancestor, 2 = delta overflow.
    static REFRESH_CAUSE: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    #[inline(always)]
    pub fn record_refresh_cause(c: usize) {
        REFRESH_CAUSE[c.min(2)].fetch_add(1, Ordering::Relaxed);
    }

    // Replay-gap distribution: plies replayed per materialization (index -
    // ancestor). double_inc_update (SF cross-ply cancellation) can ONLY help
    // when gap >= 2; gap == 1 has zero cross-ply opportunity. Bounds the lever.
    static REPLAY_CALLS: AtomicU64 = AtomicU64::new(0);
    static REPLAY_GAP_SUM: AtomicU64 = AtomicU64::new(0);
    static REPLAY_GAP1: AtomicU64 = AtomicU64::new(0);
    static REPLAY_GAP2P: AtomicU64 = AtomicU64::new(0);

    #[inline(always)]
    pub fn record_replay_gap(gap: usize) {
        REPLAY_CALLS.fetch_add(1, Ordering::Relaxed);
        REPLAY_GAP_SUM.fetch_add(gap as u64, Ordering::Relaxed);
        if gap >= 2 { REPLAY_GAP2P.fetch_add(1, Ordering::Relaxed); }
        else { REPLAY_GAP1.fetch_add(1, Ordering::Relaxed); }
    }

    // Cross-ply (gap>=2 span) cancellation: net-zero add/sub pairs over the
    // COMBINED multi-ply span. Span rate vs the 3.8% per-ply rate = the extra
    // double_inc_update would capture. Plus the gap>=2 rows are where the
    // intermediate-copy saving lives.
    static XPLY_STREAMED: AtomicU64 = AtomicU64::new(0);
    static XPLY_CANCELLED: AtomicU64 = AtomicU64::new(0);
    pub fn record_crossply(adds: &[usize], subs: &[usize]) {
        XPLY_STREAMED.fetch_add((adds.len() + subs.len()) as u64, Ordering::Relaxed);
        let mut a = adds.to_vec(); a.sort_unstable();
        let mut s = subs.to_vec(); s.sort_unstable();
        let (mut i, mut j, mut c) = (0usize, 0usize, 0u64);
        while i < a.len() && j < s.len() {
            if a[i] == s[j] { c += 2; i += 1; j += 1; }
            else if a[i] < s[j] { i += 1; } else { j += 1; }
        }
        XPLY_CANCELLED.fetch_add(c, Ordering::Relaxed);
    }

    /// Count net-zero same-index add/sub pairs in one apply call. Each matched
    /// (idx in adds AND idx in subs) pair = 2 streamed rows that cancel.
    /// O(n log n); profile-only, allocations fine.
    pub fn record_cancel(adds: &[usize], subs: &[usize]) {
        STREAMED_ROWS.fetch_add((adds.len() + subs.len()) as u64, Ordering::Relaxed);
        let mut a = adds.to_vec(); a.sort_unstable();
        let mut s = subs.to_vec(); s.sort_unstable();
        let (mut i, mut j, mut cancelled) = (0usize, 0usize, 0u64);
        while i < a.len() && j < s.len() {
            if a[i] == s[j] { cancelled += 2; i += 1; j += 1; }
            else if a[i] < s[j] { i += 1; }
            else { j += 1; }
        }
        CANCELLED_ROWS.fetch_add(cancelled, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn record(n: usize) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        TOTAL_DELTAS.fetch_add(n as u64, Ordering::Relaxed);
        // Update max with CAS loop.
        let mut cur = MAX_OBSERVED.load(Ordering::Relaxed);
        while (n as u64) > cur {
            match MAX_OBSERVED.compare_exchange_weak(cur, n as u64, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }
        let bucket = match n {
            0 => &B0,
            1..=4 => &B1_4,
            5..=8 => &B5_8,
            9..=12 => &B9_12,
            13..=16 => &B13_16,
            17..=24 => &B17_24,
            25..=32 => &B25_32,
            33..=48 => &B33_48,
            49..=64 => &B49_64,
            65..=96 => &B65_96,
            97..=127 => &B97_127,
            _ => &CAP_HIT,
        };
        bucket.fetch_add(1, Ordering::Relaxed);
    }

    pub fn report() {
        let c = CALLS.load(Ordering::Relaxed);
        if c == 0 { eprintln!("apply_threat_deltas stats: 0 calls"); return; }
        let td = TOTAL_DELTAS.load(Ordering::Relaxed);
        let max = MAX_OBSERVED.load(Ordering::Relaxed);
        let cap = CAP_HIT.load(Ordering::Relaxed);
        let pct = |n: u64| -> f64 { 100.0 * n as f64 / c.max(1) as f64 };
        eprintln!("apply_threat_deltas: {} calls, total {} deltas, avg {:.2}, max {}, cap-hits {} ({:.4}%)",
            c, td, td as f64 / c.max(1) as f64, max, cap, pct(cap));
        eprintln!("  0:       {:>10} ({:.1}%)", B0.load(Ordering::Relaxed), pct(B0.load(Ordering::Relaxed)));
        eprintln!("  1-4:     {:>10} ({:.1}%)", B1_4.load(Ordering::Relaxed), pct(B1_4.load(Ordering::Relaxed)));
        eprintln!("  5-8:     {:>10} ({:.1}%)", B5_8.load(Ordering::Relaxed), pct(B5_8.load(Ordering::Relaxed)));
        eprintln!("  9-12:    {:>10} ({:.1}%)", B9_12.load(Ordering::Relaxed), pct(B9_12.load(Ordering::Relaxed)));
        eprintln!("  13-16:   {:>10} ({:.1}%)", B13_16.load(Ordering::Relaxed), pct(B13_16.load(Ordering::Relaxed)));
        eprintln!("  17-24:   {:>10} ({:.1}%)", B17_24.load(Ordering::Relaxed), pct(B17_24.load(Ordering::Relaxed)));
        eprintln!("  25-32:   {:>10} ({:.1}%)", B25_32.load(Ordering::Relaxed), pct(B25_32.load(Ordering::Relaxed)));
        eprintln!("  33-48:   {:>10} ({:.1}%)", B33_48.load(Ordering::Relaxed), pct(B33_48.load(Ordering::Relaxed)));
        eprintln!("  49-64:   {:>10} ({:.1}%)", B49_64.load(Ordering::Relaxed), pct(B49_64.load(Ordering::Relaxed)));
        eprintln!("  65-96:   {:>10} ({:.1}%)", B65_96.load(Ordering::Relaxed), pct(B65_96.load(Ordering::Relaxed)));
        eprintln!("  97-127:  {:>10} ({:.1}%)", B97_127.load(Ordering::Relaxed), pct(B97_127.load(Ordering::Relaxed)));
        eprintln!("  128(cap):{:>10} ({:.4}%)  [forced fallback]", cap, pct(cap));
        let streamed = STREAMED_ROWS.load(Ordering::Relaxed);
        let cancelled = CANCELLED_ROWS.load(Ordering::Relaxed);
        eprintln!(
            "  CANCELLATION: {} rows streamed, {} cancellable (net-zero add/sub pairs) = {:.2}% wasted bandwidth (SF cancels these)",
            streamed, cancelled,
            100.0 * cancelled as f64 / streamed.max(1) as f64
        );
        let gm = GEN_MOVES.load(Ordering::Relaxed);
        let gd = GEN_DELTAS.load(Ordering::Relaxed);
        eprintln!(
            "  GENERATED (caching-immune): {} moves, {} deltas, avg {:.2} deltas/move (vs deltas/apply-call above which lazy-replay inflates)",
            gm, gd, gd as f64 / gm.max(1) as f64
        );
        {
            let mc = REFRESH_CAUSE[0].load(Ordering::Relaxed);
            let na = REFRESH_CAUSE[1].load(Ordering::Relaxed);
            let ov = REFRESH_CAUSE[2].load(Ordering::Relaxed);
            let tot = (mc + na + ov).max(1);
            eprintln!("  REFRESH CAUSES (can_update=None): mirror-cross {} ({:.1}%), no-ancestor {} ({:.1}%), overflow {} ({:.1}%)",
                mc, 100.0*mc as f64/tot as f64, na, 100.0*na as f64/tot as f64, ov, 100.0*ov as f64/tot as f64);
        }
        let gc = GEN_CONSUMED.load(Ordering::Relaxed);
        eprintln!(
            "  CONSUMED (unique first-replay): {} of {} generated = {:.1}% (remainder = {:.1}% of delta-generation work a lazy scheme could skip)",
            gc, gm,
            100.0 * gc as f64 / gm.max(1) as f64,
            100.0 * (gm.saturating_sub(gc)) as f64 / gm.max(1) as f64
        );
        let rc = REPLAY_CALLS.load(Ordering::Relaxed);
        let g2p = REPLAY_GAP2P.load(Ordering::Relaxed);
        eprintln!(
            "  REPLAY GAP: {} materializations, avg {:.3} plies, gap==1 {:.1}%, gap>=2 {:.1}% (only gap>=2 can use double_inc_update cross-ply cancellation)",
            rc, REPLAY_GAP_SUM.load(Ordering::Relaxed) as f64 / rc.max(1) as f64,
            100.0 * REPLAY_GAP1.load(Ordering::Relaxed) as f64 / rc.max(1) as f64,
            100.0 * g2p as f64 / rc.max(1) as f64
        );
        let xs = XPLY_STREAMED.load(Ordering::Relaxed);
        let xc = XPLY_CANCELLED.load(Ordering::Relaxed);
        eprintln!(
            "  CROSS-PLY (gap>=2 spans): {} rows in spans, {} cancellable over combined span = {:.2}% (vs 3.8% per-ply; excess is double_inc upside #1). These spans are {:.1}% of all streamed rows.",
            xs, xc, 100.0 * xc as f64 / xs.max(1) as f64,
            100.0 * xs as f64 / streamed.max(1) as f64
        );
    }
}

/// Per-position active-feature histogram from threat_accum::refresh.
/// Used to right-size the inference [usize; 256] full-refresh buffer
/// and to compare against training-side MAX_THREAT_ACTIVE distribution.
/// Buckets every 16 from 0..255 plus a 256+ overflow row.
#[cfg(feature = "profile-threats")]
pub mod refresh_stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    const NUM_BUCKETS: usize = 17; // 16 sized buckets + overflow
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static TOTAL_INDICES: AtomicU64 = AtomicU64::new(0);
    static MAX_OBSERVED: AtomicU64 = AtomicU64::new(0);
    static OVERFLOWS: AtomicU64 = AtomicU64::new(0);
    static BUCKETS: [AtomicU64; NUM_BUCKETS] = [
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0),
    ];

    #[inline(always)]
    pub fn record(n: usize, overflowed: bool) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        TOTAL_INDICES.fetch_add(n as u64, Ordering::Relaxed);
        if overflowed { OVERFLOWS.fetch_add(1, Ordering::Relaxed); }
        let mut cur = MAX_OBSERVED.load(Ordering::Relaxed);
        while (n as u64) > cur {
            match MAX_OBSERVED.compare_exchange_weak(cur, n as u64, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }
        let idx = if n >= 256 { 16 } else { n / 16 };
        BUCKETS[idx].fetch_add(1, Ordering::Relaxed);
    }

    pub fn report() {
        let c = CALLS.load(Ordering::Relaxed);
        if c == 0 { eprintln!("threat refresh stats: 0 calls"); return; }
        let total = TOTAL_INDICES.load(Ordering::Relaxed);
        let max = MAX_OBSERVED.load(Ordering::Relaxed);
        let ovf = OVERFLOWS.load(Ordering::Relaxed);
        let pct = |n: u64| -> f64 { 100.0 * n as f64 / c.max(1) as f64 };
        eprintln!("threat refresh: {} calls, total {} active, avg {:.2}, max {}, overflow {} ({:.4}%)",
            c, total, total as f64 / c.max(1) as f64, max, ovf, pct(ovf));
        // Cumulative percentile column makes "what cap covers X% of calls" obvious.
        let mut cum = 0u64;
        for i in 0..16 {
            let lo = i * 16;
            let hi = lo + 15;
            let n = BUCKETS[i].load(Ordering::Relaxed);
            cum += n;
            eprintln!("  {:>3}-{:<3}: {:>10} ({:>5.2}%)  cum {:>5.2}%", lo, hi, n, pct(n), pct(cum));
        }
        let n = BUCKETS[16].load(Ordering::Relaxed);
        eprintln!("  256+   : {:>10} ({:>5.4}%)  [forced fallback]", n, pct(n));
    }
}

#[cfg(feature = "profile-threats")]
pub mod thr_stats {
    //! Per-bench push_threats_for_piece section-level CPU counters.
    //! Gated behind `--features profile-threats` — zero release cost.
    //!
    //! Tracks cycle deltas for each logical block inside the hot
    //! function, plus delta counts. Used to decide whether a
    //! vectorised "direct only" variant is worth
    //! pursuing as a hybrid with our existing scalar x-ray code.
    use std::sync::atomic::{AtomicU64, Ordering};

    static CALLS: AtomicU64 = AtomicU64::new(0);
    // Cycle-timestamp-counter deltas (rdtsc) per section.
    static CYC_DIRECT: AtomicU64 = AtomicU64::new(0);      // step 1: direct threats FROM this piece
    static CYC_OWN_XRAY: AtomicU64 = AtomicU64::new(0);    // step 1b: x-ray FROM this piece
    static CYC_SLIDERS: AtomicU64 = AtomicU64::new(0);     // step 2: sliders seeing this square + Z-finding
    static CYC_SLIDERS_2B: AtomicU64 = AtomicU64::new(0);  // step 2b: sliders x-raying to this square
    static CYC_NONSLIDERS: AtomicU64 = AtomicU64::new(0);  // step 3: pawn/knight/king attackers
    static CYC_TOTAL: AtomicU64 = AtomicU64::new(0);

    // Delta counts emitted per section.
    static DELTAS_DIRECT: AtomicU64 = AtomicU64::new(0);
    static DELTAS_OWN_XRAY: AtomicU64 = AtomicU64::new(0);
    static DELTAS_SLIDERS: AtomicU64 = AtomicU64::new(0);  // step 2 total (includes Z-level)
    static DELTAS_SLIDERS_2B: AtomicU64 = AtomicU64::new(0);
    static DELTAS_NONSLIDERS: AtomicU64 = AtomicU64::new(0);

    // Per-section zero-emitter counters. A section "zero-emitter" is a call
    // that ran through its scalar walks/ray tests but produced no deltas —
    // wasted work. Heavy skew on 2b would justify a tighter early-out.
    static ZERO_DIRECT: AtomicU64 = AtomicU64::new(0);
    static ZERO_OWN_XRAY: AtomicU64 = AtomicU64::new(0);
    static ZERO_SLIDERS: AtomicU64 = AtomicU64::new(0);
    static ZERO_SLIDERS_2B: AtomicU64 = AtomicU64::new(0);
    static ZERO_NONSLIDERS: AtomicU64 = AtomicU64::new(0);

    static OWN_XRAY_NONSLIDER: AtomicU64 = AtomicU64::new(0);
    static OWN_XRAY_NO_DIRECT: AtomicU64 = AtomicU64::new(0);
    static OWN_XRAY_BLOCKERS: AtomicU64 = AtomicU64::new(0);
    static OWN_XRAY_NO_BEHIND: AtomicU64 = AtomicU64::new(0);
    static OWN_XRAY_INVALID: AtomicU64 = AtomicU64::new(0);
    static OWN_XRAY_EMITS: AtomicU64 = AtomicU64::new(0);

    static S2B_NO_CANDIDATES: AtomicU64 = AtomicU64::new(0);
    static S2B_CANDIDATES: AtomicU64 = AtomicU64::new(0);
    static S2B_BLOCKERS_ZERO: AtomicU64 = AtomicU64::new(0);
    static S2B_BLOCKERS_MULTI: AtomicU64 = AtomicU64::new(0);
    static S2B_EXACT_ONE: AtomicU64 = AtomicU64::new(0);
    static S2B_INVALID: AtomicU64 = AtomicU64::new(0);
    static S2B_SQ_EMITS: AtomicU64 = AtomicU64::new(0);
    static S2B_NO_W: AtomicU64 = AtomicU64::new(0);
    static S2B_W_EMITS: AtomicU64 = AtomicU64::new(0);

    #[inline(always)]
    pub fn rdtsc() -> u64 {
        #[cfg(target_arch = "x86_64")]
        unsafe { std::arch::x86_64::_rdtsc() }
        #[cfg(not(target_arch = "x86_64"))]
        { 0 }
    }

    #[inline(always)]
    pub fn record_call() { CALLS.fetch_add(1, Ordering::Relaxed); }

    #[inline(always)]
    pub fn record_section(idx: u8, cycles: u64, deltas: u64) {
        let (cyc, dlt, zero) = match idx {
            0 => (&CYC_DIRECT,      &DELTAS_DIRECT,      &ZERO_DIRECT),
            1 => (&CYC_OWN_XRAY,    &DELTAS_OWN_XRAY,    &ZERO_OWN_XRAY),
            2 => (&CYC_SLIDERS,     &DELTAS_SLIDERS,     &ZERO_SLIDERS),
            3 => (&CYC_SLIDERS_2B,  &DELTAS_SLIDERS_2B,  &ZERO_SLIDERS_2B),
            4 => (&CYC_NONSLIDERS,  &DELTAS_NONSLIDERS,  &ZERO_NONSLIDERS),
            _ => return,
        };
        cyc.fetch_add(cycles, Ordering::Relaxed);
        dlt.fetch_add(deltas, Ordering::Relaxed);
        if deltas == 0 {
            zero.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn record_total(cycles: u64) {
        CYC_TOTAL.fetch_add(cycles, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn record_own_xray_reasons(
        nonslider: u64,
        no_direct: u64,
        blockers: u64,
        no_behind: u64,
        invalid: u64,
        emits: u64,
    ) {
        OWN_XRAY_NONSLIDER.fetch_add(nonslider, Ordering::Relaxed);
        OWN_XRAY_NO_DIRECT.fetch_add(no_direct, Ordering::Relaxed);
        OWN_XRAY_BLOCKERS.fetch_add(blockers, Ordering::Relaxed);
        OWN_XRAY_NO_BEHIND.fetch_add(no_behind, Ordering::Relaxed);
        OWN_XRAY_INVALID.fetch_add(invalid, Ordering::Relaxed);
        OWN_XRAY_EMITS.fetch_add(emits, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn record_s2b_reasons(
        no_candidates: u64,
        candidates: u64,
        blockers_zero: u64,
        blockers_multi: u64,
        exact_one: u64,
        invalid: u64,
        sq_emits: u64,
        no_w: u64,
        w_emits: u64,
    ) {
        S2B_NO_CANDIDATES.fetch_add(no_candidates, Ordering::Relaxed);
        S2B_CANDIDATES.fetch_add(candidates, Ordering::Relaxed);
        S2B_BLOCKERS_ZERO.fetch_add(blockers_zero, Ordering::Relaxed);
        S2B_BLOCKERS_MULTI.fetch_add(blockers_multi, Ordering::Relaxed);
        S2B_EXACT_ONE.fetch_add(exact_one, Ordering::Relaxed);
        S2B_INVALID.fetch_add(invalid, Ordering::Relaxed);
        S2B_SQ_EMITS.fetch_add(sq_emits, Ordering::Relaxed);
        S2B_NO_W.fetch_add(no_w, Ordering::Relaxed);
        S2B_W_EMITS.fetch_add(w_emits, Ordering::Relaxed);
    }

    pub fn report() {
        let c = CALLS.load(Ordering::Relaxed);
        if c == 0 { eprintln!("threats stats: 0 calls (feature not hit)"); return; }
        let tot = CYC_TOTAL.load(Ordering::Relaxed);
        let sections = [
            ("direct     (step 1)  ", CYC_DIRECT.load(Ordering::Relaxed),      DELTAS_DIRECT.load(Ordering::Relaxed),      ZERO_DIRECT.load(Ordering::Relaxed)),
            ("own-xray   (step 1b) ", CYC_OWN_XRAY.load(Ordering::Relaxed),    DELTAS_OWN_XRAY.load(Ordering::Relaxed),    ZERO_OWN_XRAY.load(Ordering::Relaxed)),
            ("sliders    (step 2)  ", CYC_SLIDERS.load(Ordering::Relaxed),     DELTAS_SLIDERS.load(Ordering::Relaxed),     ZERO_SLIDERS.load(Ordering::Relaxed)),
            ("sliders-2b (step 2b) ", CYC_SLIDERS_2B.load(Ordering::Relaxed),  DELTAS_SLIDERS_2B.load(Ordering::Relaxed),  ZERO_SLIDERS_2B.load(Ordering::Relaxed)),
            ("nonsliders (step 3)  ", CYC_NONSLIDERS.load(Ordering::Relaxed),  DELTAS_NONSLIDERS.load(Ordering::Relaxed),  ZERO_NONSLIDERS.load(Ordering::Relaxed)),
        ];
        eprintln!("push_threats_for_piece: {} calls, total {} Mcy", c, tot / 1_000_000);
        for (name, cyc, dlt, zero) in &sections {
            let pct = 100.0 * *cyc as f64 / tot.max(1) as f64;
            let zero_pct = 100.0 * *zero as f64 / c.max(1) as f64;
            eprintln!("  {}  {:>5.1}%   {:>8} Mcy   {:>5.1} cy/call   {:.2} deltas/call   zero-emit: {:>5.1}%",
                name, pct, cyc / 1_000_000,
                *cyc as f64 / c as f64,
                *dlt as f64 / c as f64,
                zero_pct);
        }
        let own_nonslider = OWN_XRAY_NONSLIDER.load(Ordering::Relaxed);
        let own_no_direct = OWN_XRAY_NO_DIRECT.load(Ordering::Relaxed);
        let own_blockers = OWN_XRAY_BLOCKERS.load(Ordering::Relaxed);
        let own_no_behind = OWN_XRAY_NO_BEHIND.load(Ordering::Relaxed);
        let own_invalid = OWN_XRAY_INVALID.load(Ordering::Relaxed);
        let own_emits = OWN_XRAY_EMITS.load(Ordering::Relaxed);
        eprintln!(
            "  own-xray reasons: nonslider={} no-direct={} blockers={} no-behind={} invalid={} emits={}",
            own_nonslider, own_no_direct, own_blockers, own_no_behind, own_invalid, own_emits,
        );

        let s2b_no_candidates = S2B_NO_CANDIDATES.load(Ordering::Relaxed);
        let s2b_candidates = S2B_CANDIDATES.load(Ordering::Relaxed);
        let s2b_blockers_zero = S2B_BLOCKERS_ZERO.load(Ordering::Relaxed);
        let s2b_blockers_multi = S2B_BLOCKERS_MULTI.load(Ordering::Relaxed);
        let s2b_exact_one = S2B_EXACT_ONE.load(Ordering::Relaxed);
        let s2b_invalid = S2B_INVALID.load(Ordering::Relaxed);
        let s2b_sq_emits = S2B_SQ_EMITS.load(Ordering::Relaxed);
        let s2b_no_w = S2B_NO_W.load(Ordering::Relaxed);
        let s2b_w_emits = S2B_W_EMITS.load(Ordering::Relaxed);
        eprintln!(
            "  sliders-2b reasons: no-candidates={} candidates={} blockers0={} blockers2p={} exact1={} invalid={} sq-emits={} no-w={} w-emits={}",
            s2b_no_candidates, s2b_candidates, s2b_blockers_zero, s2b_blockers_multi,
            s2b_exact_one, s2b_invalid, s2b_sq_emits, s2b_no_w, s2b_w_emits,
        );
    }
}

use crate::attacks::*;
use crate::bitboard::*;
use crate::types::*;

/// Cached x86 SIMD tier for the threat-apply dispatch: 2 = AVX-512
/// (f+bw), 1 = AVX2, 0 = scalar. `is_x86_feature_detected!` caches each
/// feature in its own atomic, but the old dispatch paid THREE checks per
/// call (two always-false AVX-512 probes before the AVX2 arm) on AVX2-only
/// hosts, on functions called 1-2× per push plus per replay ply. One
/// OnceLock load + compare replaces all three.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn x86_simd_tier() -> u8 {
    use std::sync::OnceLock;
    // Bypass the cache while a test override is active, for the same reason
    // `isa_max` does: a tier sweep in one process cannot use a memoised value.
    // `cfg(test)` — this function is on the hot path, so release builds keep
    // the plain memoised read with no extra load or branch.
    #[cfg(test)]
    {
        if crate::nnue::ISA_MAX_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) != 0 {
            return compute_x86_simd_tier();
        }
    }
    static TIER: OnceLock<u8> = OnceLock::new();
    *TIER.get_or_init(compute_x86_simd_tier)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn compute_x86_simd_tier() -> u8 {
    let cap = crate::nnue::isa_max();
    if cap >= 3 && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        2
    } else if cap >= 1 && is_x86_feature_detected!("avx2") {
        1
    } else {
        0
    }
}

/// Piece interaction map: which attacker×victim pairs are tracked.
/// Rows = attacker piece type (P/N/B/R/Q/K), columns = victim piece type.
/// -1 = excluded (not tracked). Non-negative values index into target buckets.
/// Symmetric pairs (same piece type) are semi-excluded: only one ordering kept.
const PIECE_INTERACTION_MAP: [[i32; 6]; 6] = [
    [0,  1, -1,  2, -1, -1],  // Pawn attacks:   P, N, _, R, _, _
    [0,  1,  2,  3,  4, -1],  // Knight attacks:  P, N, B, R, Q, _
    [0,  1,  2,  3, -1, -1],  // Bishop attacks:  P, N, B, R, _, _
    [0,  1,  2,  3, -1, -1],  // Rook attacks:    P, N, B, R, _, _
    [0,  1,  2,  3,  4, -1],  // Queen attacks:   P, N, B, R, Q, _
    [0,  1,  2,  3, -1, -1],  // King attacks:    P, N, B, R, _, _
];

/// Per-attacker target count (friendly + enemy combined).
/// Pawn: 3 target types × 2 sides = 6, Knight: 5 × 2 = 10, etc.
const PIECE_TARGET_COUNT: [i32; 6] = [6, 10, 8, 8, 10, 8];

/// Number of colored pieces (white P, white N, ..., black K).
const NUM_COLORED_PIECES: usize = 12;

/// Per (attacker, victim) colored-piece pair, precomputed by `init_threats`:
/// `base` is the first feature index of that pair's (from, to) block, and the
/// two flags decide whether a given occurrence contributes a feature at all.
/// `tracked` is false for pairs the interaction map does not score. `symmetric`
/// marks pairs whose attacker and victim share a piece type; those are scored
/// once per unordered square pair, keeping the from >= to ordering (compared on
/// physical squares so both perspectives agree).
#[derive(Copy, Clone, Default)]
struct ThreatPair {
    base: i32,
    tracked: bool,
    symmetric: bool,
}

impl ThreatPair {
    /// True when this pair yields no feature for the given physical squares:
    /// the pair is untracked, or it is symmetric and seen in the discarded
    /// square order (we keep from >= to).
    #[inline]
    const fn skip(self, phys_from: u32, phys_to: u32) -> bool {
        !self.tracked || (self.symmetric && (phys_from as u8) < (phys_to as u8))
    }
}

// Static lookup tables — initialised once at startup via init_threats().
//
// Wrapped in OnceLock so that init's writes are visible to reader threads
// with the correct Acquire/Release memory ordering (CLAUDE.md ARM-correctness
// standard). The prior `static mut` form was technically a data race under
// the Rust memory model — safe in practice because init happens on the main
// thread before any helper search thread spawns, but the project standard is
// explicit Acquire/Release on cross-thread shared atomics.
//
// Reads use `unwrap_unchecked` — init_threats() MUST be called from
// `crate::init()` before any helper thread reads from these tables. Violating
// that invariant is undefined behaviour.
struct ThreatTables {
    pairs: [[ThreatPair; NUM_COLORED_PIECES]; NUM_COLORED_PIECES],
    from_offset: [[i32; 64]; NUM_COLORED_PIECES],
    ray_rank: [[[u8; 64]; 64]; NUM_COLORED_PIECES],
    num_features: usize,
}

// Two feature spaces are built at startup and the loaded net selects one:
//   king-attacker ON  = 66,864 features (legacy/current prod nets)
//   king-attacker OFF = 60,144 features (matches SF SFNNv13 full_threats.h
//                       and Hobbes 3.0, both of which exclude the king as a
//                       threat ATTACKER; king-as-victim was never tracked)
// The net header's num_threat_features IS the marker — no extra flag bit.
// Both table sets cost ~50 KB each, so building both is cheaper than any
// scheme that defers construction until a net is known.
static THREAT_TABLES_KING: std::sync::OnceLock<ThreatTables> = std::sync::OnceLock::new();
static THREAT_TABLES_NOKING: std::sync::OnceLock<ThreatTables> = std::sync::OnceLock::new();
/// Active table set. Release-stored at init and on net load, Acquire-loaded
/// by readers (ARM ordering standard) — helper threads must see a fully
/// constructed table set, and net loads happen before search threads spawn.
static ACTIVE_TABLES: std::sync::atomic::AtomicPtr<ThreatTables> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
/// Mirrors the active set's king-attacker flag for the generation-side skips
/// (hot path: avoids dereferencing the table pointer just to read one bool).
static KING_ATTACKER_ON: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Feature counts of the two supported threat spaces.
pub const THREAT_FEATURES_KING: usize = 66864;
pub const THREAT_FEATURES_NOKING: usize = 60144;

/// Point the threat machinery at the feature space a just-loaded net was
/// trained for. Returns Err for any count that is neither space — loading a
/// mismatched net previously SUCCEEDED SILENTLY, indexing weight rows with a
/// different feature space and producing garbage eval with no crash.
pub fn select_feature_space(net_num_threat_features: usize) -> Result<(), String> {
    use std::sync::atomic::Ordering;
    let (cell, king) = match net_num_threat_features {
        THREAT_FEATURES_KING => (&THREAT_TABLES_KING, true),
        THREAT_FEATURES_NOKING => (&THREAT_TABLES_NOKING, false),
        n => {
            return Err(format!(
                "net declares {} threat features; this build supports {} (king-attacker) \
                 or {} (no-king). The count is the feature-space marker — a mismatched \
                 net would index the wrong weight rows and eval garbage.",
                n, THREAT_FEATURES_KING, THREAT_FEATURES_NOKING
            ))
        }
    };
    let ptr = cell.get().expect("init_threats() must run before net load")
        as *const ThreatTables as *mut ThreatTables;
    KING_ATTACKER_ON.store(king, Ordering::Release);
    ACTIVE_TABLES.store(ptr, Ordering::Release);
    Ok(())
}


/// Serialises tests that touch the process-global threat feature space.
///
/// `select_feature_space()` (called on every net load) swaps ACTIVE_TABLES and
/// KING_ATTACKER_ON. Tests that load a real net therefore mutate global state
/// that the threat-consistency tests depend on — and the repo root contains
/// both 66,864 and 60,144 nets, so a parallel net-load can flip the space out
/// from under a fuzz walk that already sized its weights for the other one
/// (observed: "gap-fuzz divergence ... incr=-177 scratch=780").
/// Both sides of that race must hold this lock. `unwrap_or_else(into_inner)`
/// so one genuinely-failing test doesn't poison every other test.
#[cfg(test)]
pub(crate) static FEATURE_SPACE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// True when the active feature space tracks the king as an attacker.
/// Generation-side skips read this to avoid emitting deltas the index
/// mapping would discard anyway.
#[inline(always)]
fn king_attacker_on() -> bool {
    KING_ATTACKER_ON.load(std::sync::atomic::Ordering::Acquire)
}
const FLIPPED_COLORED_PIECE: [usize; NUM_COLORED_PIECES] = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5];

#[inline(always)]
fn flipped_colored_piece(cp: usize) -> usize {
    debug_assert!(cp < NUM_COLORED_PIECES);
    unsafe { *FLIPPED_COLORED_PIECE.get_unchecked(cp) }
}

/// SAFETY: caller must ensure init_threats() has completed before invoking.
#[inline(always)]
fn get_threat_tables() -> &'static ThreatTables {
    // SAFETY: init_threats() is called from `crate::init()` before any
    // helper search threads spawn and publishes a non-null pointer to a
    // 'static OnceLock payload; select_feature_space only ever swaps it for
    // the other 'static payload. Acquire pairs with those Release stores.
    unsafe { &*ACTIVE_TABLES.load(std::sync::atomic::Ordering::Acquire) }
}

/// Get the total threat feature count (call after init_threats).
pub fn num_threat_features() -> usize {
    get_threat_tables().num_features
}

/// Colored piece index: 0=WP, 1=WN, 2=WB, 3=WR, 4=WQ, 5=WK, 6=BP, ..., 11=BK
#[inline]
pub fn colored_piece(color: Color, piece_type: u8) -> usize {
    color as usize * 6 + piece_type as usize
}

/// Piece type from colored piece index.
#[inline]
fn piece_type_of(cp: usize) -> usize {
    cp % 6
}

/// Color from colored piece index.
#[inline]
fn color_of(cp: usize) -> usize {
    cp / 6
}

/// Compute attack bitboard for a colored piece on a square (empty board for init).
fn piece_attacks_empty(cp: usize, sq: u32) -> Bitboard {
    let pt = piece_type_of(cp);
    match pt {
        0 => pawn_attacks(if color_of(cp) == 0 { WHITE } else { BLACK }, sq),
        1 => knight_attacks(sq),
        2 => bishop_attacks_empty(sq), // empty board
        3 => rook_attacks_empty(sq),
        4 => bishop_attacks_empty(sq) | rook_attacks_empty(sq),
        5 => king_attacks(sq),
        _ => 0,
    }
}

/// Compute attack bitboard for a piece type on a square with given occupancy.
///
/// `#[inline]` so call sites with constrained piece_type (e.g. slider-only
/// loops) can specialize away the 6-way switch dispatch under LTO. The
/// dispatch (jump-table address calc) measured 10.69% inside this function,
/// ~0.3% of total NPS.
#[inline]
pub fn piece_attacks_occ(piece_type: u8, color: Color, sq: u32, occ: Bitboard) -> Bitboard {
    match piece_type {
        PAWN => pawn_attacks(color, sq),
        KNIGHT => knight_attacks(sq),
        BISHOP => bishop_attacks(sq, occ),
        ROOK => rook_attacks(sq, occ),
        QUEEN => queen_attacks(sq, occ),
        KING => king_attacks(sq),
        _ => 0,
    }
}

/// Initialise threat feature lookup tables. Must be called at startup
/// before any helper search thread spawns.
pub fn init_threats() {
    use std::sync::atomic::Ordering;
    let king = build_threat_tables(true);
    let noking = build_threat_tables(false);
    let n_king = king.num_features;
    let n_noking = noking.num_features;
    let _ = THREAT_TABLES_KING.set(king);
    let _ = THREAT_TABLES_NOKING.set(noking);
    debug_assert_eq!(n_king, THREAT_FEATURES_KING);
    debug_assert_eq!(n_noking, THREAT_FEATURES_NOKING);
    // Default to the king-attacker space (current prod nets); a net load
    // re-points this via select_feature_space.
    let ptr = THREAT_TABLES_KING.get().expect("just set")
        as *const ThreatTables as *mut ThreatTables;
    KING_ATTACKER_ON.store(true, Ordering::Release);
    ACTIVE_TABLES.store(ptr, Ordering::Release);
    eprintln!("Threat features initialised: {} (king-attacker) / {} (no-king)",
        n_king, n_noking);
}

/// Build one feature space. `king_attacker` selects whether the king is
/// tracked as a threat attacker; it zeroes both the king's interaction-map
/// row and its per-attacker target count, which is what drops the derived
/// total from 66,864 to 60,144.
fn build_threat_tables(king_attacker: bool) -> ThreatTables {
    let mut pairs = [[ThreatPair::default(); NUM_COLORED_PIECES]; NUM_COLORED_PIECES];
    let mut from_offset = [[0i32; 64]; NUM_COLORED_PIECES];
    let mut ray_rank = [[[0u8; 64]; 64]; NUM_COLORED_PIECES];

    // Per colored piece: `slots` = number of (from-square, attack-target)
    // slots it can occupy; `block_base` = first feature index of the piece's
    // block. Features are attacker-major; `from_offset[cp][sq]` accumulates the
    // slot count for from-squares below `sq`.
    let mut slots = [0i32; NUM_COLORED_PIECES];
    let mut block_base = [0i32; NUM_COLORED_PIECES];
    let mut next_base: i32 = 0;

    for color in 0..2usize {
        for pt in 0..6usize {
            let cp = color * 6 + pt;
            let mut count: i32 = 0;
            for sq in 0..64u32 {
                from_offset[cp][sq as usize] = count;
                // Pawns never sit on ranks 1/8, so contribute no slots there.
                if pt == 0 && !(8..56).contains(&sq) {
                    continue;
                }
                count += popcount(piece_attacks_empty(cp, sq)) as i32;
            }
            slots[cp] = count;
            block_base[cp] = next_base;
            let targets = if pt == 5 && !king_attacker { 0 } else { PIECE_TARGET_COUNT[pt] };
            next_base += targets * count;
        }
    }
    let num_features = next_base as usize;

    // Per (attacker, victim) pair: block base + skip flags. Within an attacker
    // block, features group by victim color, then by interaction-map slot, each
    // group spanning `slots[attacker]` entries.
    for att in 0..NUM_COLORED_PIECES {
        let att_pt = piece_type_of(att);
        let att_color = color_of(att);
        for vic in 0..NUM_COLORED_PIECES {
            let vic_pt = piece_type_of(vic);
            let vic_color = color_of(vic);

            let map = if att_pt == 5 && !king_attacker { -1 } else { PIECE_INTERACTION_MAP[att_pt][vic_pt] };
            let tracked = map >= 0;
            // Same piece-type pairs are symmetric — except same-color pawns.
            let symmetric = att_pt == vic_pt && (att_color != vic_color || att_pt != 0);
            // map is < 0 only for untracked pairs, whose base is never read.
            let base = block_base[att]
                + (vic_color as i32 * (PIECE_TARGET_COUNT[att_pt] / 2) + map.max(0))
                    * slots[att];

            pairs[att][vic] = ThreatPair { base, tracked, symmetric };
        }
    }

    // Per attacker + from-square: rank of each to-square within the ray order
    // (how many of the attacker's target squares precede it).
    for cp in 0..NUM_COLORED_PIECES {
        for from in 0..64u32 {
            let attacks = piece_attacks_empty(cp, from);
            for to in 0..64u32 {
                let below = if to > 0 { (1u64 << to) - 1 } else { 0 };
                ray_rank[cp][from as usize][to as usize] = popcount(below & attacks) as u8;
            }
        }
    }

    ThreatTables { pairs, from_offset, ray_rank, num_features }
}

/// Compute a single threat feature index.
///
/// Returns negative if this pair is excluded (should be skipped).
/// `pov` is the perspective (WHITE or BLACK).
/// `mirrored` is true when the perspective king is on files e-h.
#[inline]
pub fn threat_index(
    attacker_cp: usize, // colored piece index of attacker
    from: u32,          // attacker square (physical, pre-flip)
    victim_cp: usize,   // colored piece index of victim
    to: u32,            // victim square (physical, pre-flip)
    mirrored: bool,
    pov: Color,
) -> i32 {
    // Colored-piece indices are stored white-relative; remap to POV.
    let attacker = if pov == BLACK { (attacker_cp + 6) % 12 } else { attacker_cp };
    let victim = if pov == BLACK { (victim_cp + 6) % 12 } else { victim_cp };

    let tables = get_threat_tables();
    let pair = tables.pairs[attacker][victim];
    // The symmetric-pair tie-break is decided on PHYSICAL squares (pre-flip) so
    // both perspectives make the same choice — matching how the net was trained.
    // (A perspective-flipped tie-break here previously caused a ~3-5% NTM
    // feature mismatch vs training.)
    if pair.skip(from, to) {
        return -1;
    }

    // The perspective flip applies only to the square-indexed tables below.
    let flip = (7 * mirrored as u32) ^ (56 * pov as u32);
    let from_f = (from ^ flip) as usize;
    let to_f = (to ^ flip) as usize;

    pair.base
        + tables.from_offset[attacker][from_f]
        + tables.ray_rank[attacker][from_f][to_f] as i32
}

/// Enumerate all threat features active in a position.
/// Calls `callback(feature_index)` for each active threat.
pub fn enumerate_threats<F: FnMut(usize)>(
    pieces_bb: &[Bitboard; 6],  // by piece type
    colors_bb: &[Bitboard; 2],  // by color
    mailbox: &[u8; 64],         // square → piece type (NO_PIECE_TYPE for empty)
    occ: Bitboard,
    pov: Color,
    mirrored: bool,
    mut callback: F,
) {
    let white_bb = colors_bb[WHITE as usize];

    for color in [WHITE, BLACK] {
        for pt in 0..6u8 {
            let mut piece_bb = pieces_bb[pt as usize] & colors_bb[color as usize];
            let cp = colored_piece(color, pt);

            while piece_bb != 0 {
                let sq = piece_bb.trailing_zeros();
                piece_bb &= piece_bb - 1; // clear LSB

                // Compute attacks for this piece (with occupancy for sliders)
                let attacks = piece_attacks_occ(pt, color, sq, occ);

                // Find attacked occupied squares (direct threats)
                let mut attacked_occ = attacks & occ;
                while attacked_occ != 0 {
                    let target_sq = attacked_occ.trailing_zeros();
                    attacked_occ &= attacked_occ - 1;

                    let victim_pt = mailbox[target_sq as usize];
                    if victim_pt >= 6 { continue; }
                    let victim_color = if white_bb & (1u64 << target_sq) != 0 { WHITE } else { BLACK };
                    let victim_cp = colored_piece(victim_color, victim_pt);

                    let idx = threat_index(cp, sq, victim_cp, target_sq, mirrored, pov);
                    if idx >= 0 {
                        callback(idx as usize);
                    }
                }

            }
        }
    }
}

/// HISTORICAL reference — an OLD trainer-side enumeration that evaluated the
/// same-type-pair skip in the wrong frame. DOES NOT describe current training,
/// and is retained only to characterise that bug.
///
/// It applied the same-type-pair skip (`sq < to`) in the trainer's internal
/// bf-frame (rank-flipped when the real STM is black), which disagrees with
/// Coda's physical-frame inference for black-STM positions. The trainer now
/// does the skip on physical squares (`phys_flip`), matching Coda exactly —
/// see `enumerate_threats_bullet_postfix_ref` below, which
/// `fuzz-threats --postfix` verifies at **0 mismatches** over 40000 evals in
/// both STMs. There is no live train/inference divergence.
///
/// Note: the physical-frame skip both sides now use is STM-invariant (so
/// Coda's incremental threat deltas stay clean) but is NOT mirror-symmetric
/// — a deliberate tradeoff. It is the source of a small, consistent
/// color-eval asymmetry on same-type-pair (bishop/rook) threats. That is a
/// feature-design property, not a divergence.
///
/// Only direct-attack features are handled here (no x-ray).
pub fn enumerate_threats_bullet_ref<F: FnMut(usize)>(
    pieces_bb: &[Bitboard; 6],
    colors_bb: &[Bitboard; 2],
    mailbox: &[u8; 64],
    occ: Bitboard,
    pov: Color,
    mirrored: bool,
    real_stm: Color,
    mut callback: F,
) {
    let white_bb = colors_bb[WHITE as usize];
    let bf_flip: u32 = if real_stm == BLACK { 56 } else { 0 };

    for color in [WHITE, BLACK] {
        for pt in 0..6u8 {
            let mut piece_bb = pieces_bb[pt as usize] & colors_bb[color as usize];
            let cp = colored_piece(color, pt);
            while piece_bb != 0 {
                let sq = piece_bb.trailing_zeros();
                piece_bb &= piece_bb - 1;
                let sq_bf = sq ^ bf_flip;

                let attacks = piece_attacks_occ(pt, color, sq, occ);
                let mut hits = attacks & occ;
                while hits != 0 {
                    let to = hits.trailing_zeros();
                    hits &= hits - 1;
                    let to_bf = to ^ bf_flip;

                    let victim_pt = mailbox[to as usize];
                    if victim_pt >= 6 { continue; }
                    let victim_color = if white_bb & (1u64 << to) != 0 { WHITE } else { BLACK };
                    let victim_cp = colored_piece(victim_color, victim_pt);

                    let idx = threat_index_bullet_ref(
                        cp, sq, sq_bf,
                        victim_cp, to, to_bf,
                        mirrored, pov,
                    );
                    if idx >= 0 {
                        callback(idx as usize);
                    }
                }

            }
        }
    }
}

/// Index computation with Bullet's board-frame symmetric-pair tie-break.
/// Identical to `threat_index` except the tie-break is decided on the
/// board-frame (bf) squares rather than physical ones.
#[inline]
fn threat_index_bullet_ref(
    attacker_cp: usize,
    from_phys: u32,
    from_bf: u32,
    victim_cp: usize,
    to_phys: u32,
    to_bf: u32,
    mirrored: bool,
    pov: Color,
) -> i32 {
    let attacker = if pov == BLACK { (attacker_cp + 6) % 12 } else { attacker_cp };
    let victim = if pov == BLACK { (victim_cp + 6) % 12 } else { victim_cp };

    let tables = get_threat_tables();
    let pair = tables.pairs[attacker][victim];
    // The one difference vs threat_index: the symmetric tie-break uses the
    // board-frame (bf) squares rather than the physical squares.
    if pair.skip(from_bf, to_bf) {
        return -1;
    }
    let flip = (7 * mirrored as u32) ^ (56 * pov as u32);
    let from_f = (from_phys ^ flip) as usize;
    let to_f = (to_phys ^ flip) as usize;
    pair.base
        + tables.from_offset[attacker][from_f]
        + tables.ray_rank[attacker][from_f][to_f] as i32
}

/// Byte-exact match of Bullet's post-C8fix-2 `map_features` (required for net interop)
/// (`bullet/crates/bullet_lib/src/game/inputs/chess_threats.rs` lines 374-578,
/// commit 62931d1 + a8e2c7d).
///
/// Bullet operates in **bf-frame** internally (bulletformat rank-flips
/// bitboards and color-swaps when real STM = Black). It compensates for
/// the resulting non-STM-invariance of the bf-frame `sq < to` semi-excl by
/// using `phys_flip = 56 if real_stm = Black else 0` to convert bf squares
/// back to physical for the comparison. We replicate that walk here so the
/// fuzzer can verify Coda inference (physical-frame) and post-fix Bullet
/// training agree on every position.
///
/// Expected fuzzer result vs `enumerate_threats`: **0 mismatches**. Any
/// nonzero result indicates a residual training/inference divergence.
pub fn enumerate_threats_bullet_postfix_ref<F: FnMut(usize)>(
    pieces_bb: &[Bitboard; 6],
    colors_bb: &[Bitboard; 2],
    mailbox: &[u8; 64],
    occ: Bitboard,
    pov: Color,
    mirrored: bool,
    real_stm: Color,
    mut callback: F,
) {
    let _ = occ;
    let phys_flip: u32 = if real_stm == BLACK { 56 } else { 0 };
    let bf_flip = phys_flip;

    // Build bf-frame view: rank-flip + color-swap if real_stm = Black, so
    // bf_colors[0] = real-STM pieces (matching bulletformat's convention).
    let mut bf_pieces = [0u64; 6];
    let mut bf_colors = [0u64; 2];
    let mut bf_mailbox = [0xFFu8; 64];
    if bf_flip == 0 {
        bf_pieces.copy_from_slice(pieces_bb);
        bf_colors.copy_from_slice(colors_bb);
        bf_mailbox.copy_from_slice(mailbox);
    } else {
        for pt in 0..6 {
            bf_pieces[pt] = pieces_bb[pt].swap_bytes();
        }
        bf_colors[0] = colors_bb[real_stm as usize].swap_bytes();
        bf_colors[1] = colors_bb[(1 - real_stm) as usize].swap_bytes();
        for sq in 0..64 {
            bf_mailbox[sq] = mailbox[sq ^ 56];
        }
    }
    let bf_occ = bf_colors[0] | bf_colors[1];

    for bf_color in [0u8, 1u8] {
        let real_color: Color = if bf_color == 0 { real_stm } else { 1 - real_stm };
        let pov_for_attack: Color = bf_color;

        for pt in 0..6u8 {
            let mut piece_bb = bf_pieces[pt as usize] & bf_colors[bf_color as usize];
            let cp = colored_piece(real_color, pt);

            while piece_bb != 0 {
                let sq_bf = piece_bb.trailing_zeros();
                piece_bb &= piece_bb - 1;
                let sq_phys = sq_bf ^ bf_flip;

                let attacks = piece_attacks_occ(pt, pov_for_attack, sq_bf, bf_occ);

                let mut hits = attacks & bf_occ;
                while hits != 0 {
                    let to_bf = hits.trailing_zeros();
                    hits &= hits - 1;
                    let to_phys = to_bf ^ bf_flip;

                    let victim_pt = bf_mailbox[to_bf as usize];
                    if victim_pt >= 6 { continue; }
                    let victim_bf_color: u8 =
                        if (bf_colors[0] >> to_bf) & 1 != 0 { 0 } else { 1 };
                    let victim_real_color: Color =
                        if victim_bf_color == 0 { real_stm } else { 1 - real_stm };
                    let victim_cp = colored_piece(victim_real_color, victim_pt);

                    // Post-fix semi-excl: physical-frame, via phys_flip.
                    let is_pawn = pt == PAWN;
                    let same_type = pt == victim_pt;
                    let semi_excl_pair =
                        same_type && (bf_color != victim_bf_color || !is_pawn);
                    if semi_excl_pair && (sq_bf ^ phys_flip) < (to_bf ^ phys_flip) {
                        continue;
                    }

                    let idx = threat_index(cp, sq_phys, victim_cp, to_phys, mirrored, pov);
                    if idx >= 0 {
                        callback(idx as usize);
                    }
                }

            }
        }
    }
}

/// Maximum threat deltas per ply.
pub const MAX_THREAT_DELTAS: usize = 128;

/// Packed threat delta. Purely internal to the incremental threat accumulator
/// (built pre-move, consumed to add/sub feature rows) — never serialised, so the
/// bit layout is Coda's own. The two colored pieces are 0..11 (4 bits each) and
/// the two squares 0..63 (6 bits each), packed low-to-high with the add/sub flag
/// on top: [attacker_cp:4][victim_cp:4][from_sq:6][to_sq:6][add:1].
#[derive(Copy, Clone)]
pub struct RawThreatDelta(u32);

impl RawThreatDelta {
    #[inline(always)]
    pub const fn new(attacker_cp: u8, from_sq: u8, victim_cp: u8, to_sq: u8, add: bool) -> Self {
        Self((attacker_cp as u32 & 0xF)
            | ((victim_cp as u32 & 0xF) << 4)
            | ((from_sq as u32 & 0x3F) << 8)
            | ((to_sq as u32 & 0x3F) << 14)
            | ((add as u32) << 20))
    }

    pub const ZERO: Self = Self(0);

    #[inline(always)] pub fn attacker_cp(self) -> u8 { (self.0 & 0xF) as u8 }
    #[inline(always)] pub fn victim_cp(self) -> u8 { ((self.0 >> 4) & 0xF) as u8 }
    #[inline(always)] pub fn from_sq(self) -> u8 { ((self.0 >> 8) & 0x3F) as u8 }
    #[inline(always)] pub fn to_sq(self) -> u8 { ((self.0 >> 14) & 0x3F) as u8 }
    #[inline(always)] pub fn add(self) -> bool { (self.0 >> 20) & 1 != 0 }
}

/// Compute raw threat deltas when a piece moves from `from` to `to`.
/// Must be called BEFORE the move is applied on the board (board still has old state).
/// `occ_without_dest` = occupancy with `from` removed but `to` not yet occupied.
pub fn push_threats_on_move(
    deltas: &mut Vec<RawThreatDelta>,
    pieces_bb: &[Bitboard; 6],
    colors_bb: &[Bitboard; 2],
    mailbox: &[u8; 64],
    occ: Bitboard,
    piece_color: Color,
    piece_type: u8,
    from: u32,
    to: u32,
) {
    let white_bb = colors_bb[WHITE as usize];
    let cp = colored_piece(piece_color, piece_type);
    // Use occupancy with the moving piece removed from `from` but not yet at `to`
    // Transit occupancy for the moving piece: occ ^ to_bb
    let occ_transit = occ ^ (1u64 << to);

    // Remove threats from old square
    push_threats_for_piece(deltas, pieces_bb, colors_bb, mailbox, occ_transit, white_bb, cp, piece_color, piece_type, from, false);
    // Add threats at new square
    push_threats_for_piece(deltas, pieces_bb, colors_bb, mailbox, occ_transit, white_bb, cp, piece_color, piece_type, to, true);
}

/// The slice of `Board` that threat generation reads: piece bitboards, colour
/// bitboards, mailbox. 128 bytes, `Copy`, no hashes — deliberately NOT a Board,
/// so a scratch replay can never disturb the live search state (the NNUE
/// accumulator stack and the Zobrist keys both alias the real board).
#[derive(Clone, Copy)]
pub struct PieceState {
    pub pieces: [Bitboard; 6],
    pub colors: [Bitboard; 2],
    pub mailbox: [u8; 64],
}

impl PieceState {
    #[inline]
    pub fn from_board(b: &crate::board::Board) -> Self {
        Self { pieces: b.pieces, colors: b.colors, mailbox: b.mailbox }
    }

    #[inline]
    pub fn occ(&self) -> Bitboard { self.colors[0] | self.colors[1] }

    #[inline]
    fn remove(&mut self, color: Color, pt: u8, sq: u8) {
        let bb = 1u64 << sq;
        self.pieces[pt as usize] ^= bb;
        self.colors[color as usize] ^= bb;
        self.mailbox[sq as usize] = NO_PIECE_TYPE;
    }

    #[inline]
    fn put(&mut self, color: Color, pt: u8, sq: u8) {
        let bb = 1u64 << sq;
        self.pieces[pt as usize] |= bb;
        self.colors[color as usize] |= bb;
        self.mailbox[sq as usize] = pt;
    }

    #[inline]
    fn shift(&mut self, color: Color, pt: u8, from: u8, to: u8) {
        let from_to = (1u64 << from) | (1u64 << to);
        self.pieces[pt as usize] ^= from_to;
        self.colors[color as usize] ^= from_to;
        self.mailbox[from as usize] = NO_PIECE_TYPE;
        self.mailbox[to as usize] = pt;
    }
}

/// Castling rook squares for `us`, given the king's `from`/`to`.
#[inline]
fn castle_rook_squares(us: Color, from: u8, to: u8) -> (u8, u8) {
    if to > from {
        if us == WHITE { (7, 5) } else { (63, 61) }
    } else if us == WHITE { (0, 3) } else { (56, 59) }
}

/// Regenerate the threat deltas for `mv` from the PRE-move piece state,
/// advancing `st` to the post-move state as a side effect.
///
/// This mirrors the mutation-and-emit sequence inside `Board::make_move`
/// exactly, and it has to: each emit call observes a DIFFERENT intermediate
/// state, so reordering them changes the deltas. In particular the promotion
/// case applies BOTH mutations before either emit, so both emits see the
/// post-promotion board.
///
/// The duplication of that sequence is deliberate — the alternative was routing
/// `make_move`'s own mutations through this type, which would have put a scratch
/// abstraction in the hottest path in the engine. The cost of duplicating is
/// drift, and `lazy_deltas_match_eager_generation` is the guard against it:
/// it walks the fuzz corpus asserting this function reproduces `make_move`'s
/// deltas exactly, so the two cannot silently diverge.
pub fn replay_move_deltas(
    st: &mut PieceState,
    us: Color,
    mv: Move,
    captured: u8,
    out: &mut Vec<RawThreatDelta>,
) {
    out.clear();
    let from = move_from(mv);
    let to = move_to(mv);
    let flags = move_flags(mv);
    let them = flip_color(us);
    let pt = st.mailbox[from as usize];

    if flags == FLAG_EN_PASSANT {
        let cap_sq = if us == WHITE { to.wrapping_sub(8) } else { to.wrapping_add(8) };
        st.remove(them, PAWN, cap_sq);
        push_threats_on_change(out, &st.pieces, &st.colors, &st.mailbox,
                               st.occ(), them, PAWN, cap_sq as u32, false);
    } else if captured != NO_PIECE_TYPE {
        st.remove(them, captured, to);
        push_threats_on_change(out, &st.pieces, &st.colors, &st.mailbox,
                               st.occ(), them, captured, to as u32, false);
    }

    st.shift(us, pt, from, to);
    push_threats_on_move(out, &st.pieces, &st.colors, &st.mailbox,
                         st.occ(), us, pt, from as u32, to as u32);

    if is_promotion(mv) {
        let promo_pt = promotion_piece_type(mv);
        st.remove(us, pt, to);
        st.put(us, promo_pt, to);
        push_threats_on_change(out, &st.pieces, &st.colors, &st.mailbox,
                               st.occ(), us, pt, to as u32, false);
        push_threats_on_change(out, &st.pieces, &st.colors, &st.mailbox,
                               st.occ(), us, promo_pt, to as u32, true);
    }

    if flags == FLAG_CASTLE {
        let (rook_from, rook_to) = castle_rook_squares(us, from, to);
        st.shift(us, ROOK, rook_from, rook_to);
        push_threats_on_move(out, &st.pieces, &st.colors, &st.mailbox,
                             st.occ(), us, ROOK, rook_from as u32, rook_to as u32);
    }
}

/// Step `st` BACKWARDS over `mv`, turning a post-move piece state into the
/// pre-move one. Undoes in the reverse of `replay_move_deltas`'s order.
///
/// This is what makes lazy generation possible without snapshotting boards:
/// `UndoInfo` already carries `mv` and `captured` for every ply on the current
/// path, so any ancestor's piece state can be recovered by walking back from
/// the live board.
pub fn undo_move_state(st: &mut PieceState, us: Color, mv: Move, captured: u8) {
    let from = move_from(mv);
    let to = move_to(mv);
    let flags = move_flags(mv);
    let them = flip_color(us);

    if flags == FLAG_CASTLE {
        let (rook_from, rook_to) = castle_rook_squares(us, from, to);
        st.shift(us, ROOK, rook_to, rook_from);
    }

    if is_promotion(mv) {
        let promo_pt = promotion_piece_type(mv);
        st.remove(us, promo_pt, to);
        st.put(us, PAWN, to);
    }

    // Read AFTER undoing promotion, so this is the piece that originally moved.
    let pt = st.mailbox[to as usize];
    st.shift(us, pt, to, from);

    if flags == FLAG_EN_PASSANT {
        let cap_sq = if us == WHITE { to.wrapping_sub(8) } else { to.wrapping_add(8) };
        st.put(them, PAWN, cap_sq);
    } else if captured != NO_PIECE_TYPE {
        st.put(them, captured, to);
    }
}

/// Compute raw threat deltas when a piece appears or disappears.
pub fn push_threats_on_change(
    deltas: &mut Vec<RawThreatDelta>,
    pieces_bb: &[Bitboard; 6],
    colors_bb: &[Bitboard; 2],
    mailbox: &[u8; 64],
    occ: Bitboard,
    piece_color: Color,
    piece_type: u8,
    square: u32,
    add: bool,
) {
    let white_bb = colors_bb[WHITE as usize];
    let cp = colored_piece(piece_color, piece_type);
    push_threats_for_piece(deltas, pieces_bb, colors_bb, mailbox, occ, white_bb, cp, piece_color, piece_type, square, add);
}

/// Ablation flag (CODA_NO_SLIDER_SEES=1): skip emitting step-2 "slider-sees"
/// threat deltas (incoming slider attacks on the square). NOT bit-identical —
/// the net was trained with these features, so eval is wrong/weaker; this is a
/// fast NPS-impact test of the feature type's cost. Cached once.
#[inline]
fn skip_slider_sees() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("CODA_NO_SLIDER_SEES").is_ok())
}

/// Core: compute all threat deltas for a piece on a square.
/// Independent implementation; the three-step structure is:
/// 1. Threats FROM this piece to occupied squares
/// 2. Sliders that see this square + x-ray targets behind it
/// 3. Non-sliders (pawns, knights, kings) that attack this square
fn push_threats_for_piece(
    deltas: &mut Vec<RawThreatDelta>,
    pieces_bb: &[Bitboard; 6],
    colors_bb: &[Bitboard; 2],
    mailbox: &[u8; 64],
    occ: Bitboard,
    white_bb: Bitboard,
    cp: usize,
    piece_color: Color,
    piece_type: u8,
    square: u32,
    add: bool,
) {
    #[cfg(feature = "profile-threats")]
    let fn_start_tsc = crate::threats::thr_stats::rdtsc();
    #[cfg(feature = "profile-threats")]
    crate::threats::thr_stats::record_call();

    // 1. Threats FROM this piece to occupied squares
    #[cfg(feature = "profile-threats")]
    let s1_start = crate::threats::thr_stats::rdtsc();
    #[cfg(feature = "profile-threats")]
    let s1_deltas_before = deltas.len() as u64;

    // No-king space: the king emits no attacker-side features, so skip the
    // enumeration rather than emitting deltas the index mapping discards.
    let skip_king_attacks = piece_type == 5 && !king_attacker_on();
    let my_attacks = if skip_king_attacks {
        0
    } else {
        piece_attacks_occ(piece_type, piece_color, square, occ)
    };
    let mut attacked_occ = my_attacks & occ;
    while attacked_occ != 0 {
        let target_sq = attacked_occ.trailing_zeros();
        attacked_occ &= attacked_occ - 1;
        let victim_pt = mailbox[target_sq as usize];
        if victim_pt >= 6 { continue; }
        let victim_color = if white_bb & (1u64 << target_sq) != 0 { WHITE } else { BLACK };
        deltas.push(RawThreatDelta::new(cp as u8, square as u8, colored_piece(victim_color, victim_pt) as u8, target_sq as u8, add));
    }

    #[cfg(feature = "profile-threats")]
    crate::threats::thr_stats::record_section(
        0,
        crate::threats::thr_stats::rdtsc().wrapping_sub(s1_start),
        deltas.len() as u64 - s1_deltas_before,
    );

    #[cfg(feature = "profile-threats")]
    let s1b_start = crate::threats::thr_stats::rdtsc();
    #[cfg(feature = "profile-threats")]
    let s1b_deltas_before = deltas.len() as u64;
    #[cfg(feature = "profile-threats")]
    let mut own_xray_nonslider = 0u64;
    #[cfg(feature = "profile-threats")]
    let mut own_xray_no_direct = 0u64;
    #[cfg(feature = "profile-threats")]
    let mut own_xray_blockers = 0u64;
    #[cfg(feature = "profile-threats")]
    let mut own_xray_no_behind = 0u64;
    #[cfg(feature = "profile-threats")]
    let mut own_xray_invalid = 0u64;
    #[cfg(feature = "profile-threats")]
    let mut own_xray_emits = 0u64;

    #[cfg(feature = "profile-threats")]
    crate::threats::thr_stats::record_section(
        1,
        crate::threats::thr_stats::rdtsc().wrapping_sub(s1b_start),
        deltas.len() as u64 - s1b_deltas_before,
    );
    #[cfg(feature = "profile-threats")]
    crate::threats::thr_stats::record_own_xray_reasons(
        own_xray_nonslider,
        own_xray_no_direct,
        own_xray_blockers,
        own_xray_no_behind,
        own_xray_invalid,
        own_xray_emits,
    );

    // 2. Sliding pieces that see this square
    // Compute rook/bishop attacks FROM this square to find which sliders can reach it
    #[cfg(feature = "profile-threats")]
    let s2_start = crate::threats::thr_stats::rdtsc();
    #[cfg(feature = "profile-threats")]
    let s2_deltas_before = deltas.len() as u64;
    let rook_att = rook_attacks(square, occ);
    let bishop_att = bishop_attacks(square, occ);

    let diagonal_sliders = (pieces_bb[BISHOP as usize] | pieces_bb[QUEEN as usize]) & bishop_att;
    let orthogonal_sliders = (pieces_bb[ROOK as usize] | pieces_bb[QUEEN as usize]) & rook_att;

    // Z-finding cull (analogous to 2b's cull). The Z-level x-ray delta
    // inside the slider loop below needs a chain S→square→Y→Z where Y is
    // the first occupant past `square` on the slider's ray and Z is the
    // first occupant past Y. If NO ray from `square` has 2+ occupants,
    // no slider on any ray can produce a Z delta — skip the whole
    // Z-finding block and just emit the direct threat.
    //
    // Shared with 2b's cull below (lines ~625+): `rays_from_sq_empty`
    // is computed once here and reused. Cost: 2 magic lookups + 4
    // bitwise ops. Savings per skipped slider: 2 magic lookups.
    // Break-even at 1 skipped slider.

    let emit_slider_sees = !skip_slider_sees();
    let mut sliders = (diagonal_sliders | orthogonal_sliders) & occ;
    while sliders != 0 {
        let slider_sq = sliders.trailing_zeros();
        sliders &= sliders - 1;
        let slider_pt = mailbox[slider_sq as usize];
        if slider_pt >= 6 { continue; }
        let slider_color = if white_bb & (1u64 << slider_sq) != 0 { WHITE } else { BLACK };
        let slider_cp = colored_piece(slider_color, slider_pt);

        // X-ray book-keeping when the piece at `square` appears/disappears:
        //
        //   Let Y = first piece past `square` on the slider's ray.
        //   Let Z = piece past Y on the same ray (if any).
        //
        // When the piece at `square` appears, the slider's feature for Y is
        // unchanged — was a direct threat, becomes an x-ray to the same index.
        // The slider's feature for Z (previously x-ray through Y) is LOST
        // because x-ray depth is only 1 past the first blocker. When the
        // piece at `square` disappears, Z is GAINED. The Y feature is
        // unchanged in both directions.
        //
        // The slider's direct attack on `square` itself is emitted below.
        // This block emits only the Z-level delta.
        {
            // X-ray OFF: the slider's direct attack on the first piece past
            // `square` (Y) is blocked/unblocked as `square` appears/vanishes.
            // With x-ray ON that direct feature is PRESERVED as an x-ray to the
            // same index (Y "unchanged", handled by the block above). With x-ray
            // OFF there is no x-ray to preserve it, so the Y-level direct feature
            // must be removed/added here. Gating this block on x-ray leaves the
            // incremental accumulator over-counting Y for --xray 0 nets — good
            // static eval (full recompute) but broken search play.
            // Regression: fuzz_random_walk_xray_off.
            let y_candidates = crate::bitboard::ray_extension(slider_sq, square) & occ;
            if y_candidates != 0 {
                let y_sq = if slider_sq < square {
                    y_candidates.trailing_zeros()
                } else {
                    63 - y_candidates.leading_zeros()
                };
                let ypt = mailbox[y_sq as usize];
                if ypt < 6 {
                    let ycolor = if white_bb & (1u64 << y_sq) != 0 { WHITE } else { BLACK };
                    deltas.push(RawThreatDelta::new(
                        slider_cp as u8, slider_sq as u8,
                        colored_piece(ycolor, ypt) as u8, y_sq as u8,
                        !add,
                    ));
                }
            }
        }

        // The slider itself attacks/no longer attacks this square
        if emit_slider_sees {
            deltas.push(RawThreatDelta::new(slider_cp as u8, slider_sq as u8, cp as u8, square as u8, add));
        }
    }

    #[cfg(feature = "profile-threats")]
    crate::threats::thr_stats::record_section(
        2,
        crate::threats::thr_stats::rdtsc().wrapping_sub(s2_start),
        deltas.len() as u64 - s2_deltas_before,
    );

    // 2b. Sliders whose X-RAY target is `square` (through one blocker Y
    // between the slider and `square`). When the piece at `square`
    // appears/disappears, such a slider's x-ray feature changes from
    // (S, Y) direct + (S, sq) x-ray ↔ (S, Y) direct + (S, next_past_sq) x-ray.
    #[cfg(feature = "profile-threats")]
    let s2b_start = crate::threats::thr_stats::rdtsc();
    #[cfg(feature = "profile-threats")]
    let s2b_deltas_before = deltas.len() as u64;
    //
    // The (S, Y) direct feature is unchanged. The (S, cp, sq) x-ray
    // feature appears/disappears with `add`. Any piece beyond sq on the
    // same ray that was/would-be the x-ray target in the other state is
    // handled separately — only the sq-itself delta goes here.
    //
    // Implementation: iterate sliders on empty-board rays from `sq` and
    // test `between(S, sq) & occ` for exactly one blocker Y. This replaces
    // the previous 8-direction scalar ray walks with a slider iteration
    // driven by the precomputed `between()` table. Per-call work now
    // scales with number of sliders on aligned rays (typically 0-4),
    // not fixed at 8 directions.
    //
    // Correctness: sliders with 0 blockers between them and sq are direct
    // attackers handled by section 2; sliders with 2+ blockers are 2+
    // level x-rays not encoded in the feature set (skip). Exactly-one-
    // blocker is the 2b case.
    //
    // Set membership guarantees piece-type match: rook/queen on ortho ray,
    // bishop/queen on diag ray. A queen cannot simultaneously be on both
    // an ortho and diag ray from the same sq (disjoint ray directions).
    //
    // `ortho_ray_mask` and `diag_ray_mask` are computed above (section 2's
    // Z-finding cull) — reused here to avoid two magic bitboard lookups.
    //
    // `& occ` filters out phantom candidates during push_threats_on_move:
    // the moved piece is in pieces_bb at `to`, but `occ_transit = occ ^ (1<<to)`
    // has `to` cleared. Without this mask, a moved slider would be iterated
    // as an x-ray candidate for its own source square and emit a spurious
    // 2b delta. Section 2 applies the same filter (`sliders & occ`).
    // 2b is X-ray (slider-through-blocker); gate on the X-ray emission flag.
    #[cfg(feature = "profile-threats")]
    let mut s2b_no_candidates = 0u64;
    #[cfg(feature = "profile-threats")]
    let mut s2b_candidates = 0u64;
    #[cfg(feature = "profile-threats")]
    let mut s2b_blockers_zero = 0u64;
    #[cfg(feature = "profile-threats")]
    let mut s2b_blockers_multi = 0u64;
    #[cfg(feature = "profile-threats")]
    let mut s2b_exact_one = 0u64;
    #[cfg(feature = "profile-threats")]
    let mut s2b_invalid = 0u64;
    #[cfg(feature = "profile-threats")]
    let mut s2b_sq_emits = 0u64;
    #[cfg(feature = "profile-threats")]
    let mut s2b_no_w = 0u64;
    // This local's feeding code went with the splat/x-ray cleanup, but
    // record_s2b_reasons still takes it; keep it declared (always 0) so the
    // profile-threats feature compiles.
    #[cfg(feature = "profile-threats")]
    let s2b_w_emits = 0u64;


    #[cfg(feature = "profile-threats")]
    crate::threats::thr_stats::record_section(
        3,
        crate::threats::thr_stats::rdtsc().wrapping_sub(s2b_start),
        deltas.len() as u64 - s2b_deltas_before,
    );
    #[cfg(feature = "profile-threats")]
    crate::threats::thr_stats::record_s2b_reasons(
        s2b_no_candidates,
        s2b_candidates,
        s2b_blockers_zero,
        s2b_blockers_multi,
        s2b_exact_one,
        s2b_invalid,
        s2b_sq_emits,
        s2b_no_w,
        s2b_w_emits,
    );

    #[cfg(feature = "profile-threats")]
    let s3_start = crate::threats::thr_stats::rdtsc();
    #[cfg(feature = "profile-threats")]
    let s3_deltas_before = deltas.len() as u64;

    // 3. Non-sliding pieces that attack this square
    let black_pawns = pieces_bb[PAWN as usize] & colors_bb[BLACK as usize] & pawn_attacks(WHITE, square);
    let white_pawns = pieces_bb[PAWN as usize] & colors_bb[WHITE as usize] & pawn_attacks(BLACK, square);
    let knights = pieces_bb[KNIGHT as usize] & knight_attacks(square);
    let kings = if king_attacker_on() {
        pieces_bb[KING as usize] & king_attacks(square)
    } else {
        0
    };

    let mut non_sliders = (black_pawns | white_pawns | knights | kings) & occ;
    while non_sliders != 0 {
        let ns_sq = non_sliders.trailing_zeros();
        non_sliders &= non_sliders - 1;
        let ns_pt = mailbox[ns_sq as usize];
        if ns_pt >= 6 { continue; }
        let ns_color = if white_bb & (1u64 << ns_sq) != 0 { WHITE } else { BLACK };
        deltas.push(RawThreatDelta::new(colored_piece(ns_color, ns_pt) as u8, ns_sq as u8, cp as u8, square as u8, add));
    }

    #[cfg(feature = "profile-threats")]
    {
        crate::threats::thr_stats::record_section(
            4,
            crate::threats::thr_stats::rdtsc().wrapping_sub(s3_start),
            deltas.len() as u64 - s3_deltas_before,
        );
        crate::threats::thr_stats::record_total(
            crate::threats::thr_stats::rdtsc().wrapping_sub(fn_start_tsc)
        );
    }
}

/// Apply raw threat deltas to update the threat accumulator incrementally.
/// Copies from `prev` and applies all deltas for a specific perspective.
///
/// Runtime dispatcher (function multiversioning): on x86_64 with AVX2 it
/// calls the `target_feature`-specialized wrapper; otherwise the plain
/// scalar-codegen body. A bare `#[target_feature(enable = "avx2")]` on the
/// body autovectorizes its scalar fallback (in `apply_threat_indices`) to
/// AVX2 and SIGILLs on pre-AVX2 CPUs; dispatching keeps AVX2 codegen on
/// capable hosts while staying correct on older ones. See the matching
/// note on `NNUENet::forward_with_l1_pairwise_inner`.
///
/// # Safety
/// `threat_weights` must be at least `num_threats * hidden_size` elements,
/// and every delta index must be `< num_threats`.
pub unsafe fn apply_threat_deltas(
    dst: &mut [i16],           // destination threat accumulator (one perspective)
    src: &[i16],               // source (previous position's threat accumulator)
    deltas: &[RawThreatDelta],
    threat_weights: &[i8],     // [num_threats × hidden_size]
    hidden_size: usize,
    num_threats: usize,
    pov: Color,
    mirrored: bool,
    pp_deltas: &[crate::pawn_pair::PawnPairDelta],
    pp_base: usize,
) {
    // Runtime dispatch only in non-AVX2-baseline builds; AVX2-baseline (native
    // fleet) builds fall through to the body directly — see the note on
    // `NNUENet::forward_with_l1_pairwise_inner`.
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if is_x86_feature_detected!("avx2") {
        return unsafe {
            apply_threat_deltas_avx2(
                dst, src, deltas, threat_weights, hidden_size, num_threats, pov, mirrored,
                pp_deltas, pp_base)
        };
    }
    unsafe {
        apply_threat_deltas_body(
            dst, src, deltas, threat_weights, hidden_size, num_threats, pov, mirrored,
            pp_deltas, pp_base)
    }
}

/// AVX2-specialized wrapper for [`apply_threat_deltas`]. Only call when AVX2
/// is available.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[target_feature(enable = "avx2")]
unsafe fn apply_threat_deltas_avx2(
    dst: &mut [i16],
    src: &[i16],
    deltas: &[RawThreatDelta],
    threat_weights: &[i8],
    hidden_size: usize,
    num_threats: usize,
    pov: Color,
    mirrored: bool,
    pp_deltas: &[crate::pawn_pair::PawnPairDelta],
    pp_base: usize,
) {
    unsafe {
        apply_threat_deltas_body(
            dst, src, deltas, threat_weights, hidden_size, num_threats, pov, mirrored,
            pp_deltas, pp_base)
    }
}

/// Shared body for [`apply_threat_deltas`] — no `target_feature`, so its
/// (and `apply_threat_indices`') scalar fallbacks compile to scalar code.
#[inline(always)]
unsafe fn apply_threat_deltas_body(
    dst: &mut [i16],
    src: &[i16],
    deltas: &[RawThreatDelta],
    threat_weights: &[i8],
    hidden_size: usize,
    num_threats: usize,
    pov: Color,
    mirrored: bool,
    pp_deltas: &[crate::pawn_pair::PawnPairDelta],
    pp_base: usize,
) {
    #[cfg(feature = "profile-threats")]
    crate::threats::apply_stats::record(deltas.len());

    // Collect valid add/sub indices (stack-allocated, no heap). Use
    // MaybeUninit to skip the 2 KB zero-init per array — only [..n_adds]
    // and [..n_subs] are written/read. Fired twice per push (one per
    // perspective) at ~600k pushes per bench = ~2.4 GB of avoided
    // memset traffic. Same pattern that gave +3% bench in
    // forward_with_l1_pairwise_inner.
    let mut adds_storage = std::mem::MaybeUninit::<[usize; MAX_THREAT_DELTAS + crate::pawn_pair::MAX_PAWN_PAIR_DELTAS]>::uninit();
    let mut subs_storage = std::mem::MaybeUninit::<[usize; MAX_THREAT_DELTAS + crate::pawn_pair::MAX_PAWN_PAIR_DELTAS]>::uninit();
    let adds_ptr = scratch_ptr!(adds_storage, usize);
    let subs_ptr = scratch_ptr!(subs_storage, usize);
    let mut n_adds = 0usize;
    let mut n_subs = 0usize;
    for delta in deltas {
        let idx = threat_index(
            delta.attacker_cp() as usize,
            delta.from_sq() as u32,
            delta.victim_cp() as usize,
            delta.to_sq() as u32,
            mirrored,
            pov,
        );
        if idx < 0 || (idx as usize) >= num_threats { continue; }
        if delta.add() {
            unsafe { adds_ptr.add(n_adds).write(idx as usize); }
            n_adds += 1;
        } else {
            unsafe { subs_ptr.add(n_subs).write(idx as usize); }
            n_subs += 1;
        }
    }
    // Pawn-pair indices join the SAME lists: they address the same weight
    // array above the threat block, so folding them in here means one SIMD
    // pass instead of two and no second source copy. A separate scalar pass
    // measured -10.7% NPS on its own.
    for d in pp_deltas {
        if let Some(i) = crate::pawn_pair::pp_index_for(*d, pov, mirrored) {
            if d.add() {
                unsafe { adds_ptr.add(n_adds).write(pp_base + i); }
                n_adds += 1;
            } else {
                unsafe { subs_ptr.add(n_subs).write(pp_base + i); }
                n_subs += 1;
            }
        }
    }
    let adds = scratch_slice!(adds_ptr, n_adds);
    let subs = scratch_slice!(subs_ptr, n_subs);

    #[cfg(feature = "profile-threats")]
    crate::threats::apply_stats::record_cancel(adds, subs);

    unsafe {
        apply_threat_indices(dst, src, threat_weights, hidden_size, adds, subs);
    }
}

/// Apply raw threat deltas for both perspectives after a shared replay walk.
///
/// This avoids traversing the raw delta list twice when the WHITE and BLACK
/// threat accumulators can replay from the same ancestor. The SIMD apply phase
/// still runs once per perspective because the destination accumulators differ.
///
/// # Safety
/// Same requirements as [`apply_threat_deltas`].
///
/// Runtime dispatcher (function multiversioning) — see [`apply_threat_deltas`].
pub unsafe fn apply_threat_deltas_dual(
    dst_w: &mut [i16],
    src_w: &[i16],
    dst_b: &mut [i16],
    src_b: &[i16],
    deltas: &[RawThreatDelta],
    threat_weights: &[i8],
    hidden_size: usize,
    num_threats: usize,
    mirrored_w: bool,
    mirrored_b: bool,
    pp_deltas: &[crate::pawn_pair::PawnPairDelta],
    pp_base: usize,
) {
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if is_x86_feature_detected!("avx2") {
        return unsafe {
            apply_threat_deltas_dual_avx2(
                dst_w, src_w, dst_b, src_b, deltas, threat_weights,
                hidden_size, num_threats, mirrored_w, mirrored_b, pp_deltas, pp_base)
        };
    }
    unsafe {
        apply_threat_deltas_dual_body(
            dst_w, src_w, dst_b, src_b, deltas, threat_weights,
            hidden_size, num_threats, mirrored_w, mirrored_b, pp_deltas, pp_base)
    }
}

/// AVX2-specialized wrapper for [`apply_threat_deltas_dual`]. Only call when
/// AVX2 is available.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[target_feature(enable = "avx2")]
unsafe fn apply_threat_deltas_dual_avx2(
    dst_w: &mut [i16],
    src_w: &[i16],
    dst_b: &mut [i16],
    src_b: &[i16],
    deltas: &[RawThreatDelta],
    threat_weights: &[i8],
    hidden_size: usize,
    num_threats: usize,
    mirrored_w: bool,
    mirrored_b: bool,
    pp_deltas: &[crate::pawn_pair::PawnPairDelta],
    pp_base: usize,
) {
    unsafe {
        apply_threat_deltas_dual_body(
            dst_w, src_w, dst_b, src_b, deltas, threat_weights,
            hidden_size, num_threats, mirrored_w, mirrored_b, pp_deltas, pp_base)
    }
}

/// Shared body for [`apply_threat_deltas_dual`] — no `target_feature`.
#[inline(always)]
unsafe fn apply_threat_deltas_dual_body(
    dst_w: &mut [i16],
    src_w: &[i16],
    dst_b: &mut [i16],
    src_b: &[i16],
    deltas: &[RawThreatDelta],
    threat_weights: &[i8],
    hidden_size: usize,
    num_threats: usize,
    mirrored_w: bool,
    mirrored_b: bool,
    pp_deltas: &[crate::pawn_pair::PawnPairDelta],
    pp_base: usize,
) {
    #[cfg(feature = "profile-threats")]
    {
        crate::threats::apply_stats::record(deltas.len());
        crate::threats::apply_stats::record(deltas.len());
    }

    let mut adds_w_storage = std::mem::MaybeUninit::<[usize; MAX_THREAT_DELTAS + crate::pawn_pair::MAX_PAWN_PAIR_DELTAS]>::uninit();
    let mut subs_w_storage = std::mem::MaybeUninit::<[usize; MAX_THREAT_DELTAS + crate::pawn_pair::MAX_PAWN_PAIR_DELTAS]>::uninit();
    let mut adds_b_storage = std::mem::MaybeUninit::<[usize; MAX_THREAT_DELTAS + crate::pawn_pair::MAX_PAWN_PAIR_DELTAS]>::uninit();
    let mut subs_b_storage = std::mem::MaybeUninit::<[usize; MAX_THREAT_DELTAS + crate::pawn_pair::MAX_PAWN_PAIR_DELTAS]>::uninit();
    let adds_w_ptr = scratch_ptr!(adds_w_storage, usize);
    let subs_w_ptr = scratch_ptr!(subs_w_storage, usize);
    let adds_b_ptr = scratch_ptr!(adds_b_storage, usize);
    let subs_b_ptr = scratch_ptr!(subs_b_storage, usize);
    let mut n_adds_w = 0usize;
    let mut n_subs_w = 0usize;
    let mut n_adds_b = 0usize;
    let mut n_subs_b = 0usize;
    let tables = get_threat_tables();
    let flip_w = 7 * mirrored_w as u32;
    let flip_b = (7 * mirrored_b as u32) ^ 56;

    for delta in deltas {
        let attacker = delta.attacker_cp() as usize;
        let from = delta.from_sq() as u32;
        let victim = delta.victim_cp() as usize;
        let to = delta.to_sq() as u32;
        let add = delta.add();

        let pair_w = tables.pairs[attacker][victim];
        if !pair_w.skip(from, to) {
            let from_w = (from ^ flip_w) as usize;
            let to_w = (to ^ flip_w) as usize;
            let idx_w = pair_w.base
                + tables.from_offset[attacker][from_w]
                + tables.ray_rank[attacker][from_w][to_w] as i32;
            if (idx_w as usize) < num_threats {
                if add {
                    unsafe { adds_w_ptr.add(n_adds_w).write(idx_w as usize); }
                    n_adds_w += 1;
                } else {
                    unsafe { subs_w_ptr.add(n_subs_w).write(idx_w as usize); }
                    n_subs_w += 1;
                }
            }
        }

        let attacker_b = flipped_colored_piece(attacker);
        let victim_b = flipped_colored_piece(victim);
        let pair_b = tables.pairs[attacker_b][victim_b];
        if !pair_b.skip(from, to) {
            let from_b = (from ^ flip_b) as usize;
            let to_b = (to ^ flip_b) as usize;
            let idx_b = pair_b.base
                + tables.from_offset[attacker_b][from_b]
                + tables.ray_rank[attacker_b][from_b][to_b] as i32;
            if (idx_b as usize) < num_threats {
                if add {
                    unsafe { adds_b_ptr.add(n_adds_b).write(idx_b as usize); }
                    n_adds_b += 1;
                } else {
                    unsafe { subs_b_ptr.add(n_subs_b).write(idx_b as usize); }
                    n_subs_b += 1;
                }
            }
        }
    }

    // Same fold as the single-perspective path, once per perspective.
    for d in pp_deltas {
        for (pov, mirrored, a_ptr, s_ptr, na, ns) in [
            (WHITE, mirrored_w, adds_w_ptr, subs_w_ptr, &mut n_adds_w, &mut n_subs_w),
            (BLACK, mirrored_b, adds_b_ptr, subs_b_ptr, &mut n_adds_b, &mut n_subs_b),
        ] {
            if let Some(i) = crate::pawn_pair::pp_index_for(*d, pov, mirrored) {
                if d.add() {
                    unsafe { a_ptr.add(*na).write(pp_base + i); }
                    *na += 1;
                } else {
                    unsafe { s_ptr.add(*ns).write(pp_base + i); }
                    *ns += 1;
                }
            }
        }
    }
    let adds_w = scratch_slice!(adds_w_ptr, n_adds_w);
    let subs_w = scratch_slice!(subs_w_ptr, n_subs_w);
    let adds_b = scratch_slice!(adds_b_ptr, n_adds_b);
    let subs_b = scratch_slice!(subs_b_ptr, n_subs_b);

    #[cfg(feature = "profile-threats")]
    {
        crate::threats::apply_stats::record_cancel(adds_w, subs_w);
        crate::threats::apply_stats::record_cancel(adds_b, subs_b);
    }

    unsafe {
        apply_threat_indices(dst_w, src_w, threat_weights, hidden_size, adds_w, subs_w);
        apply_threat_indices(dst_b, src_b, threat_weights, hidden_size, adds_b, subs_b);
    }
}

/// No `target_feature` (see [`apply_threat_deltas`]): the heavy SIMD work is
/// dispatched internally to the separately-gated `apply_deltas_avx2` /
/// `apply_deltas_avx512` kernels via `is_x86_feature_detected!`, so the
/// attribute would only autovectorize the (never-taken-on-AVX2) scalar
/// fallback below — which SIGILLs on pre-AVX2 CPUs. Dropping it keeps the
/// fallback scalar and correct everywhere with no perf cost on capable hosts.
unsafe fn apply_threat_indices(
    dst: &mut [i16],
    src: &[i16],
    threat_weights: &[i8],
    hidden_size: usize,
    adds: &[usize],
    subs: &[usize],
) {
    // Prefetch the FIRST CHUNK (128 bytes = 2 lines) of every row before the
    // kernel starts. The kernels walk all rows chunk-by-chunk, so chunk-0
    // accesses are the cold misses; later chunks are covered by the in-loop
    // next-chunk prefetch plus hardware stream prefetchers. Line-0-of-first-
    // 4-rows (the previous form) left ~15 rows × 2 lines cold per call —
    // perf annotate showed the row loads (vpmovsxbw) stalling at 16%+ of
    // the function. Capped at 24 rows to bound issue cost on refresh-sized
    // delta lists (avg 9.4 rows, p99 ≈ 48).
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        for &idx in adds.iter().chain(subs.iter()).take(24) {
            unsafe {
                let row = threat_weights.as_ptr().add(idx * hidden_size);
                _mm_prefetch(row as *const i8, _MM_HINT_T0);
                _mm_prefetch(row.add(64) as *const i8, _MM_HINT_T0);
            }
        }
    }

    // Apply weight rows with SIMD when available. Fused pattern: load src
    // chunk into registers, apply all adds/subs, store to dst. Avoids the
    // separate copy_from_slice pass that used to precede apply_deltas_avx2.
    //
    // Dispatch order: AVX-512 (zmm, 32 i16 per reg) > AVX-2 (ymm, 16 i16
    // per reg) > scalar. The AVX-512 path matters — without it this function
    // measured 17.98% of cycles.
    #[cfg(target_arch = "x86_64")]
    {
        let tier = x86_simd_tier();
        if tier >= 2 && hidden_size.is_multiple_of(32) {
            unsafe {
                apply_deltas_avx512(dst, src, threat_weights, hidden_size, adds, subs);
            }
            return;
        }
        if tier >= 1 && hidden_size.is_multiple_of(16) {
            unsafe {
                apply_deltas_avx2(dst, src, threat_weights, hidden_size, adds, subs);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if hidden_size % 8 == 0 {
            unsafe {
                apply_deltas_neon(dst, src, threat_weights, hidden_size, &adds, &subs);
            }
            return;
        }
    }

    // Scalar fallback: single pass, reads src once, writes dst once.
    for j in 0..hidden_size {
        let mut v = src[j];
        for &idx in adds {
            v += threat_weights[idx * hidden_size + j] as i16;
        }
        for &idx in subs {
            v -= threat_weights[idx * hidden_size + j] as i16;
        }
        dst[j] = v;
    }
}

/// AVX2 SIMD: apply threat weight rows to accumulator using register tiling.
/// Loads accumulator chunk into registers ONCE, applies ALL deltas while in
/// registers, then stores ONCE — 21× less memory
/// traffic than per-delta streaming.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_deltas_avx2(
    dst: &mut [i16],
    src: &[i16],
    threat_weights: &[i8],
    hidden_size: usize,
    adds: &[usize],
    subs: &[usize],
) {
    use std::arch::x86_64::*;

    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();
    let w_ptr = threat_weights.as_ptr();

    // 8 AVX2 registers × 16 i16 = 128 elements per chunk. REGS=12 (the
    // simd_acc_fused_avx2 / SF AVX2 budget) was tried and measured +5.6%
    // SLOWER on Zen 1 (perf stat -r 3, IPC
    // 1.26 -> 1.18): Zen 1 cracks every 256-bit op into 2x128-bit uops,
    // and 12 live YMM accumulators plus the two cvtepi8_epi16 widening
    // temps per step oversubscribe the physical register file — the same
    // failure mode as REGS=24 on the AVX-512 variant (-4.4%, see its
    // comment). This kernel differs from simd_acc_fused_avx2 (fine at 12)
    // in needing widening temps instead of folded memory operands. Keep 8.
    const REGS: usize = 8;
    const CHUNK: usize = REGS * 16; // 128 elements

    let mut offset = 0;

    // Inner-loop body, parameterised on the iteration count so the
    // full-chunk fast path uses REGS as a const and the tail uses runtime
    // nregs. Both call sites share the same logic. `offset` must be
    // declared before this macro_rules! definition for macro hygiene.
    macro_rules! apply_chunk {
        ($nregs:expr) => {{
            let nregs: usize = $nregs;
            let mut regs: [__m256i; REGS] = [_mm256_setzero_si256(); REGS];
            for i in 0..nregs {
                regs[i] = _mm256_loadu_si256(src_ptr.add(offset + i * 16) as *const __m256i);
            }
            // Per-row next-chunk prefetch: while summing this row's bytes
            // [offset, offset+CHUNK), pull its [offset+CHUNK, +2 lines) into
            // L1 so the next outer-chunk iteration streams. Prefetch never
            // faults, so no bounds guard is needed at the table tail.
            let pf = offset + CHUNK < hidden_size;
            let mut ai = 0;
            let mut si = 0;
            while ai < adds.len() && si < subs.len() {
                let aw = w_ptr.add(adds[ai] * hidden_size + offset);
                let sw = w_ptr.add(subs[si] * hidden_size + offset);
                if pf {
                    _mm_prefetch(aw.add(CHUNK) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(aw.add(CHUNK + 64) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(sw.add(CHUNK) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(sw.add(CHUNK + 64) as *const i8, _MM_HINT_T0);
                }
                for i in 0..nregs {
                    let add_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(aw.add(i * 16) as *const __m128i));
                    let sub_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(sw.add(i * 16) as *const __m128i));
                    regs[i] = _mm256_sub_epi16(_mm256_add_epi16(regs[i], add_w), sub_w);
                }
                ai += 1;
                si += 1;
            }
            while ai < adds.len() {
                let aw = w_ptr.add(adds[ai] * hidden_size + offset);
                if pf {
                    _mm_prefetch(aw.add(CHUNK) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(aw.add(CHUNK + 64) as *const i8, _MM_HINT_T0);
                }
                for i in 0..nregs {
                    let add_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(aw.add(i * 16) as *const __m128i));
                    regs[i] = _mm256_add_epi16(regs[i], add_w);
                }
                ai += 1;
            }
            while si < subs.len() {
                let sw = w_ptr.add(subs[si] * hidden_size + offset);
                if pf {
                    _mm_prefetch(sw.add(CHUNK) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(sw.add(CHUNK + 64) as *const i8, _MM_HINT_T0);
                }
                for i in 0..nregs {
                    let sub_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(sw.add(i * 16) as *const __m128i));
                    regs[i] = _mm256_sub_epi16(regs[i], sub_w);
                }
                si += 1;
            }
            for i in 0..nregs {
                _mm256_storeu_si256(dst_ptr.add(offset + i * 16) as *mut __m256i, regs[i]);
            }
        }};
    }

    // Full-chunk fast path: REGS is a compile-time constant, inner loops
    // unroll without dispatch. For h=1024 (v10 prod, CHUNK=128) all 8
    // iterations hit this branch.
    while offset + CHUNK <= hidden_size {
        apply_chunk!(REGS);
        offset += CHUNK;
    }

    // Tail: never fires on prod hidden sizes (multiples of 128); a 64-elem
    // remainder gets a const-4-reg unrolled path, anything else the
    // runtime-nregs form.
    if offset < hidden_size {
        let nregs = (hidden_size - offset).div_ceil(16);
        if nregs == 4 {
            apply_chunk!(4);
        } else {
            apply_chunk!(nregs);
        }
    }
}

/// AVX-512 SIMD: apply threat weight rows to accumulator using register tiling.
/// Mirrors `apply_deltas_avx2` exactly — same pattern, twice-wide registers
/// (zmm = 32 i16 per reg vs ymm = 16). 8 zmm regs × 32 i16 = 256 elements
/// per chunk vs AVX2's 128. For hidden_size=768 that's 3 chunks instead of
/// 6 — half the outer-loop iterations.
///
/// Why this exists: without it `apply_threat_deltas` measured 17.98% of
/// cycles. Faster engines have a dedicated AVX-512 threat path and spend only
/// ~1.8% of cycles on the analogous update.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn apply_deltas_avx512(
    dst: &mut [i16],
    src: &[i16],
    threat_weights: &[i8],
    hidden_size: usize,
    adds: &[usize],
    subs: &[usize],
) {
    use std::arch::x86_64::*;

    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();
    let w_ptr = threat_weights.as_ptr();

    // 16 AVX-512 registers × 32 i16 = 512 elements per chunk — a partial
    // step toward a full-L1-tile register layout. REGS=24 here regressed
    // −4.4% on Zen 5 because the
    // i8→i16 expansion (`_mm512_cvtepi8_epi16`) needs temporary registers,
    // and 24 + temps spilled past the 32 ZMM file. PSQ-side `simd_acc_fused_avx512`
    // uses REGS=24 because direct i16 add has no expansion.
    const REGS: usize = 16;
    const CHUNK: usize = REGS * 32; // 512 elements

    let mut offset = 0;

    // Compile-time nregs everywhere — same dispatch-elimination split as
    // apply_deltas_avx2. h=1024 (current prod) → two const 16-reg passes;
    // h=768 → one const 16-reg + one const 8-reg (256-element) pass.
    // Runtime-nregs tail only for h not a multiple of 256.
    //
    // Paired add+sub inside: _mm512_cvtepi8_epi16 widens 32 i8 (loaded as
    // a 256-bit ymm) into 32 i16 in a 512-bit zmm — sign-extending in a
    // single VPMOVSXBW instruction. Same pattern as the AVX2 helper but
    // at twice the width.
    macro_rules! apply_chunk {
        ($nregs:expr) => {{
            let nregs: usize = $nregs;
            // Seed chunk accumulator from src (parent).
            let mut regs: [__m512i; REGS] = [_mm512_setzero_si512(); REGS];
            for i in 0..nregs {
                regs[i] = _mm512_loadu_si512(src_ptr.add(offset + i * 32) as *const _);
            }
            // Per-row next-chunk prefetch — same rationale as the AVX2 twin
            // (chunk-0 cold misses dominate; see apply_threat_indices entry
            // prefetch). CHUNK here is 512 bytes = 8 lines per row.
            let pf = offset + CHUNK < hidden_size;
            macro_rules! pf_row {
                ($row:expr) => {
                    if pf {
                        let mut l = 0;
                        while l < CHUNK {
                            _mm_prefetch($row.add(CHUNK + l) as *const i8, _MM_HINT_T0);
                            l += 64;
                        }
                    }
                };
            }
            let mut ai = 0;
            let mut si = 0;
            while ai < adds.len() && si < subs.len() {
                let aw = w_ptr.add(adds[ai] * hidden_size + offset);
                let sw = w_ptr.add(subs[si] * hidden_size + offset);
                pf_row!(aw);
                pf_row!(sw);
                for i in 0..nregs {
                    let add_w = _mm512_cvtepi8_epi16(_mm256_loadu_si256(aw.add(i * 32) as *const __m256i));
                    let sub_w = _mm512_cvtepi8_epi16(_mm256_loadu_si256(sw.add(i * 32) as *const __m256i));
                    regs[i] = _mm512_sub_epi16(_mm512_add_epi16(regs[i], add_w), sub_w);
                }
                ai += 1;
                si += 1;
            }
            while ai < adds.len() {
                let aw = w_ptr.add(adds[ai] * hidden_size + offset);
                pf_row!(aw);
                for i in 0..nregs {
                    let add_w = _mm512_cvtepi8_epi16(_mm256_loadu_si256(aw.add(i * 32) as *const __m256i));
                    regs[i] = _mm512_add_epi16(regs[i], add_w);
                }
                ai += 1;
            }
            while si < subs.len() {
                let sw = w_ptr.add(subs[si] * hidden_size + offset);
                pf_row!(sw);
                for i in 0..nregs {
                    let sub_w = _mm512_cvtepi8_epi16(_mm256_loadu_si256(sw.add(i * 32) as *const __m256i));
                    regs[i] = _mm512_sub_epi16(regs[i], sub_w);
                }
                si += 1;
            }
            for i in 0..nregs {
                _mm512_storeu_si512(dst_ptr.add(offset + i * 32) as *mut _, regs[i]);
            }
        }};
    }

    while offset + CHUNK <= hidden_size {
        apply_chunk!(REGS);
        offset += CHUNK;
    }
    while offset + 8 * 32 <= hidden_size {
        apply_chunk!(8);
        offset += 8 * 32;
    }
    if offset < hidden_size {
        apply_chunk!((hidden_size - offset).div_ceil(32));
    }
}

/// Add multiple weight rows to an accumulator (SIMD for refresh).
/// dst is already zeroed. Adds weight rows for each feature index.
pub fn add_weight_rows(
    dst: &mut [i16],
    threat_weights: &[i8],
    hidden_size: usize,
    indices: &[usize],
) {
    if indices.is_empty() { return; }

    // Dispatch order: AVX-512 (zmm, 32 i16/reg) > AVX-2 > scalar.
    // Sibling of apply_threat_deltas's AVX-512 dispatch — same perf
    // rationale (zmm register width on AVX-512+VNNI hosts).
    #[cfg(target_arch = "x86_64")]
    {
        let tier = x86_simd_tier();
        if tier >= 2 && hidden_size.is_multiple_of(32) {
            unsafe {
                add_weight_rows_avx512(dst, threat_weights, hidden_size, indices);
            }
            return;
        }
        if tier >= 1 {
            unsafe {
                add_weight_rows_avx2(dst, threat_weights, hidden_size, indices);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    if hidden_size % 8 == 0 {
        unsafe {
            add_weight_rows_neon(dst, threat_weights, hidden_size, indices);
        }
        return;
    }

    // Scalar fallback
    for &idx in indices {
        let w_off = idx * hidden_size;
        for j in 0..hidden_size {
            dst[j] += threat_weights[w_off + j] as i16;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_weight_rows_avx2(
    dst: &mut [i16],
    threat_weights: &[i8],
    hidden_size: usize,
    indices: &[usize],
) {
    use std::arch::x86_64::*;

    let dst_ptr = dst.as_mut_ptr();
    let w_ptr = threat_weights.as_ptr();

    const REGS: usize = 8;
    const CHUNK: usize = REGS * 16; // 128 elements

    let mut offset = 0;

    // Compile-time nregs on the main loop (h=768 and h=1024 both divide
    // CHUNK evenly); runtime-nregs only in the off-boundary tail.
    macro_rules! apply_chunk {
        ($nregs:expr) => {{
            let nregs: usize = $nregs;
            // Load accumulator chunk into registers
            let mut regs: [__m256i; REGS] = [_mm256_setzero_si256(); REGS];
            for i in 0..nregs {
                regs[i] = _mm256_loadu_si256(dst_ptr.add(offset + i * 16) as *const __m256i);
            }
            // Add all weight rows with prefetch for next row
            for (fi, &idx) in indices.iter().enumerate() {
                let aw = w_ptr.add(idx * hidden_size + offset);
                // Prefetch next feature's weight row
                if fi + 1 < indices.len() {
                    _mm_prefetch(w_ptr.add(indices[fi + 1] * hidden_size + offset) as *const i8, _MM_HINT_T0);
                }
                for i in 0..nregs {
                    let add_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(aw.add(i * 16) as *const __m128i));
                    regs[i] = _mm256_add_epi16(regs[i], add_w);
                }
            }
            // Store registers back
            for i in 0..nregs {
                _mm256_storeu_si256(dst_ptr.add(offset + i * 16) as *mut __m256i, regs[i]);
            }
        }};
    }

    while offset + CHUNK <= hidden_size {
        apply_chunk!(REGS);
        offset += CHUNK;
    }
    if offset < hidden_size {
        apply_chunk!((hidden_size - offset).div_ceil(16));
    }
}

/// AVX-512 SIMD: accumulate multiple weight rows into dst (full threat refresh).
/// Mirrors `add_weight_rows_avx2` — 8 zmm regs × 32 i16 = 256 elements per
/// chunk vs AVX2's 128. Half the outer-loop iterations on hidden_size=768.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn add_weight_rows_avx512(
    dst: &mut [i16],
    threat_weights: &[i8],
    hidden_size: usize,
    indices: &[usize],
) {
    use std::arch::x86_64::*;

    let dst_ptr = dst.as_mut_ptr();
    let w_ptr = threat_weights.as_ptr();

    const REGS: usize = 8;
    const CHUNK: usize = REGS * 32; // 256 elements

    let mut offset = 0;

    // Compile-time nregs on the main loop (h=768 and h=1024 both divide
    // CHUNK evenly); runtime-nregs only in the off-boundary tail.
    macro_rules! apply_chunk {
        ($nregs:expr) => {{
            let nregs: usize = $nregs;
            // Load existing dst chunk (the function adds to it, doesn't replace).
            let mut regs: [__m512i; REGS] = [_mm512_setzero_si512(); REGS];
            for i in 0..nregs {
                regs[i] = _mm512_loadu_si512(dst_ptr.add(offset + i * 32) as *const _);
            }
            // Add all weight rows; prefetch the next row to hide L3 latency.
            for (fi, &idx) in indices.iter().enumerate() {
                let aw = w_ptr.add(idx * hidden_size + offset);
                if fi + 1 < indices.len() {
                    _mm_prefetch(
                        w_ptr.add(indices[fi + 1] * hidden_size + offset) as *const i8,
                        _MM_HINT_T0,
                    );
                }
                for i in 0..nregs {
                    let add_w = _mm512_cvtepi8_epi16(_mm256_loadu_si256(aw.add(i * 32) as *const __m256i));
                    regs[i] = _mm512_add_epi16(regs[i], add_w);
                }
            }
            // Store back.
            for i in 0..nregs {
                _mm512_storeu_si512(dst_ptr.add(offset + i * 32) as *mut _, regs[i]);
            }
        }};
    }

    while offset + CHUNK <= hidden_size {
        apply_chunk!(REGS);
        offset += CHUNK;
    }
    if offset < hidden_size {
        apply_chunk!((hidden_size - offset).div_ceil(32));
    }
}

/// NEON SIMD: apply threat weight rows to accumulator using register tiling.
/// Mirrors apply_deltas_avx2 — fused load src / apply adds+subs / store dst.
/// 16 regs × 8 i16 = 128 elements per chunk (same footprint as AVX2 8×16).
#[cfg(target_arch = "aarch64")]
unsafe fn apply_deltas_neon(
    dst: &mut [i16],
    src: &[i16],
    threat_weights: &[i8],
    hidden_size: usize,
    adds: &[usize],
    subs: &[usize],
) {
    use std::arch::aarch64::*;

    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();
    let w_ptr = threat_weights.as_ptr();

    const REGS: usize = 16;
    const CHUNK: usize = REGS * 8; // 128 elements

    let mut offset = 0;
    while offset < hidden_size {
        let chunk_size = (hidden_size - offset).min(CHUNK);
        let nregs = (chunk_size + 7) / 8;

        // Seed chunk accumulator from src.
        let mut regs: [int16x8_t; REGS] = [vdupq_n_s16(0); REGS];
        for i in 0..nregs {
            regs[i] = vld1q_s16(src_ptr.add(offset + i * 8));
        }

        // Paired add+sub: one register of each per iteration, reuses chunk regs.
        // Uses vaddw_s8/vsubw_s8 which fuse widen+add and widen+sub into a
        // single instruction each, avoiding a separate vmovl_s8 pass.
        let mut ai = 0;
        let mut si = 0;
        while ai < adds.len() && si < subs.len() {
            let aw = w_ptr.add(adds[ai] * hidden_size + offset);
            let sw = w_ptr.add(subs[si] * hidden_size + offset);
            for i in 0..nregs {
                regs[i] = vaddw_s8(regs[i], vld1_s8(aw.add(i * 8)));
                regs[i] = vsubw_s8(regs[i], vld1_s8(sw.add(i * 8)));
            }
            ai += 1;
            si += 1;
        }

        while ai < adds.len() {
            let aw = w_ptr.add(adds[ai] * hidden_size + offset);
            for i in 0..nregs {
                regs[i] = vaddw_s8(regs[i], vld1_s8(aw.add(i * 8)));
            }
            ai += 1;
        }

        while si < subs.len() {
            let sw = w_ptr.add(subs[si] * hidden_size + offset);
            for i in 0..nregs {
                regs[i] = vsubw_s8(regs[i], vld1_s8(sw.add(i * 8)));
            }
            si += 1;
        }

        for i in 0..nregs {
            vst1q_s16(dst_ptr.add(offset + i * 8), regs[i]);
        }

        offset += CHUNK;
    }
}

/// NEON SIMD: accumulate multiple weight rows into dst (for full threat refresh).
/// Mirrors add_weight_rows_avx2.
#[cfg(target_arch = "aarch64")]
unsafe fn add_weight_rows_neon(
    dst: &mut [i16],
    threat_weights: &[i8],
    hidden_size: usize,
    indices: &[usize],
) {
    use std::arch::aarch64::*;

    let dst_ptr = dst.as_mut_ptr();
    let w_ptr = threat_weights.as_ptr();

    const REGS: usize = 16;
    const CHUNK: usize = REGS * 8;

    let mut offset = 0;
    while offset < hidden_size {
        let chunk_size = (hidden_size - offset).min(CHUNK);
        let nregs = (chunk_size + 7) / 8;

        let mut regs: [int16x8_t; REGS] = [vdupq_n_s16(0); REGS];
        for i in 0..nregs {
            regs[i] = vld1q_s16(dst_ptr.add(offset + i * 8));
        }

        for &idx in indices.iter() {
            let aw = w_ptr.add(idx * hidden_size + offset);
            for i in 0..nregs {
                regs[i] = vaddw_s8(regs[i], vld1_s8(aw.add(i * 8)));
            }
        }

        for i in 0..nregs {
            vst1q_s16(dst_ptr.add(offset + i * 8), regs[i]);
        }

        offset += CHUNK;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar reference for apply_deltas_{avx2,neon} — mirrors the
    /// dispatcher's scalar fallback exactly so SIMD paths can be
    /// validated against it.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn apply_deltas_scalar_ref(
        dst: &mut [i16],
        src: &[i16],
        threat_weights: &[i8],
        hidden_size: usize,
        adds: &[usize],
        subs: &[usize],
    ) {
        for j in 0..hidden_size {
            let mut v = src[j];
            for &idx in adds { v += threat_weights[idx * hidden_size + j] as i16; }
            for &idx in subs { v -= threat_weights[idx * hidden_size + j] as i16; }
            dst[j] = v;
        }
    }

    /// Seeded xorshift64* for deterministic test inputs.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    /// Shared body for the x86 apply_deltas SIMD-vs-scalar tests.
    /// `h` sweeps cover FT768 (768), FT1024 (512/1024, current prod) and
    /// an off-chunk-boundary width (320) so the runtime tails are hit.
    #[cfg(target_arch = "x86_64")]
    fn check_apply_deltas_kernel(
        h: usize,
        seed: u64,
        kernel: unsafe fn(&mut [i16], &[i16], &[i8], usize, &[usize], &[usize]),
        name: &str,
    ) {
        let n_threats = 64;
        let mut r = rng(seed ^ h as u64);

        let mut weights = vec![0i8; n_threats * h];
        for w in weights.iter_mut() { *w = (r() % 256) as i8; }

        let mut src = vec![0i16; h];
        for v in src.iter_mut() { *v = (r() as i32 as i16).rem_euclid(2001) - 1000; }

        // Mixed adds+subs — exercises the paired inner loop.
        let adds = [3usize, 8, 21, 40];
        let subs = [5usize, 12, 30, 55, 63];
        let mut scalar_dst = vec![0i16; h];
        apply_deltas_scalar_ref(&mut scalar_dst, &src, &weights, h, &adds, &subs);
        let mut simd_dst = vec![0i16; h];
        unsafe { kernel(&mut simd_dst, &src, &weights, h, &adds, &subs); }
        assert_eq!(scalar_dst, simd_dst, "{} mixed diverged at h={}", name, h);

        // Adds-only — tail loop for adds.
        let mut scalar_dst = vec![0i16; h];
        apply_deltas_scalar_ref(&mut scalar_dst, &src, &weights, h, &adds, &[]);
        let mut simd_dst = vec![0i16; h];
        unsafe { kernel(&mut simd_dst, &src, &weights, h, &adds, &[]); }
        assert_eq!(scalar_dst, simd_dst, "{} adds-only diverged at h={}", name, h);

        // Subs-only — tail loop for subs.
        let mut scalar_dst = vec![0i16; h];
        apply_deltas_scalar_ref(&mut scalar_dst, &src, &weights, h, &[], &subs);
        let mut simd_dst = vec![0i16; h];
        unsafe { kernel(&mut simd_dst, &src, &weights, h, &[], &subs); }
        assert_eq!(scalar_dst, simd_dst, "{} subs-only diverged at h={}", name, h);

        // Empty — identity copy of src.
        let mut simd_dst = vec![0i16; h];
        unsafe { kernel(&mut simd_dst, &src, &weights, h, &[], &[]); }
        assert_eq!(src, simd_dst, "{} empty-deltas should be identity at h={}", name, h);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_apply_deltas_avx512_matches_scalar() {
        // Skip when AVX-512 BW isn't available on the host — test
        // becomes a no-op rather than a false failure.
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            eprintln!("apply_deltas_avx512 test: AVX-512 unavailable on this host, skipping");
            return;
        }
        for &h in &[320usize, 512, 768, 1024] {
            check_apply_deltas_kernel(h, 0xc0da_d317_a5_0512, apply_deltas_avx512, "apply_deltas_avx512");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_apply_deltas_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("apply_deltas_avx2 test: AVX2 unavailable on this host, skipping");
            return;
        }
        for &h in &[320usize, 512, 768, 1024] {
            check_apply_deltas_kernel(h, 0xc0da_d317_a5_0002, apply_deltas_avx2, "apply_deltas_avx2");
        }
    }

    /// Shared body for the x86 add_weight_rows SIMD-vs-scalar tests.
    #[cfg(target_arch = "x86_64")]
    fn check_add_weight_rows_kernel(
        h: usize,
        seed: u64,
        kernel: unsafe fn(&mut [i16], &[i8], usize, &[usize]),
        name: &str,
    ) {
        let n_features = 32;
        let mut r = rng(seed ^ h as u64);

        let mut weights = vec![0i8; n_features * h];
        for w in weights.iter_mut() { *w = (r() % 256) as i8; }

        let indices = [0usize, 3, 7, 11, 15, 19, 23, 27, 31];

        let mut scalar_dst = vec![0i16; h];
        for v in scalar_dst.iter_mut() { *v = (r() as i32 as i16).rem_euclid(501) - 250; }
        let mut simd_dst = scalar_dst.clone();

        for &idx in &indices {
            let base = idx * h;
            for j in 0..h { scalar_dst[j] += weights[base + j] as i16; }
        }
        unsafe { kernel(&mut simd_dst, &weights, h, &indices); }
        assert_eq!(scalar_dst, simd_dst, "{} diverged at h={}", name, h);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_add_weight_rows_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            eprintln!("add_weight_rows_avx512 test: AVX-512 unavailable, skipping");
            return;
        }
        for &h in &[320usize, 512, 768, 1024] {
            check_add_weight_rows_kernel(h, 0xc0da_add1_a5_0512, add_weight_rows_avx512, "add_weight_rows_avx512");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_add_weight_rows_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("add_weight_rows_avx2 test: AVX2 unavailable, skipping");
            return;
        }
        for &h in &[320usize, 512, 768, 1024] {
            check_add_weight_rows_kernel(h, 0xc0da_add1_a5_0002, add_weight_rows_avx2, "add_weight_rows_avx2");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_apply_deltas_neon_matches_scalar() {
        let h = 768;
        let n_threats = 64;
        let mut r = rng(0xc0da_d317_a5_0002);

        let mut weights = vec![0i8; n_threats * h];
        for w in weights.iter_mut() { *w = (r() % 256) as i8; }

        let mut src = vec![0i16; h];
        for v in src.iter_mut() { *v = (r() as i32 as i16).rem_euclid(2001) - 1000; }

        // Mixed adds+subs — covers the paired inner loop.
        let adds = [3usize, 8, 21, 40];
        let subs = [5usize, 12, 30, 55, 63];
        let mut scalar_dst = vec![0i16; h];
        apply_deltas_scalar_ref(&mut scalar_dst, &src, &weights, h, &adds, &subs);
        let mut neon_dst = vec![0i16; h];
        unsafe { apply_deltas_neon(&mut neon_dst, &src, &weights, h, &adds, &subs); }
        assert_eq!(scalar_dst, neon_dst, "apply_deltas_neon mixed diverged");

        // Adds-only (exercises the post-paired tail loop for adds).
        let mut scalar_dst = vec![0i16; h];
        apply_deltas_scalar_ref(&mut scalar_dst, &src, &weights, h, &adds, &[]);
        let mut neon_dst = vec![0i16; h];
        unsafe { apply_deltas_neon(&mut neon_dst, &src, &weights, h, &adds, &[]); }
        assert_eq!(scalar_dst, neon_dst, "apply_deltas_neon adds-only diverged");

        // Subs-only (exercises the post-paired tail loop for subs).
        let mut scalar_dst = vec![0i16; h];
        apply_deltas_scalar_ref(&mut scalar_dst, &src, &weights, h, &[], &subs);
        let mut neon_dst = vec![0i16; h];
        unsafe { apply_deltas_neon(&mut neon_dst, &src, &weights, h, &[], &subs); }
        assert_eq!(scalar_dst, neon_dst, "apply_deltas_neon subs-only diverged");

        // Empty deltas — identity copy.
        let mut neon_dst = vec![0i16; h];
        unsafe { apply_deltas_neon(&mut neon_dst, &src, &weights, h, &[], &[]); }
        assert_eq!(src, neon_dst, "apply_deltas_neon empty-deltas should be identity");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_add_weight_rows_neon_matches_scalar() {
        let h = 768;
        let n_features = 32;
        let mut r = rng(0xc0da_add1_a5_0004);

        let mut weights = vec![0i8; n_features * h];
        for w in weights.iter_mut() { *w = (r() % 256) as i8; }

        let indices = [0usize, 3, 7, 11, 15, 19, 23, 27, 31];

        let mut scalar_dst = vec![0i16; h];
        for v in scalar_dst.iter_mut() { *v = (r() as i32 as i16).rem_euclid(501) - 250; }
        let mut neon_dst = scalar_dst.clone();

        for &idx in &indices {
            let base = idx * h;
            for j in 0..h { scalar_dst[j] += weights[base + j] as i16; }
        }
        unsafe { add_weight_rows_neon(&mut neon_dst, &weights, h, &indices); }
        assert_eq!(scalar_dst, neon_dst, "add_weight_rows_neon diverged");
    }

    #[test]
    fn test_init_threats() {
        crate::init();
        let total = num_threat_features();
        // the production threat-feature layout has 66,864 — we should match
        assert!(total > 60000, "Expected >60K threat features, got {}", total);
        assert!(total < 70000, "Expected <70K threat features, got {}", total);
        eprintln!("Total threat features: {}", total);
    }

    #[test]
    fn test_threat_index_basic() {
        crate::init();

        // White knight on c3 attacks black pawn on d5
        let wn = colored_piece(WHITE, KNIGHT);
        let bp = colored_piece(BLACK, PAWN);
        let idx = threat_index(wn, 18, bp, 35, false, WHITE); // c3=18, d5=35
        assert!(idx >= 0, "WN c3 × BP d5 should be a valid threat, got {}", idx);

        // Same threat from black's perspective should give different index
        let idx_black = threat_index(wn, 18, bp, 35, false, BLACK);
        assert!(idx_black >= 0, "Should be valid from black POV too");
        assert_ne!(idx, idx_black, "Different POV should give different index");
    }

    #[test]
    fn test_excluded_pairs() {
        crate::init();

        // Pawn attacks bishop: excluded (PIECE_INTERACTION_MAP[0][2] = -1)
        let wp = colored_piece(WHITE, PAWN);
        let bb = colored_piece(BLACK, BISHOP);
        let idx = threat_index(wp, 28, bb, 35, false, WHITE); // e4 → d5
        assert!(idx < 0, "Pawn×Bishop should be excluded, got {}", idx);

        // King attacks queen: excluded (PIECE_INTERACTION_MAP[5][4] = -1)
        let wk = colored_piece(WHITE, KING);
        let bq = colored_piece(BLACK, QUEEN);
        let idx = threat_index(wk, 4, bq, 5, false, WHITE);
        assert!(idx < 0, "King×Queen should be excluded, got {}", idx);
    }

    #[test]
    fn test_mirroring() {
        crate::init();

        // Same attack, mirrored vs not, should differ
        let wn = colored_piece(WHITE, KNIGHT);
        let bp = colored_piece(BLACK, PAWN);
        let idx_normal = threat_index(wn, 18, bp, 35, false, WHITE);
        let idx_mirror = threat_index(wn, 18, bp, 35, true, WHITE);
        assert_ne!(idx_normal, idx_mirror, "Mirrored should differ");
    }

    #[test]
    fn test_enumerate_startpos() {
        crate::init();

        // Standard starting position — count active threat features
        // In startpos, pieces attack each other across the board
        // Pawns attack nothing occupied, knights attack nothing, etc.
        // Only threats should be from pawns that are diagonal to opposing pawns (none in startpos)
        // and any other piece attacking an occupied square

        // Simplified: just verify we get a reasonable count
        let pieces_bb: [Bitboard; 6] = [
            0x00FF00000000FF00, // pawns
            0x4200000000000042, // knights
            0x2400000000000024, // bishops
            0x8100000000000081, // rooks
            0x0800000000000008, // queens
            0x1000000000000010, // kings
        ];
        let colors_bb: [Bitboard; 2] = [
            0x000000000000FFFF, // white
            0xFFFF000000000000, // black
        ];
        let occ = colors_bb[0] | colors_bb[1];

        // Build mailbox from bitboards
        let mut mailbox = [NO_PIECE_TYPE; 64];
        for sq in 0..64u32 {
            let bit = 1u64 << sq;
            if occ & bit == 0 { continue; }
            for pt in 0..6u8 {
                if pieces_bb[pt as usize] & bit != 0 {
                    mailbox[sq as usize] = pt;
                    break;
                }
            }
        }

        let mut count = 0;
        enumerate_threats(
            &pieces_bb, &colors_bb, &mailbox,
            occ, WHITE, false,
            |_idx| { count += 1; },
        );
        eprintln!("Startpos threat count (white POV): {}", count);
        // In startpos, pieces behind pawns don't attack much occupied territory.
        // But knights on b1/g1 attack no occupied squares, etc.
        // Expect a small number — mostly cross-pawn structure.
        assert!(count > 0, "Should have some threats in startpos");
        assert!(count < 200, "Shouldn't have too many threats in startpos");
    }

    #[test]
    fn test_bench_threat_enumeration() {
        crate::init();

        // Test positions: startpos + several middlegame/endgame positions
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "2r3k1/pp3ppp/2n1b3/3pP3/3P4/2NB4/PP3PPP/R4RK1 w - - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];

        let mut total_threats = 0usize;
        let mut total_positions = 0usize;

        for fen in &fens {
            let mut board = crate::board::Board::new();
            board.set_fen(fen);
            let occ = board.colors[0] | board.colors[1];
            let king_sq = (board.pieces[KING as usize] & board.colors[WHITE as usize]).trailing_zeros();
            let mirrored = (king_sq % 8) >= 4;

            let mut count = 0;
            enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, WHITE, mirrored,
                |_idx| { count += 1; },
            );
            eprintln!("  {} → {} threats", fen, count);
            total_threats += count;
            total_positions += 1;
        }

        eprintln!("Average threats per position: {}", total_threats / total_positions);

        // Benchmark: enumerate threats 100K times on the complex middlegame position
        let mut board = crate::board::Board::new();
        board.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        let occ = board.colors[0] | board.colors[1];
        let king_sq = (board.pieces[KING as usize] & board.colors[WHITE as usize]).trailing_zeros();
        let mirrored = (king_sq % 8) >= 4;

        let iterations = 100_000;
        let start = std::time::Instant::now();
        let mut total = 0usize;
        for _ in 0..iterations {
            enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, WHITE, mirrored,
                |_idx| { total += 1; },
            );
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() / iterations as u128;
        let calls_per_sec = if elapsed.as_secs_f64() > 0.0 {
            iterations as f64 / elapsed.as_secs_f64()
        } else { 0.0 };
        eprintln!("Threat enumeration benchmark (kiwipete, {}x):", iterations);
        eprintln!("  Total time: {:?}", elapsed);
        eprintln!("  Per call: {} ns ({:.0} K calls/sec)", per_call_ns, calls_per_sec / 1000.0);
        eprintln!("  Threats per call: {}", total / iterations);

        // Sanity: should complete in reasonable time
        assert!(elapsed.as_secs() < 10, "Benchmark took too long: {:?}", elapsed);
    }

    /// THE gate for lazy threat-delta generation. `replay_move_deltas`
    /// duplicates the mutation-and-emit sequence inside `Board::make_move`, so
    /// the two can drift apart silently — and a divergence would not crash, it
    /// would quietly feed a wrong accumulator into every eval downstream.
    ///
    /// Walks random legal games from varied positions and, for every single
    /// move, checks all three properties the lazy scheme depends on:
    ///   1. `undo_move_state` turns the post-move piece state back into exactly
    ///      the pre-move one (the inverse is exact);
    ///   2. replaying forward from that recovered state reproduces `make_move`'s
    ///      deltas EXACTLY — same values, same order, same length;
    ///   3. replaying forward leaves the piece state where `make_move` left it.
    ///
    /// Property 2 is the one that matters. Order is asserted, not just the
    /// multiset: each emit observes a different intermediate board, so an
    /// ordering difference means the sequence was mirrored wrongly even if the
    /// set happens to match on these positions.
    #[test]
    fn lazy_deltas_match_eager_generation() {
        use crate::board::Board;
        use crate::movegen::generate_legal_moves;
        crate::init();
        let _space = FEATURE_SPACE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Same corpus the threat-accumulator fuzz uses: opening, kiwipete,
        // slider-heavy midgame, pawn endgame, and a promotion testbed.
        const START_FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1",
            // EP is immediately available here (exf6). Uniform random play
            // essentially never reaches an en-passant position on its own —
            // the first version of this test ran 5000+ moves without one — so
            // the corpus seeds it directly as well as biasing toward it below.
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "rnbqkbnr/pppp1ppp/8/8/3PpP2/8/PPP1P1PP/RNBQKBNR b KQkq f3 0 3",
        ];

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            *state = x; x
        }

        let mut checked = 0usize;
        let mut promos = 0usize;
        let mut eps = 0usize;
        let mut castles = 0usize;

        for (fen_idx, fen) in START_FENS.iter().enumerate() {
            for game in 0..20 {
                let seed = 0x51ED_2701u32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((game as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };

                let mut board = Board::new();
                board.set_fen(fen);
                board.generate_threat_deltas = true;

                for _ply in 0..120 {
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    // Bias one move in four toward a special move when one is
                    // legal. Promotions, castles and EP are exactly the cases
                    // where `replay_move_deltas` does something other than
                    // "remove victim, shift piece", so leaving them to chance
                    // would leave the interesting half of the function unproven.
                    let mut special: Vec<Move> = Vec::new();
                    for i in 0..legal.len {
                        let m = legal.get(i);
                        if is_promotion(m)
                            || move_flags(m) == FLAG_EN_PASSANT
                            || move_flags(m) == FLAG_CASTLE
                        {
                            special.push(m);
                        }
                    }
                    let mv = if !special.is_empty() && next_u32(&mut rng) % 4 == 0 {
                        special[(next_u32(&mut rng) as usize) % special.len()]
                    } else {
                        legal.get((next_u32(&mut rng) as usize) % legal.len)
                    };

                    let us = board.side_to_move;
                    let pre = PieceState::from_board(&board);
                    if !board.make_move(mv) { break; }

                    let eager: Vec<RawThreatDelta> = board.threat_deltas.clone();
                    let captured = board.undo_stack.last().unwrap().captured;
                    let post = PieceState::from_board(&board);

                    // 1. inverse is exact
                    let mut walked = post;
                    undo_move_state(&mut walked, us, mv, captured);
                    assert_eq!(walked.pieces, pre.pieces, "pieces mismatch after undo, fen {fen_idx} game {game} mv {mv:#06x}");
                    assert_eq!(walked.colors, pre.colors, "colors mismatch after undo, fen {fen_idx} game {game} mv {mv:#06x}");
                    assert_eq!(walked.mailbox, pre.mailbox, "mailbox mismatch after undo, fen {fen_idx} game {game} mv {mv:#06x}");

                    // 2. forward replay from the recovered state == eager deltas
                    let mut lazy = Vec::new();
                    replay_move_deltas(&mut walked, us, mv, captured, &mut lazy);
                    assert_eq!(lazy.len(), eager.len(),
                        "delta COUNT differs (lazy {} vs eager {}), fen {fen_idx} game {game} mv {mv:#06x}",
                        lazy.len(), eager.len());
                    for (i, (l, e)) in lazy.iter().zip(eager.iter()).enumerate() {
                        assert_eq!(l.0, e.0,
                            "delta {i} differs, fen {fen_idx} game {game} mv {mv:#06x}: \
                             lazy(att={} from={} vic={} to={} add={}) \
                             eager(att={} from={} vic={} to={} add={})",
                            l.attacker_cp(), l.from_sq(), l.victim_cp(), l.to_sq(), l.add(),
                            e.attacker_cp(), e.from_sq(), e.victim_cp(), e.to_sq(), e.add());
                    }

                    // 3. forward replay lands on the real post-move state
                    assert_eq!(walked.pieces, post.pieces, "replay end-state pieces, fen {fen_idx} game {game}");
                    assert_eq!(walked.mailbox, post.mailbox, "replay end-state mailbox, fen {fen_idx} game {game}");

                    checked += 1;
                    if is_promotion(mv) { promos += 1; }
                    if move_flags(mv) == FLAG_EN_PASSANT { eps += 1; }
                    if move_flags(mv) == FLAG_CASTLE { castles += 1; }
                }
            }
        }

        // The corpus must actually reach the special cases, or properties 1-3
        // are only proven for quiet moves and plain captures.
        assert!(checked > 5000, "too few moves checked: {checked}");
        assert!(promos > 0, "corpus never promoted — promotion path unproven");
        assert!(eps > 0, "corpus never played en passant — EP path unproven");
        assert!(castles > 0, "corpus never castled — castling path unproven");
        eprintln!("lazy==eager over {checked} moves ({promos} promotions, {eps} EP, {castles} castles)");
    }
}

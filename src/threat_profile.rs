/// Lightweight profiling counters for threat feature performance.
/// Measures time spent in each phase of threat processing.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub static MAKE_MOVE_DELTA_NS: AtomicU64 = AtomicU64::new(0);
pub static MAKE_MOVE_DELTA_COUNT: AtomicU64 = AtomicU64::new(0);
pub static EVAL_INCR_NS: AtomicU64 = AtomicU64::new(0);
pub static EVAL_INCR_COUNT: AtomicU64 = AtomicU64::new(0);
pub static EVAL_FULL_NS: AtomicU64 = AtomicU64::new(0);
pub static EVAL_FULL_COUNT: AtomicU64 = AtomicU64::new(0);
pub static EVAL_COPY_NS: AtomicU64 = AtomicU64::new(0);
pub static EVAL_COPY_COUNT: AtomicU64 = AtomicU64::new(0);
pub static FORWARD_COMBINE_NS: AtomicU64 = AtomicU64::new(0);
pub static FORWARD_COMBINE_COUNT: AtomicU64 = AtomicU64::new(0);
pub static STORE_DELTAS_NS: AtomicU64 = AtomicU64::new(0);
pub static STORE_DELTAS_COUNT: AtomicU64 = AtomicU64::new(0);
pub static DELTAS_PER_MOVE: AtomicU64 = AtomicU64::new(0);

pub fn reset() {
    MAKE_MOVE_DELTA_NS.store(0, Ordering::Relaxed);
    MAKE_MOVE_DELTA_COUNT.store(0, Ordering::Relaxed);
    EVAL_INCR_NS.store(0, Ordering::Relaxed);
    EVAL_INCR_COUNT.store(0, Ordering::Relaxed);
    EVAL_FULL_NS.store(0, Ordering::Relaxed);
    EVAL_FULL_COUNT.store(0, Ordering::Relaxed);
    EVAL_COPY_NS.store(0, Ordering::Relaxed);
    EVAL_COPY_COUNT.store(0, Ordering::Relaxed);
    FORWARD_COMBINE_NS.store(0, Ordering::Relaxed);
    FORWARD_COMBINE_COUNT.store(0, Ordering::Relaxed);
    STORE_DELTAS_NS.store(0, Ordering::Relaxed);
    STORE_DELTAS_COUNT.store(0, Ordering::Relaxed);
    DELTAS_PER_MOVE.store(0, Ordering::Relaxed);
}

pub fn print_summary(total_nodes: u64) {
    let mm = MAKE_MOVE_DELTA_NS.load(Ordering::Relaxed);
    let mm_c = MAKE_MOVE_DELTA_COUNT.load(Ordering::Relaxed);
    let ei = EVAL_INCR_NS.load(Ordering::Relaxed);
    let ei_c = EVAL_INCR_COUNT.load(Ordering::Relaxed);
    let ef = EVAL_FULL_NS.load(Ordering::Relaxed);
    let ef_c = EVAL_FULL_COUNT.load(Ordering::Relaxed);
    let ec = EVAL_COPY_NS.load(Ordering::Relaxed);
    let ec_c = EVAL_COPY_COUNT.load(Ordering::Relaxed);
    let fc = FORWARD_COMBINE_NS.load(Ordering::Relaxed);
    let fc_c = FORWARD_COMBINE_COUNT.load(Ordering::Relaxed);
    let sd = STORE_DELTAS_NS.load(Ordering::Relaxed);
    let sd_c = STORE_DELTAS_COUNT.load(Ordering::Relaxed);
    let dpm = DELTAS_PER_MOVE.load(Ordering::Relaxed);

    let total_eval = ei_c + ef_c + ec_c;
    let total_ns = mm + ei + ef + ec + fc + sd;

    eprintln!("=== Threat Profile ===");
    eprintln!("  make_move deltas:  {:>8} calls, {:>10} ns total, {:>6} ns/call, {:>5.1} per node",
        mm_c, mm, if mm_c > 0 { mm / mm_c } else { 0 }, mm as f64 / total_nodes as f64);
    eprintln!("  store_deltas:      {:>8} calls, {:>10} ns total, {:>6} ns/call",
        sd_c, sd, if sd_c > 0 { sd / sd_c } else { 0 });
    eprintln!("  eval incremental:  {:>8} calls, {:>10} ns total, {:>6} ns/call",
        ei_c, ei, if ei_c > 0 { ei / ei_c } else { 0 });
    eprintln!("  eval full recomp:  {:>8} calls, {:>10} ns total, {:>6} ns/call",
        ef_c, ef, if ef_c > 0 { ef / ef_c } else { 0 });
    eprintln!("  eval copy (null):  {:>8} calls, {:>10} ns total, {:>6} ns/call",
        ec_c, ec, if ec_c > 0 { ec / ec_c } else { 0 });
    eprintln!("  forward combine:   {:>8} calls, {:>10} ns total, {:>6} ns/call",
        fc_c, fc, if fc_c > 0 { fc / fc_c } else { 0 });
    eprintln!("  avg deltas/move:   {:.1}", if mm_c > 0 { dpm as f64 / mm_c as f64 } else { 0.0 });
    eprintln!("  incr/full/copy:    {}/{}/{} ({:.0}%/{:.0}%/{:.0}%)",
        ei_c, ef_c, ec_c,
        if total_eval > 0 { ei_c as f64 / total_eval as f64 * 100.0 } else { 0.0 },
        if total_eval > 0 { ef_c as f64 / total_eval as f64 * 100.0 } else { 0.0 },
        if total_eval > 0 { ec_c as f64 / total_eval as f64 * 100.0 } else { 0.0 });
    eprintln!("  total threat ns:   {} ({:.1} ns/node)", total_ns, total_ns as f64 / total_nodes as f64);
}

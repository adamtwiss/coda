//! coda-filterstats — measure the ACTUAL accept/drop rate of each training
//! filter (quality, fenskip, soft-ply, piece-count rebalance) over a real T80
//! binpack, by replaying the EXACT filter chain from the bullet
//! `coda_v9_768_threats` example (same SplitMix64 hashes, same adaptive pc
//! state). Purpose: disentangle "coverage" from "reshaping". A filter that
//! barely changes total acceptance cannot be helping via coverage/diversity;
//! one that changes it a lot is coverage-confounded. fenskip is the calibrated
//! pure-coverage ruler (uniform 50% drop), so each filter's marginal drop
//! converts directly to a coverage delta.
//!
//!   coda-filterstats -i test80-2024-01-...binpack --max 8000000 --seed 42

use std::fs::File;

use clap::Parser;
use sfbinpack::CompressedTrainingDataEntryReader;
use sfbinpack::chess::color::Color;
use sfbinpack::chess::piecetype::PieceType;
use sfbinpack::chess::r#move::MoveType;

#[derive(Parser)]
#[command(about = "Measure per-filter accept/drop rates over a binpack (coverage vs reshaping probe)")]
struct Args {
    /// Input binpack
    #[arg(short, long)]
    input: String,
    /// Max entries to scan (cap for speed; pc converges well under 1M)
    #[arg(long, default_value_t = 8_000_000)]
    max: u64,
    /// --seed used in the training runs (drives all three skip hashes)
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// soft-ply peak ply (accept reaches 1.0 here); f25 = 28
    #[arg(long, default_value_t = 28)]
    soft_early_ply: u16,
    /// soft-ply floor accept at ply 16; f25 = 0.25
    #[arg(long, default_value_t = 0.25)]
    soft_early_floor: f64,
    /// If >0, run the fenskip WRAP-COVERAGE simulation instead of the accept
    /// table. Over K simulated passes (wrap counts 0..K) measure how much of the
    /// quality-passing set is kept at least once. OLD (fixed partition, salt=0)
    /// stays flat at the keep rate; NEW (salt=wrap, the fix) climbs toward 100%.
    #[arg(long, default_value_t = 0)]
    wrap_passes: u64,
}

const HARD_PLY_CUT: u16 = 16;
const PLY_LUT_LEN: usize = 512;

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Single-threaded port of the example's thread-local `PcRebalState`.
struct PcState {
    alpha: f64,
    hist: [f64; 33],
    total: f64,
    step: u64,
}
impl PcState {
    fn new() -> Self {
        PcState { alpha: 1.0, hist: [0.0; 33], total: 0.0, step: 0 }
    }
    fn accept(&mut self, pc: usize, target: &[f64; 33], target_total: f64) -> f64 {
        const MAX_PC_SKIP_RATE: f64 = 0.975;
        self.hist[pc] += 1.0;
        self.total += 1.0;
        self.step += 1;
        let upd = matches!(self.step, 100 | 500 | 1000 | 2500 | 5000)
            || (self.step > 5000 && self.step % 10000 == 0);
        if upd {
            let mut min_ratio = f64::INFINITY;
            let mut found = false;
            for i in 0..33 {
                if target[i] > 0.0 && self.hist[i] > 0.0 {
                    let r = (self.total * target[i]) / (target_total * self.hist[i]);
                    if r < min_ratio {
                        min_ratio = r;
                        found = true;
                    }
                }
            }
            if found && min_ratio > 0.0 {
                self.alpha = (1.0 - MAX_PC_SKIP_RATE) / min_ratio;
            }
            if self.step >= 10000 && self.step % 10000 == 0 {
                for i in 0..33 {
                    self.hist[i] *= 0.5;
                }
                self.total *= 0.5;
            }
        }
        let mut accept = 0.0;
        if target[pc] > 0.0 && self.hist[pc] > 0.0 {
            let r = (self.total * target[pc]) / (target_total * self.hist[pc]);
            accept = self.alpha * r;
        }
        accept.clamp(0.0, 1.0)
    }
}

fn main() {
    let a = Args::parse();

    // soft-ply accept LUT (legacy linear floor mode: Y0@ply16 -> 1.0@N)
    let mut accept_lut = [1.0f64; PLY_LUT_LEN];
    if a.soft_early_ply > HARD_PLY_CUT {
        let n = a.soft_early_ply as f64;
        for (i, slot) in accept_lut.iter_mut().enumerate().take(a.soft_early_ply as usize) {
            let ply = i as f64;
            *slot = if ply <= HARD_PLY_CUT as f64 {
                a.soft_early_floor
            } else {
                let t = (ply - HARD_PLY_CUT as f64) / (n - HARD_PLY_CUT as f64);
                a.soft_early_floor + (1.0 - a.soft_early_floor) * t
            };
        }
    }

    // pc target spline (SF threats control points, Hermite through 0/8/16/24/32)
    let pc_y = [-0.20f64, 0.45, 1.0, 0.95, 0.75];
    let mut target = [0.0f64; 33];
    let mut target_total = 0.0f64;
    {
        let pc_weight = |pc: i32| -> f64 {
            let y = pc_y;
            let x = pc as f64;
            if x <= 0.0 {
                return y[0];
            }
            if x >= 32.0 {
                return y[4];
            }
            let mut i = (x / 8.0) as usize;
            if i > 3 {
                i = 3;
            }
            let x0 = i as f64 * 8.0;
            let t = (x - x0) / 8.0;
            let slope = |idx: usize| -> f64 {
                if idx == 0 {
                    y[1] - y[0]
                } else if idx == 4 {
                    y[4] - y[3]
                } else {
                    (y[idx + 1] - y[idx - 1]) / 2.0
                }
            };
            let (m0, m1) = (slope(i), slope(i + 1));
            let (t2, t3) = (t * t, t * t * t);
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;
            (h00 * y[i] + h10 * m0 + h01 * y[i + 1] + h11 * m1).max(0.0)
        };
        for i in 0..33 {
            target[i] = pc_weight(i as i32);
            target_total += target[i];
        }
    }

    let fen_skip_seed = a.seed ^ 0xC0DA_5C19_F00D_BEEFu64;
    let soft_seed = a.seed ^ 0x5EAF_EA12_B19B_0075u64;
    let pc_seed = a.seed ^ 0x9C5E_D11B_A1A0_C0D5u64;
    let fen_thresh: u32 = (0.5f32 * u32::MAX as f32) as u32;

    let file = File::open(&a.input).expect("open binpack");
    let mut reader = CompressedTrainingDataEntryReader::new(file).expect("binpack reader");

    // ---- fenskip wrap-coverage simulation (fail-fast behaviour check) ----
    // Replays the EXACT trainer hash (incl. the per-pass wrap salt) over one
    // file, computing for each quality position the first wrap on which it is
    // kept. OLD re-skips the same set every wrap (coverage flat); NEW exposes a
    // fresh subset each wrap (coverage climbs 50 -> 75 -> 87.5 -> ...).
    if a.wrap_passes > 0 {
        let k = a.wrap_passes as usize;
        let salt = |w: usize| (w as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let keeps = |white_bb: u64, black_bb: u64, ply: u16, score_u: u64, mv_bits: u64, s: u64| -> bool {
            let x = splitmix(
                white_bb
                    ^ black_bb.rotate_left(13)
                    ^ ((ply as u64) << 7)
                    ^ score_u.rotate_left(23)
                    ^ (mv_bits << 32)
                    ^ fen_skip_seed
                    ^ s,
            );
            ((x >> 32) as u32) >= fen_thresh // keep iff NOT below skip threshold
        };
        let mut first_kept = vec![0u64; k + 1]; // [k] = never kept across k passes
        let mut total_q: u64 = 0;
        let mut read2: u64 = 0;
        while reader.has_next() && read2 < a.max {
            let e = reader.next();
            read2 += 1;
            let stm = e.pos.side_to_move();
            if !(e.ply >= HARD_PLY_CUT
                && !e.pos.is_checked(stm)
                && e.score.unsigned_abs() <= 10000
                && e.mv.mtype() == MoveType::Normal
                && e.pos.piece_at(e.mv.to()).piece_type() == PieceType::None)
            {
                continue;
            }
            total_q += 1;
            let white_bb = e.pos.pieces_bb(Color::White).bits();
            let black_bb = e.pos.pieces_bb(Color::Black).bits();
            let mv_bits = ((e.mv.from().index() as u64) << 6) | (e.mv.to().index() as u64);
            let score_u = e.score as i64 as u64;
            let mut fw = k;
            for w in 0..k {
                if keeps(white_bb, black_bb, e.ply, score_u, mv_bits, salt(w)) {
                    fw = w;
                    break;
                }
            }
            first_kept[fw] += 1;
        }
        let q = total_q.max(1) as f64;
        // OLD ever-kept (salt always 0) == kept on wrap 0 == first_kept[0].
        let old_cov = 100.0 * first_kept[0] as f64 / q;
        println!("fenskip wrap-coverage — {} quality positions from {}", total_q, a.input);
        println!("seed {}  keep-rate {:.1}% / skip-rate {:.1}%\n", a.seed, old_cov, 100.0 - old_cov);
        println!("  passes   NEW unique-kept   OLD unique-kept");
        let mut cum = 0u64;
        for p in 1..=k {
            cum += first_kept[p - 1];
            println!("  {:>5}    {:>12.1}%   {:>12.1}%", p, 100.0 * cum as f64 / q, old_cov);
        }
        let never = 100.0 * first_kept[k] as f64 / q;
        println!(
            "\n  after {} passes:  NEW {:.1}% still never seen  |  OLD {:.1}% PERMANENTLY excluded",
            k, never, 100.0 - old_cov
        );
        return;
    }

    let mut read: u64 = 0;
    let mut quality: u64 = 0;
    let mut acc_fen: u64 = 0; // quality + fenskip
    let mut acc_f25: u64 = 0; // quality + fenskip + softply
    let mut acc_pc: u64 = 0; // quality + fenskip + pc
    let mut acc_f25pc: u64 = 0; // quality + fenskip + softply + pc
    let mut ply_hist = [0u64; PLY_LUT_LEN]; // of quality-passing positions
    let mut occ_hist = [0u64; 33];
    let mut pc_state = PcState::new();
    let mut pc_state_after_sp = PcState::new();

    while reader.has_next() && read < a.max {
        let e = reader.next();
        read += 1;

        // ---- quality filter ----
        let stm = e.pos.side_to_move();
        let q = e.ply >= HARD_PLY_CUT
            && !e.pos.is_checked(stm)
            && e.score.unsigned_abs() <= 10000
            && e.mv.mtype() == MoveType::Normal
            && e.pos.piece_at(e.mv.to()).piece_type() == PieceType::None;
        if !q {
            continue;
        }
        quality += 1;
        let ply = e.ply as usize;
        if ply < PLY_LUT_LEN {
            ply_hist[ply] += 1;
        }
        let pc = (e.pos.occupied().count() as usize).min(32);
        occ_hist[pc] += 1;

        let white_bb = e.pos.pieces_bb(Color::White).bits();
        let black_bb = e.pos.pieces_bb(Color::Black).bits();
        let mv_bits = ((e.mv.from().index() as u64) << 6) | (e.mv.to().index() as u64);
        let score_u = e.score as i64 as u64;

        // ---- fenskip ----
        let x = splitmix(
            white_bb
                ^ black_bb.rotate_left(13)
                ^ ((e.ply as u64) << 7)
                ^ score_u.rotate_left(23)
                ^ (mv_bits << 32)
                ^ fen_skip_seed,
        );
        let fen_ok = !(((x >> 32) as u32) < fen_thresh);
        if !fen_ok {
            continue;
        }
        acc_fen += 1;

        // ---- soft-ply ----
        let sp_ok = if a.soft_early_ply > 0 && ply < a.soft_early_ply as usize {
            let accept = accept_lut[ply];
            let thresh = (accept as f32 * u32::MAX as f32) as u32;
            let y = splitmix(
                white_bb
                    ^ black_bb.rotate_left(29)
                    ^ ((e.ply as u64) << 11)
                    ^ score_u.rotate_left(7)
                    ^ (mv_bits << 16)
                    ^ soft_seed,
            );
            ((y >> 32) as u32) < thresh
        } else {
            true
        };
        if sp_ok {
            acc_f25 += 1;
        }

        // ---- pc (config: quality+fenskip+pc) ----
        {
            let accept = pc_state.accept(pc, &target, target_total);
            let thresh = (accept * u32::MAX as f64) as u32;
            let z = splitmix(
                white_bb
                    ^ black_bb.rotate_left(17)
                    ^ ((e.ply as u64) << 5)
                    ^ score_u.rotate_left(41)
                    ^ (mv_bits << 24)
                    ^ pc_seed,
            );
            if ((z >> 32) as u32) < thresh {
                acc_pc += 1;
            }
        }

        // ---- f25pc (pc runs only on the post-soft-ply stream) ----
        if sp_ok {
            let accept = pc_state_after_sp.accept(pc, &target, target_total);
            let thresh = (accept * u32::MAX as f64) as u32;
            let z = splitmix(
                white_bb
                    ^ black_bb.rotate_left(17)
                    ^ ((e.ply as u64) << 5)
                    ^ score_u.rotate_left(41)
                    ^ (mv_bits << 24)
                    ^ pc_seed,
            );
            if ((z >> 32) as u32) < thresh {
                acc_f25pc += 1;
            }
        }
    }

    let qf = quality as f64;
    let fenf = acc_fen as f64;
    println!("scanned {read} entries from {}", a.input);
    println!("seed {} | soft-ply {}@floor{}", a.seed, a.soft_early_ply, a.soft_early_floor);
    println!();
    println!("                         accepted     of-quality   of-fenskip   coverage(ep, S200)");
    let report = |name: &str, acc: u64, ep_base: f64| {
        let of_q = acc as f64 / qf;
        let of_fen = acc as f64 / fenf;
        // baseline (fenskip) coverage anchor = 0.85 ep; coverage scales 1/accept
        let cov = ep_base * (acc_fen as f64 / acc.max(1) as f64);
        println!(
            "  {name:<22} {acc:>10}   {:>7.3}      {:>7.3}      {:>5.2}",
            of_q, of_fen, cov
        );
    };
    println!("  {:<22} {:>10}   {:>7.3}", "quality", quality, quality as f64 / read as f64);
    report("fenskip(0.5)", acc_fen, 0.85);
    report("f25 (softply)", acc_f25, 0.85);
    report("pc (rebal)", acc_pc, 0.85);
    report("f25pc", acc_f25pc, 0.85);

    // marginal drop rates (vs the fenskip stream each filter sits on)
    println!();
    println!("marginal drop vs fenskip-stream (the coverage-relevant number):");
    println!("  soft-ply:  {:.1}%", 100.0 * (1.0 - acc_f25 as f64 / fenf));
    println!("  pc:        {:.1}%", 100.0 * (1.0 - acc_pc as f64 / fenf));
    println!("  f25pc:     {:.1}%", 100.0 * (1.0 - acc_f25pc as f64 / fenf));

    // ply histogram of quality-passing positions in the soft-ply-touched band
    println!();
    println!("quality-passing ply distribution (the band soft-ply reshapes):");
    let mut band = 0u64;
    for p in HARD_PLY_CUT as usize..a.soft_early_ply as usize {
        band += ply_hist[p];
    }
    println!(
        "  plies [{},{}): {} = {:.1}% of quality-passing (soft-ply only touches these)",
        HARD_PLY_CUT,
        a.soft_early_ply,
        band,
        100.0 * band as f64 / qf
    );
}

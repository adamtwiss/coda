/// Threat feature computation for NNUE.
///
/// Encodes (attacker_piece, attacker_sq, victim_piece, victim_sq) relationships.
/// Each active threat on the board contributes one feature index into the threat
/// accumulator. Feature indices are perspective-relative with king-file mirroring.
///
/// Reference: Reckless engine (src/nnue/threats.rs) — same encoding pattern.
/// Total features: ~66,864 (depends on piece-pair filtering).

#[cfg(feature = "profile-threats")]
pub mod apply_stats {
    //! apply_threat_deltas delta-count histogram.
    //! Used to decide whether a long-tail of high-delta-count moves is
    //! worth capping/batching, or whether delta counts are uniform.
    use std::sync::atomic::{AtomicU64, Ordering};

    // Buckets: 0, 1-4, 5-8, 9-12, 13-16, 17-24, 25-32, 33+
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static TOTAL_DELTAS: AtomicU64 = AtomicU64::new(0);
    static B0: AtomicU64 = AtomicU64::new(0);
    static B1_4: AtomicU64 = AtomicU64::new(0);
    static B5_8: AtomicU64 = AtomicU64::new(0);
    static B9_12: AtomicU64 = AtomicU64::new(0);
    static B13_16: AtomicU64 = AtomicU64::new(0);
    static B17_24: AtomicU64 = AtomicU64::new(0);
    static B25_32: AtomicU64 = AtomicU64::new(0);
    static B33_PLUS: AtomicU64 = AtomicU64::new(0);

    #[inline(always)]
    pub fn record(n: usize) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        TOTAL_DELTAS.fetch_add(n as u64, Ordering::Relaxed);
        let bucket = match n {
            0 => &B0,
            1..=4 => &B1_4,
            5..=8 => &B5_8,
            9..=12 => &B9_12,
            13..=16 => &B13_16,
            17..=24 => &B17_24,
            25..=32 => &B25_32,
            _ => &B33_PLUS,
        };
        bucket.fetch_add(1, Ordering::Relaxed);
    }

    pub fn report() {
        let c = CALLS.load(Ordering::Relaxed);
        if c == 0 { eprintln!("apply_threat_deltas stats: 0 calls"); return; }
        let td = TOTAL_DELTAS.load(Ordering::Relaxed);
        let pct = |n: u64| -> f64 { 100.0 * n as f64 / c.max(1) as f64 };
        eprintln!("apply_threat_deltas: {} calls, total {} deltas, avg {:.2}",
            c, td, td as f64 / c.max(1) as f64);
        eprintln!("  0:       {:>10} ({:.1}%)", B0.load(Ordering::Relaxed), pct(B0.load(Ordering::Relaxed)));
        eprintln!("  1-4:     {:>10} ({:.1}%)", B1_4.load(Ordering::Relaxed), pct(B1_4.load(Ordering::Relaxed)));
        eprintln!("  5-8:     {:>10} ({:.1}%)", B5_8.load(Ordering::Relaxed), pct(B5_8.load(Ordering::Relaxed)));
        eprintln!("  9-12:    {:>10} ({:.1}%)", B9_12.load(Ordering::Relaxed), pct(B9_12.load(Ordering::Relaxed)));
        eprintln!("  13-16:   {:>10} ({:.1}%)", B13_16.load(Ordering::Relaxed), pct(B13_16.load(Ordering::Relaxed)));
        eprintln!("  17-24:   {:>10} ({:.1}%)", B17_24.load(Ordering::Relaxed), pct(B17_24.load(Ordering::Relaxed)));
        eprintln!("  25-32:   {:>10} ({:.1}%)", B25_32.load(Ordering::Relaxed), pct(B25_32.load(Ordering::Relaxed)));
        eprintln!("  33+:     {:>10} ({:.1}%)", B33_PLUS.load(Ordering::Relaxed), pct(B33_PLUS.load(Ordering::Relaxed)));
    }
}

#[cfg(feature = "profile-threats")]
pub mod thr_stats {
    //! Per-bench push_threats_for_piece section-level CPU counters.
    //! Gated behind `--features profile-threats` — zero release cost.
    //!
    //! Tracks cycle deltas for each logical block inside the hot
    //! function, plus delta counts. Used to decide whether a
    //! vectorised "direct only" port (Reckless-style) is worth
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
    }
}

use crate::attacks::*;
use crate::bitboard::*;
use crate::types::*;

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

/// Compact encoding of a piece-pair interaction.
/// Bit layout: bits 0..23 = base index, bit 30 = semi-excluded, bit 31 = excluded.
#[derive(Copy, Clone)]
struct PiecePair {
    inner: u32,
}

impl PiecePair {
    const fn new(excluded: bool, semi_excluded: bool, base: i32) -> Self {
        Self {
            inner: (((semi_excluded && !excluded) as u32) << 30)
                | ((excluded as u32) << 31)
                | ((base & 0x3FFFFFFF) as u32),
        }
    }

    /// Returns the base index with ordering correction.
    /// If semi-excluded and attacking_sq < attacked_sq, returns negative (skip).
    /// This keeps the pair where attacking_sq >= attacked_sq (matches Reckless).
    const fn base(self, attacking_sq: u32, attacked_sq: u32) -> i32 {
        let below = ((attacking_sq as u8) < (attacked_sq as u8)) as u32;
        ((self.inner.wrapping_add(below << 30)) & 0x80FFFFFF) as i32
    }
}

// Static lookup tables — initialised once at startup.
static mut PIECE_PAIR_LOOKUP: [[PiecePair; NUM_COLORED_PIECES]; NUM_COLORED_PIECES] =
    [[PiecePair { inner: 0 }; NUM_COLORED_PIECES]; NUM_COLORED_PIECES];
static mut PIECE_OFFSET_LOOKUP: [[i32; 64]; NUM_COLORED_PIECES] = [[0; 64]; NUM_COLORED_PIECES];
static mut ATTACK_INDEX_LOOKUP: [[[u8; 64]; 64]; NUM_COLORED_PIECES] = [[[0; 64]; 64]; NUM_COLORED_PIECES];

/// Total number of threat features. Set during init_threats().
static mut TOTAL_THREAT_FEATURES: usize = 0;

/// Get the total threat feature count (call after init_threats).
pub fn num_threat_features() -> usize {
    unsafe { TOTAL_THREAT_FEATURES }
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
        2 => bishop_attacks(sq, 0), // empty board
        3 => rook_attacks(sq, 0),
        4 => queen_attacks(sq, 0),
        5 => king_attacks(sq),
        _ => 0,
    }
}

/// Compute attack bitboard for a piece type on a square with given occupancy.
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

/// Initialise threat feature lookup tables. Must be called at startup.
pub fn init_threats() {
    let mut offset: i32 = 0;
    let mut piece_offset = [0i32; NUM_COLORED_PIECES];
    let mut offset_table = [0i32; NUM_COLORED_PIECES];

    // Build PIECE_OFFSET_LOOKUP: for each colored piece, for each square,
    // how many attack squares exist below this square (cumulative count).
    for color in 0..2usize {
        for pt in 0..6usize {
            let cp = color * 6 + pt;
            let mut count: i32 = 0;

            for sq in 0..64u32 {
                unsafe { PIECE_OFFSET_LOOKUP[cp][sq as usize] = count; }

                // Pawns on ranks 1 and 8 have no attacks (can't exist there)
                if pt == 0 && (sq < 8 || sq >= 56) {
                    continue;
                }

                let attacks = piece_attacks_empty(cp, sq);
                count += popcount(attacks) as i32;
            }

            piece_offset[cp] = count;
            offset_table[cp] = offset;
            offset += PIECE_TARGET_COUNT[pt] * count;
        }
    }

    unsafe { TOTAL_THREAT_FEATURES = offset as usize; }

    // Build PIECE_PAIR_LOOKUP: for each (attacker, victim) pair,
    // compute the base index and exclusion flags.
    for attacking_cp in 0..NUM_COLORED_PIECES {
        let attacking_pt = piece_type_of(attacking_cp);
        let attacking_color = color_of(attacking_cp);

        for attacked_cp in 0..NUM_COLORED_PIECES {
            let attacked_pt = piece_type_of(attacked_cp);
            let attacked_color = color_of(attacked_cp);

            let map = PIECE_INTERACTION_MAP[attacking_pt][attacked_pt];
            let base = offset_table[attacking_cp]
                + (attacked_color as i32 * (PIECE_TARGET_COUNT[attacking_pt] / 2) + map)
                    * piece_offset[attacking_cp];

            let enemy = attacking_color != attacked_color;
            let semi_excluded = attacking_pt == attacked_pt
                && (enemy || attacking_pt != 0); // pawn-pawn same color not semi-excluded
            let excluded = map < 0;

            unsafe {
                PIECE_PAIR_LOOKUP[attacking_cp][attacked_cp] =
                    PiecePair::new(excluded, semi_excluded, base);
            }
        }
    }

    // Build ATTACK_INDEX_LOOKUP: for each piece and source square,
    // the ray index of each target square within the attack set.
    for cp in 0..NUM_COLORED_PIECES {
        for from in 0..64u32 {
            let attacks = piece_attacks_empty(cp, from);

            for to in 0..64u32 {
                let below_mask = if to > 0 { (1u64 << to) - 1 } else { 0 };
                unsafe {
                    ATTACK_INDEX_LOOKUP[cp][from as usize][to as usize] =
                        popcount(below_mask & attacks) as u8;
                }
            }
        }
    }

    eprintln!("Threat features initialised: {} total", offset);
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
    // Remap piece colors relative to POV
    let attacking = if pov == BLACK {
        (attacker_cp + 6) % 12
    } else {
        attacker_cp
    };
    let attacked = if pov == BLACK {
        (victim_cp + 6) % 12
    } else {
        victim_cp
    };

    unsafe {
        let pair = PIECE_PAIR_LOOKUP[attacking][attacked];
        // Semi-exclusion uses PHYSICAL squares to match Bullet training.
        // Bullet decides once per pair using physical square ordering;
        // both perspectives see the same decision. Previously we used
        // perspective-flipped squares here, causing ~3-5% NTM feature
        // mismatch vs training.
        let base = pair.base(from, to);
        if base < 0 {
            return base; // excluded or semi-excluded pair
        }

        // Perspective flip for index computation only
        let flip = (7 * mirrored as u32) ^ (56 * pov as u32);
        let from_f = from ^ flip;
        let to_f = to ^ flip;

        base + PIECE_OFFSET_LOOKUP[attacking][from_f as usize]
            + ATTACK_INDEX_LOOKUP[attacking][from_f as usize][to_f as usize] as i32
    }
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

                // X-ray threats: for sliders, find the second piece on each ray
                // (the piece behind the directly attacked piece). Matches Bullet
                // training enumeration at ebdf398.
                if pt == BISHOP || pt == ROOK || pt == QUEEN {
                    // Check each ray direction using attack comparison
                    // For each directly attacked piece, see if removing it reveals another
                    let mut direct_targets = attacks & occ;
                    while direct_targets != 0 {
                        let blocker_sq = direct_targets.trailing_zeros();
                        direct_targets &= direct_targets - 1;

                        // Compute slider attacks without the blocking piece
                        let occ_without = occ & !(1u64 << blocker_sq);
                        let attacks_through = piece_attacks_occ(pt, color, sq, occ_without);

                        // Newly revealed squares: attacked without blocker but not with
                        let revealed = attacks_through & !attacks & occ_without;
                        if revealed == 0 { continue; }

                        // Take the closest revealed piece on the ray
                        // Direction: if slider < blocker, xray is above blocker
                        let xray_sq = if sq < blocker_sq {
                            let above = revealed & !((1u64 << (blocker_sq + 1)) - 1);
                            if above != 0 { above.trailing_zeros() } else { 64 }
                        } else {
                            let below = revealed & ((1u64 << blocker_sq) - 1);
                            if below != 0 { 63 - below.leading_zeros() } else { 64 }
                        };

                        if xray_sq < 64 {
                            let xpt = mailbox[xray_sq as usize];
                            if xpt < 6 {
                                let xcolor = if white_bb & (1u64 << xray_sq) != 0 { WHITE } else { BLACK };
                                let xcp = colored_piece(xcolor, xpt);
                                let idx = threat_index(cp, sq, xcp, xray_sq, mirrored, pov);
                                if idx >= 0 {
                                    callback(idx as usize);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Maximum threat deltas per ply.
pub const MAX_THREAT_DELTAS: usize = 128;

/// Packed threat delta (4 bytes, matching Reckless's ThreatDelta).
/// Layout: [attacker_cp:8][from_sq:8][victim_cp:8][to_sq:7][add:1]
#[derive(Copy, Clone)]
pub struct RawThreatDelta(u32);

impl RawThreatDelta {
    #[inline(always)]
    pub const fn new(attacker_cp: u8, from_sq: u8, victim_cp: u8, to_sq: u8, add: bool) -> Self {
        Self(attacker_cp as u32
            | (from_sq as u32) << 8
            | (victim_cp as u32) << 16
            | ((to_sq as u32) & 0x7F) << 24
            | if add { 1u32 << 31 } else { 0 })
    }

    pub const ZERO: Self = Self(0);

    #[inline(always)] pub fn attacker_cp(self) -> u8 { self.0 as u8 }
    #[inline(always)] pub fn from_sq(self) -> u8 { (self.0 >> 8) as u8 }
    #[inline(always)] pub fn victim_cp(self) -> u8 { (self.0 >> 16) as u8 }
    #[inline(always)] pub fn to_sq(self) -> u8 { ((self.0 >> 24) & 0x7F) as u8 }
    #[inline(always)] pub fn add(self) -> bool { self.0 & (1 << 31) != 0 }
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
    // This matches how Reckless handles it: occ ^ to_bb
    let occ_transit = occ ^ (1u64 << to);

    // Remove threats from old square
    push_threats_for_piece(deltas, pieces_bb, colors_bb, mailbox, occ_transit, white_bb, cp, piece_color, piece_type, from, false);
    // Add threats at new square
    push_threats_for_piece(deltas, pieces_bb, colors_bb, mailbox, occ_transit, white_bb, cp, piece_color, piece_type, to, true);
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

/// Core: compute all threat deltas for a piece on a square.
/// Matches Reckless's push_threats_single exactly:
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

    let my_attacks = piece_attacks_occ(piece_type, piece_color, square, occ);
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

    // 1b. X-ray threats FROM this piece, if it's a slider. For each
    // direct target (blocker), find the next occupant on the same ray
    // STRICTLY BEYOND the blocker. Previously recomputed full attacks
    // with the blocker removed per-iteration (one magic lookup each).
    // Now uses the precomputed RAY_EXTENSION table: one array read
    // replaces the per-blocker magic lookup.
    #[cfg(feature = "profile-threats")]
    let s1b_start = crate::threats::thr_stats::rdtsc();
    #[cfg(feature = "profile-threats")]
    let s1b_deltas_before = deltas.len() as u64;
    if piece_type == BISHOP || piece_type == ROOK || piece_type == QUEEN {
        let mut direct_targets = my_attacks & occ;
        while direct_targets != 0 {
            let blocker_sq = direct_targets.trailing_zeros();
            direct_targets &= direct_targets - 1;

            let extension = crate::bitboard::ray_extension(square, blocker_sq);
            let xray_candidates = extension & occ;
            if xray_candidates == 0 { continue; }

            // First occupant past the blocker on the same ray direction.
            let xray_sq = if square < blocker_sq {
                xray_candidates.trailing_zeros()
            } else {
                63 - xray_candidates.leading_zeros()
            };
            let xpt = mailbox[xray_sq as usize];
            if xpt >= 6 { continue; }
            let xcolor = if white_bb & (1u64 << xray_sq) != 0 { WHITE } else { BLACK };
            deltas.push(RawThreatDelta::new(
                cp as u8, square as u8,
                colored_piece(xcolor, xpt) as u8, xray_sq as u8, add,
            ));
        }
    }

    #[cfg(feature = "profile-threats")]
    crate::threats::thr_stats::record_section(
        1,
        crate::threats::thr_stats::rdtsc().wrapping_sub(s1b_start),
        deltas.len() as u64 - s1b_deltas_before,
    );

    // 2. Sliding pieces that see this square (Reckless pattern)
    // Compute rook/bishop attacks FROM this square to find which sliders can reach it
    #[cfg(feature = "profile-threats")]
    let s2_start = crate::threats::thr_stats::rdtsc();
    #[cfg(feature = "profile-threats")]
    let s2_deltas_before = deltas.len() as u64;
    let rook_att = rook_attacks(square, occ);
    let bishop_att = bishop_attacks(square, occ);
    let queen_att = rook_att | bishop_att;

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
    let ortho_ray_mask = rook_attacks(square, 0);
    let diag_ray_mask  = bishop_attacks(square, 0);
    let rays_from_sq_empty = ortho_ray_mask | diag_ray_mask;
    let past_first_region  = rays_from_sq_empty & !queen_att;
    let do_z_finding       = (occ & past_first_region) != 0;

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
        if do_z_finding {
            // Y = first occupant past `square` on slider_sq's ray through square.
            // The ray_extension table gives squares strictly beyond `square`
            // on the slider_sq→square ray direction; mask by occ and take the
            // first bit in the slider→square direction.
            //
            // Replaces two magic lookups (slider_att_through, slider_att_blocked)
            // and their difference/filter with a single table read.
            let y_candidates = crate::bitboard::ray_extension(slider_sq, square) & occ;
            if y_candidates != 0 {
                let y_sq = if slider_sq < square {
                    y_candidates.trailing_zeros()
                } else {
                    63 - y_candidates.leading_zeros()
                };

                // Z = first occupant past Y on the same ray, one hop further out.
                // Same table-driven technique.
                let z_candidates = crate::bitboard::ray_extension(slider_sq, y_sq) & occ;
                if z_candidates != 0 {
                    let z_sq = if slider_sq < square {
                        z_candidates.trailing_zeros()
                    } else {
                        63 - z_candidates.leading_zeros()
                    };
                    let zpt = mailbox[z_sq as usize];
                    if zpt < 6 {
                        let zcolor = if white_bb & (1u64 << z_sq) != 0 { WHITE } else { BLACK };
                        deltas.push(RawThreatDelta::new(
                            slider_cp as u8, slider_sq as u8,
                            colored_piece(zcolor, zpt) as u8, z_sq as u8,
                            !add,
                        ));
                    }
                }
            }
        }

        // The slider itself attacks/no longer attacks this square
        deltas.push(RawThreatDelta::new(slider_cp as u8, slider_sq as u8, cp as u8, square as u8, add));
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
    let ortho_candidates = (pieces_bb[ROOK as usize] | pieces_bb[QUEEN as usize]) & ortho_ray_mask & occ;
    let diag_candidates  = (pieces_bb[BISHOP as usize] | pieces_bb[QUEEN as usize]) & diag_ray_mask & occ;
    let mut candidates = ortho_candidates | diag_candidates;

    while candidates != 0 {
        let s_sq = candidates.trailing_zeros();
        candidates &= candidates - 1;

        // Count blockers strictly between S and sq. between() excludes
        // endpoints, so occ (not occ_rays) is the right mask — sq is not
        // in the between set.
        let between_mask = crate::bitboard::between(s_sq, square);
        let blockers_between = between_mask & occ;
        if blockers_between.count_ones() != 1 {
            // 0 → direct attacker (section 2 handles). 2+ → 2+ level x-ray
            // (not encoded). Neither emits a 2b delta.
            continue;
        }

        let s_pt = mailbox[s_sq as usize];
        // Set-membership already guarantees slider-type match for ray.
        // Defensive check retained for robustness; should never fail.
        if s_pt >= 6 { continue; }

        let s_color = if white_bb & (1u64 << s_sq) != 0 { WHITE } else { BLACK };
        let s_cp = colored_piece(s_color, s_pt);
        // (S, cp_at_sq, sq) x-ray feature appears/disappears with `add`.
        deltas.push(RawThreatDelta::new(
            s_cp as u8, s_sq as u8, cp as u8, square as u8, add,
        ));

        // W = first piece past sq on the ray from S, continuing in the
        // S→sq direction (away from S past sq). ray_extension(S, sq)
        // returns squares strictly beyond sq on that ray.
        // Direction: if S < sq we extend upward (pick lowest bit);
        //            if S > sq we extend downward (pick highest bit).
        let w_candidates_bb = crate::bitboard::ray_extension(s_sq, square) & occ;
        if w_candidates_bb != 0 {
            let w_sq = if s_sq < square {
                w_candidates_bb.trailing_zeros()
            } else {
                63 - w_candidates_bb.leading_zeros()
            };
            let w_pt = mailbox[w_sq as usize];
            if w_pt < 6 {
                let w_color = if white_bb & (1u64 << w_sq) != 0 { WHITE } else { BLACK };
                let w_cp = colored_piece(w_color, w_pt);
                deltas.push(RawThreatDelta::new(
                    s_cp as u8, s_sq as u8, w_cp as u8, w_sq as u8, !add,
                ));
            }
        }
    }

    #[cfg(feature = "profile-threats")]
    crate::threats::thr_stats::record_section(
        3,
        crate::threats::thr_stats::rdtsc().wrapping_sub(s2b_start),
        deltas.len() as u64 - s2b_deltas_before,
    );

    #[cfg(feature = "profile-threats")]
    let s3_start = crate::threats::thr_stats::rdtsc();
    #[cfg(feature = "profile-threats")]
    let s3_deltas_before = deltas.len() as u64;

    // 3. Non-sliding pieces that attack this square
    let black_pawns = pieces_bb[PAWN as usize] & colors_bb[BLACK as usize] & pawn_attacks(WHITE, square);
    let white_pawns = pieces_bb[PAWN as usize] & colors_bb[WHITE as usize] & pawn_attacks(BLACK, square);
    let knights = pieces_bb[KNIGHT as usize] & knight_attacks(square);
    let kings = pieces_bb[KING as usize] & king_attacks(square);

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

#[allow(dead_code)]
fn ray_between(_from: u32, _to: u32) -> Bitboard { 0 }
#[allow(dead_code)]
fn ray_beyond(_from: u32, _to: u32) -> Bitboard { 0 }

/// Compute threat deltas from post-move board state + undo info.
/// This reconstructs what changed without needing pre-move board access.
///
/// For each perspective (pov), computes adds/subs to the threat accumulator:
/// - Threats from/to the moved piece at old vs new square
/// - Threats from/to the captured piece (removed)
/// - Threats from sliders whose rays are affected by the from/to squares
/// - Threats from non-sliders that attack the from/to squares
///
/// `board` is the POST-move state. `from`, `to`, `moved_pt`, `moved_color` describe
/// the move that was just made. `captured_pt` is NO_PIECE_TYPE if no capture.
pub fn compute_move_deltas(
    deltas: &mut Vec<RawThreatDelta>,
    board_pieces: &[Bitboard; 6],
    board_colors: &[Bitboard; 2],
    board_mailbox: &[u8; 64],
    moved_color: Color,
    moved_pt: u8,
    from: u32,
    to: u32,
    captured_pt: u8,
    captured_color: Color,
) {
    let post_occ = board_colors[0] | board_colors[1];
    // Pre-move occupancy: piece was at `from` not `to`, captured piece was at `to` (for captures)
    let pre_occ = (post_occ | (1u64 << from)) & !(1u64 << to)
        | if captured_pt != NO_PIECE_TYPE { 1u64 << to } else { 0 };

    let moved_cp = colored_piece(moved_color, moved_pt);
    let white_bb_post = board_colors[WHITE as usize];

    // Helper: get victim info at a square in the POST-move board
    let victim_at_post = |sq: u32| -> Option<(u8, Color, usize)> {
        let pt = board_mailbox[sq as usize];
        if pt >= 6 { return None; }
        let color = if white_bb_post & (1u64 << sq) != 0 { WHITE } else { BLACK };
        Some((pt, color, colored_piece(color, pt)))
    };

    // 1. Remove threats FROM moved piece at old square (using pre-move occ)
    let old_attacks = piece_attacks_occ(moved_pt, moved_color, from, pre_occ);
    let mut old_targets = old_attacks & pre_occ & !(1u64 << from);
    while old_targets != 0 {
        let tsq = old_targets.trailing_zeros();
        old_targets &= old_targets - 1;
        // Target might have been the captured piece (at `to`) or a piece still on the board
        if tsq == to && captured_pt != NO_PIECE_TYPE {
            // Threat was to the captured piece
            let vcp = colored_piece(captured_color, captured_pt);
            deltas.push(RawThreatDelta::new(moved_cp as u8, from as u8, vcp as u8, tsq as u8, false));
        } else if let Some((_, _, vcp)) = victim_at_post(tsq) {
            deltas.push(RawThreatDelta::new(moved_cp as u8, from as u8, vcp as u8, tsq as u8, false));
        }
    }

    // 2. Add threats FROM moved piece at new square (using post-move occ)
    let new_attacks = piece_attacks_occ(moved_pt, moved_color, to, post_occ);
    let mut new_targets = new_attacks & post_occ & !(1u64 << to);
    while new_targets != 0 {
        let tsq = new_targets.trailing_zeros();
        new_targets &= new_targets - 1;
        if let Some((_, _, vcp)) = victim_at_post(tsq) {
            deltas.push(RawThreatDelta::new(moved_cp as u8, to as u8, vcp as u8, tsq as u8, true));
        }
    }

    // 3. Pieces that attacked the FROM square (they lose a threat)
    // Non-sliders: pawns, knights, kings
    let from_pawn_attackers_w = board_pieces[PAWN as usize] & board_colors[WHITE as usize] & pawn_attacks(BLACK, from);
    let from_pawn_attackers_b = board_pieces[PAWN as usize] & board_colors[BLACK as usize] & pawn_attacks(WHITE, from);
    let from_knight_attackers = board_pieces[KNIGHT as usize] & knight_attacks(from);
    let from_king_attackers = board_pieces[KING as usize] & king_attacks(from);
    let mut from_ns = (from_pawn_attackers_w | from_pawn_attackers_b | from_knight_attackers | from_king_attackers) & post_occ;
    // Remove the moved piece itself (it's no longer at from)
    from_ns &= !(1u64 << to); // moved piece is now at `to`, not relevant for `from` attackers
    while from_ns != 0 {
        let asq = from_ns.trailing_zeros();
        from_ns &= from_ns - 1;
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);
        // This piece used to attack the moved piece at `from` — remove that threat
        deltas.push(RawThreatDelta::new(acp as u8, asq as u8, moved_cp as u8, from as u8, false));
    }

    // Sliders that attacked through from (using pre-move occ to find them)
    let from_rook_att = rook_attacks(from, pre_occ);
    let from_bishop_att = bishop_attacks(from, pre_occ);
    let from_diag = (board_pieces[BISHOP as usize] | board_pieces[QUEEN as usize]) & from_bishop_att & post_occ;
    let from_orth = (board_pieces[ROOK as usize] | board_pieces[QUEEN as usize]) & from_rook_att & post_occ;
    let mut from_sliders = from_diag | from_orth;
    while from_sliders != 0 {
        let asq = from_sliders.trailing_zeros();
        from_sliders &= from_sliders - 1;
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);
        // Remove: this slider attacked the moved piece at `from`
        deltas.push(RawThreatDelta::new(acp as u8, asq as u8, moved_cp as u8, from as u8, false));
    }

    // 4. Pieces that attack the TO square (they gain a threat to the moved piece)
    let to_pawn_attackers_w = board_pieces[PAWN as usize] & board_colors[WHITE as usize] & pawn_attacks(BLACK, to);
    let to_pawn_attackers_b = board_pieces[PAWN as usize] & board_colors[BLACK as usize] & pawn_attacks(WHITE, to);
    let to_knight_attackers = board_pieces[KNIGHT as usize] & knight_attacks(to);
    let to_king_attackers = board_pieces[KING as usize] & king_attacks(to);
    let mut to_ns = (to_pawn_attackers_w | to_pawn_attackers_b | to_knight_attackers | to_king_attackers) & post_occ;
    to_ns &= !(1u64 << to); // exclude the moved piece attacking itself
    while to_ns != 0 {
        let asq = to_ns.trailing_zeros();
        to_ns &= to_ns - 1;
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);
        deltas.push(RawThreatDelta::new(acp as u8, asq as u8, moved_cp as u8, to as u8, true));
    }

    // Sliders that attack to (using post-move occ)
    let to_rook_att = rook_attacks(to, post_occ);
    let to_bishop_att = bishop_attacks(to, post_occ);
    let to_diag = (board_pieces[BISHOP as usize] | board_pieces[QUEEN as usize]) & to_bishop_att & post_occ;
    let to_orth = (board_pieces[ROOK as usize] | board_pieces[QUEEN as usize]) & to_rook_att & post_occ;
    let mut to_sliders = to_diag | to_orth;
    to_sliders &= !(1u64 << to); // exclude the moved piece
    while to_sliders != 0 {
        let asq = to_sliders.trailing_zeros();
        to_sliders &= to_sliders - 1;
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);
        deltas.push(RawThreatDelta::new(acp as u8, asq as u8, moved_cp as u8, to as u8, true));
    }

    // 5. Captured piece: remove all its threats
    if captured_pt != NO_PIECE_TYPE {
        let cap_cp = colored_piece(captured_color, captured_pt);
        // Remove threats FROM captured piece
        let cap_attacks = piece_attacks_occ(captured_pt, captured_color, to, pre_occ);
        let mut cap_targets = cap_attacks & pre_occ & !(1u64 << to);
        while cap_targets != 0 {
            let tsq = cap_targets.trailing_zeros();
            cap_targets &= cap_targets - 1;
            if tsq == from {
                // Was attacking the moved piece at its old square
                deltas.push(RawThreatDelta::new(cap_cp as u8, to as u8, moved_cp as u8, from as u8, false));
            } else if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta::new(cap_cp as u8, to as u8, vcp as u8, tsq as u8, false));
            }
        }
        // Remove threats TO captured piece (from non-sliders)
        let cap_pawn_w = board_pieces[PAWN as usize] & board_colors[WHITE as usize] & pawn_attacks(BLACK, to);
        let cap_pawn_b = board_pieces[PAWN as usize] & board_colors[BLACK as usize] & pawn_attacks(WHITE, to);
        let cap_knights = board_pieces[KNIGHT as usize] & knight_attacks(to);
        let cap_kings = board_pieces[KING as usize] & king_attacks(to);
        let mut cap_attackers = (cap_pawn_w | cap_pawn_b | cap_knights | cap_kings) & post_occ;
        cap_attackers &= !(1u64 << to);
        while cap_attackers != 0 {
            let asq = cap_attackers.trailing_zeros();
            cap_attackers &= cap_attackers - 1;
            let apt = board_mailbox[asq as usize];
            if apt >= 6 { continue; }
            let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
            let acp = colored_piece(acolor, apt);
            deltas.push(RawThreatDelta::new(acp as u8, asq as u8, cap_cp as u8, to as u8, false));
        }
        // Sliders that attacked captured piece
        let cap_rook_att = rook_attacks(to, pre_occ);
        let cap_bishop_att = bishop_attacks(to, pre_occ);
        let cap_diag = (board_pieces[BISHOP as usize] | board_pieces[QUEEN as usize]) & cap_bishop_att & post_occ;
        let cap_orth = (board_pieces[ROOK as usize] | board_pieces[QUEEN as usize]) & cap_rook_att & post_occ;
        let mut cap_sliders = cap_diag | cap_orth;
        cap_sliders &= !(1u64 << to);
        while cap_sliders != 0 {
            let asq = cap_sliders.trailing_zeros();
            cap_sliders &= cap_sliders - 1;
            let apt = board_mailbox[asq as usize];
            if apt >= 6 { continue; }
            let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
            let acp = colored_piece(acolor, apt);
            deltas.push(RawThreatDelta::new(acp as u8, asq as u8, cap_cp as u8, to as u8, false));
        }
    }

    // 6. X-ray discoveries: sliders that were blocked by the piece at `from` now
    // see through to new targets. Also, sliders that now see `to` may be blocked
    // by the piece that just arrived there.
    //
    // For the vacated `from` square: find sliders that attack through `from` in the
    // POST-move board (piece gone). Compare their attacks with pre-move attacks.
    // New targets = attacks with piece gone minus attacks with piece present.

    // Sliders that see through `from` in post-move (piece removed)
    let post_rook_from = rook_attacks(from, post_occ);
    let post_bishop_from = bishop_attacks(from, post_occ);
    let pre_rook_from = rook_attacks(from, pre_occ);
    let pre_bishop_from = bishop_attacks(from, pre_occ);

    // Orthogonal sliders (rooks + queens) near `from`
    let orth_sliders = (board_pieces[ROOK as usize] | board_pieces[QUEEN as usize]) & post_occ;
    let diag_sliders = (board_pieces[BISHOP as usize] | board_pieces[QUEEN as usize]) & post_occ;

    // Find sliders on the same rank/file/diagonal as `from` that gain new targets
    let mut orth_near = orth_sliders & (post_rook_from | pre_rook_from);
    while orth_near != 0 {
        let asq = orth_near.trailing_zeros();
        orth_near &= orth_near - 1;
        if asq == to { continue; } // the moved piece, already handled
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);

        let old_att = rook_attacks(asq, pre_occ) & pre_occ;
        let new_att = rook_attacks(asq, post_occ) & post_occ;

        // New targets (gained)
        let mut gained = new_att & !old_att & !(1u64 << from) & !(1u64 << to);
        while gained != 0 {
            let tsq = gained.trailing_zeros();
            gained &= gained - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta::new(acp as u8, asq as u8, vcp as u8, tsq as u8, true));
            }
        }
        // Lost targets
        let mut lost = old_att & !new_att & !(1u64 << from) & !(1u64 << to);
        while lost != 0 {
            let tsq = lost.trailing_zeros();
            lost &= lost - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta::new(acp as u8, asq as u8, vcp as u8, tsq as u8, false));
            }
        }
    }

    let mut diag_near = diag_sliders & (post_bishop_from | pre_bishop_from);
    while diag_near != 0 {
        let asq = diag_near.trailing_zeros();
        diag_near &= diag_near - 1;
        if asq == to { continue; }
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);

        let old_att = bishop_attacks(asq, pre_occ) & pre_occ;
        let new_att = bishop_attacks(asq, post_occ) & post_occ;

        let mut gained = new_att & !old_att & !(1u64 << from) & !(1u64 << to);
        while gained != 0 {
            let tsq = gained.trailing_zeros();
            gained &= gained - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta::new(acp as u8, asq as u8, vcp as u8, tsq as u8, true));
            }
        }
        let mut lost = old_att & !new_att & !(1u64 << from) & !(1u64 << to);
        while lost != 0 {
            let tsq = lost.trailing_zeros();
            lost &= lost - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta::new(acp as u8, asq as u8, vcp as u8, tsq as u8, false));
            }
        }
    }

    // 7. X-ray changes at `to` square: sliders that used to see through `to`
    // (or see pieces beyond `to`) may now be blocked by the moved piece.
    // Also, if `to` was occupied (capture), sliders might have new visibility.
    // Only process sliders not already handled in sections 3-4.
    let post_rook_to = rook_attacks(to, post_occ);
    let post_bishop_to = bishop_attacks(to, post_occ);
    let pre_rook_to = rook_attacks(to, pre_occ);
    let pre_bishop_to = bishop_attacks(to, pre_occ);

    let mut orth_near_to = orth_sliders & (post_rook_to | pre_rook_to);
    orth_near_to &= !(1u64 << to); // exclude moved piece
    while orth_near_to != 0 {
        let asq = orth_near_to.trailing_zeros();
        orth_near_to &= orth_near_to - 1;
        // Skip if already processed in the `from` section
        if (post_rook_from | pre_rook_from) & (1u64 << asq) != 0 { continue; }
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);

        let old_att = rook_attacks(asq, pre_occ) & pre_occ;
        let new_att = rook_attacks(asq, post_occ) & post_occ;

        let mut gained = new_att & !old_att & !(1u64 << from) & !(1u64 << to);
        while gained != 0 {
            let tsq = gained.trailing_zeros();
            gained &= gained - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta::new(acp as u8, asq as u8, vcp as u8, tsq as u8, true));
            }
        }
        let mut lost = old_att & !new_att & !(1u64 << from) & !(1u64 << to);
        while lost != 0 {
            let tsq = lost.trailing_zeros();
            lost &= lost - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta::new(acp as u8, asq as u8, vcp as u8, tsq as u8, false));
            }
        }
    }

    let mut diag_near_to = diag_sliders & (post_bishop_to | pre_bishop_to);
    diag_near_to &= !(1u64 << to);
    while diag_near_to != 0 {
        let asq = diag_near_to.trailing_zeros();
        diag_near_to &= diag_near_to - 1;
        if (post_bishop_from | pre_bishop_from) & (1u64 << asq) != 0 { continue; }
        let apt = board_mailbox[asq as usize];
        if apt >= 6 { continue; }
        let acolor = if white_bb_post & (1u64 << asq) != 0 { WHITE } else { BLACK };
        let acp = colored_piece(acolor, apt);

        let old_att = bishop_attacks(asq, pre_occ) & pre_occ;
        let new_att = bishop_attacks(asq, post_occ) & post_occ;

        let mut gained = new_att & !old_att & !(1u64 << from) & !(1u64 << to);
        while gained != 0 {
            let tsq = gained.trailing_zeros();
            gained &= gained - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta::new(acp as u8, asq as u8, vcp as u8, tsq as u8, true));
            }
        }
        let mut lost = old_att & !new_att & !(1u64 << from) & !(1u64 << to);
        while lost != 0 {
            let tsq = lost.trailing_zeros();
            lost &= lost - 1;
            if let Some((_, _, vcp)) = victim_at_post(tsq) {
                deltas.push(RawThreatDelta::new(acp as u8, asq as u8, vcp as u8, tsq as u8, false));
            }
        }
    }
}

/// Apply raw threat deltas to update the threat accumulator incrementally.
/// Copies from `prev` and applies all deltas for a specific perspective.
///
/// Marked `#[target_feature]` to propagate AVX2 codegen context into the
/// inlined `apply_deltas_avx2` helper. LTO was already inlining the helper,
/// but the attribute gives LLVM permission to emit tighter AVX2 sequences
/// inside the inlined region. Callers must ensure AVX2 is available; on
/// x86_64 with `-Ctarget-cpu=native` it is.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
pub unsafe fn apply_threat_deltas(
    dst: &mut [i16],           // destination threat accumulator (one perspective)
    src: &[i16],               // source (previous position's threat accumulator)
    deltas: &[RawThreatDelta],
    threat_weights: &[i8],     // [num_threats × hidden_size]
    hidden_size: usize,
    num_threats: usize,
    pov: Color,
    mirrored: bool,
) {
    #[cfg(feature = "profile-threats")]
    crate::threats::apply_stats::record(deltas.len());

    // Collect valid add/sub indices (stack-allocated, no heap)
    let mut adds = [0usize; MAX_THREAT_DELTAS];
    let mut subs = [0usize; MAX_THREAT_DELTAS];
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
        if delta.add() { adds[n_adds] = idx as usize; n_adds += 1; }
        else { subs[n_subs] = idx as usize; n_subs += 1; }
    }
    let adds = &adds[..n_adds];
    let subs = &subs[..n_subs];

    // Prefetch weight rows for upcoming deltas (hide L3 latency)
    #[cfg(target_arch = "x86_64")]
    {
        for &idx in adds.iter().take(4) {
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    threat_weights.as_ptr().add(idx * hidden_size) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
        for &idx in subs.iter().take(4) {
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    threat_weights.as_ptr().add(idx * hidden_size) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
    }

    // Apply weight rows with SIMD when available. Fused pattern: load src
    // chunk into registers, apply all adds/subs, store to dst. Avoids the
    // separate copy_from_slice pass that used to precede apply_deltas_avx2.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && hidden_size % 16 == 0 {
            unsafe {
                apply_deltas_avx2(dst, src, threat_weights, hidden_size, &adds, &subs);
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
/// registers, then stores ONCE. This is Reckless's pattern — 21× less memory
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

    // 8 AVX2 registers × 16 i16 = 128 elements per chunk
    const REGS: usize = 8;
    const CHUNK: usize = REGS * 16; // 128 elements

    let mut offset = 0;
    while offset < hidden_size {
        let chunk_size = (hidden_size - offset).min(CHUNK);
        let nregs = (chunk_size + 15) / 16;

        // Load source (parent) chunk into registers — seeds the per-chunk
        // accumulator. Previously loaded from dst after a separate
        // copy_from_slice pass; loading src directly eliminates that pass.
        let mut regs: [__m256i; REGS] = [_mm256_setzero_si256(); REGS];
        for i in 0..nregs {
            regs[i] = _mm256_loadu_si256(src_ptr.add(offset + i * 16) as *const __m256i);
        }

        // In-loop prefetch — GENTLER variant (T1 hint, 1-delta lookahead).
        //
        // The initial PREFETCH_AHEAD=2 + _MM_HINT_T0 variant tested at
        // SPRT #719 showed large per-uarch variance (Zeus +1.4%, ionos
        // workers −6% to −11%). Root cause: T0 hints fetch into L1, and
        // on CPUs with more aggressive HW prefetchers the software
        // prefetch was pure overhead + L1 pollution. This variant backs
        // off:
        //   - T1 hint: prefetch into L2, don't evict L1 cache lines.
        //   - lookahead=1 (was 2): shorter speculation distance, less
        //     likely to mispredict into a branch we skip.
        // Rationale: HW prefetchers are good at detecting long linear
        // streams; they're bad at random-access weight-row lookups.
        // T1 fills L2 so the real load still hits L2 (~10 cycle) rather
        // than DRAM (~100+), but avoids the L1-eviction cost that T0
        // carries.
        const PREFETCH_AHEAD: usize = 1;

        // Apply paired add+sub
        let mut ai = 0;
        let mut si = 0;
        while ai < adds.len() && si < subs.len() {
            // Prefetch future deltas at this offset.
            if ai + PREFETCH_AHEAD < adds.len() {
                _mm_prefetch(
                    w_ptr.add(adds[ai + PREFETCH_AHEAD] * hidden_size + offset) as *const i8,
                    _MM_HINT_T1,
                );
            }
            if si + PREFETCH_AHEAD < subs.len() {
                _mm_prefetch(
                    w_ptr.add(subs[si + PREFETCH_AHEAD] * hidden_size + offset) as *const i8,
                    _MM_HINT_T1,
                );
            }
            let aw = w_ptr.add(adds[ai] * hidden_size + offset);
            let sw = w_ptr.add(subs[si] * hidden_size + offset);
            for i in 0..nregs {
                let add_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(aw.add(i * 16) as *const __m128i));
                let sub_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(sw.add(i * 16) as *const __m128i));
                regs[i] = _mm256_sub_epi16(_mm256_add_epi16(regs[i], add_w), sub_w);
            }
            ai += 1;
            si += 1;
        }

        // Remaining adds
        while ai < adds.len() {
            if ai + PREFETCH_AHEAD < adds.len() {
                _mm_prefetch(
                    w_ptr.add(adds[ai + PREFETCH_AHEAD] * hidden_size + offset) as *const i8,
                    _MM_HINT_T1,
                );
            }
            let aw = w_ptr.add(adds[ai] * hidden_size + offset);
            for i in 0..nregs {
                let add_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(aw.add(i * 16) as *const __m128i));
                regs[i] = _mm256_add_epi16(regs[i], add_w);
            }
            ai += 1;
        }

        // Remaining subs
        while si < subs.len() {
            if si + PREFETCH_AHEAD < subs.len() {
                _mm_prefetch(
                    w_ptr.add(subs[si + PREFETCH_AHEAD] * hidden_size + offset) as *const i8,
                    _MM_HINT_T1,
                );
            }
            let sw = w_ptr.add(subs[si] * hidden_size + offset);
            for i in 0..nregs {
                let sub_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(sw.add(i * 16) as *const __m128i));
                regs[i] = _mm256_sub_epi16(regs[i], sub_w);
            }
            si += 1;
        }

        // Store registers back
        for i in 0..nregs {
            _mm256_storeu_si256(dst_ptr.add(offset + i * 16) as *mut __m256i, regs[i]);
        }

        offset += CHUNK;
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

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            add_weight_rows_avx2(dst, threat_weights, hidden_size, indices);
        }
        return;
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
    while offset < hidden_size {
        let chunk_size = (hidden_size - offset).min(CHUNK);
        let nregs = (chunk_size + 15) / 16;

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

        offset += CHUNK;
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
        // Reckless has 66,864 — we should match
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
        let bb = colored_piece(BLACK, BISHOP as u8);
        let idx = threat_index(wp, 28, bb, 35, false, WHITE); // e4 → d5
        assert!(idx < 0, "Pawn×Bishop should be excluded, got {}", idx);

        // King attacks queen: excluded (PIECE_INTERACTION_MAP[5][4] = -1)
        let wk = colored_piece(WHITE, KING as u8);
        let bq = colored_piece(BLACK, QUEEN as u8);
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

    /// Section 2 Z-finding cull: regression guard.
    ///
    /// The cull skips the Z-level x-ray delta block when no ray from
    /// `square` has 2+ occupants. These tests pin down the semantics:
    /// (a) an endgame with 0/1 occupants per ray must produce the same
    ///     delta list as a no-cull reference (trivially, since the cull's
    ///     "skipped" branch produces no Z deltas and the Z block also
    ///     short-circuits at `revealed_y == 0` / `revealed_z == 0`).
    /// (b) a position with a genuine Z-chain (S → square → Y → Z) MUST
    ///     emit the Z delta — the cull must NOT fire.
    ///
    /// The primary correctness net is `threat_accum::fuzz_random_games`
    /// which plays thousands of random moves and compares incremental vs
    /// refresh. These targeted tests pin the specific cull boundary.
    #[test]
    fn test_z_finding_cull_endgame_no_z() {
        crate::init();
        // KP endgame: two kings, one pawn. No sliders means Z-finding
        // doesn't even enter the slider loop, but this exercises the
        // pre-check bitboard shape (rays_from_sq_empty, past_first_region)
        // on a real position.
        let b = crate::board::Board::from_fen("4k3/8/8/8/8/4P3/8/4K3 w - - 0 1");
        let mut deltas: Vec<RawThreatDelta> = Vec::new();
        // Enumerate on the pawn's square (e3 = 20). With no sliders,
        // section 2 has nothing to emit regardless of cull state.
        push_threats_for_piece(
            &mut deltas,
            &b.pieces, &b.colors, &b.mailbox,
            b.occupied(), b.colors[WHITE as usize],
            colored_piece(WHITE, PAWN), WHITE, PAWN,
            20, true,
        );
        // No sliders in this position, no slider → square threats.
        // Pawn attacks nothing (e3 attacks d4/f4, both empty).
        // Nothing in sections 1/2/3 applies meaningfully.
        // The key assertion is we don't panic / produce a sane output.
        for d in &deltas {
            assert!(d.from_sq() < 64);
            assert!(d.to_sq() < 64);
        }
    }

    #[test]
    fn test_z_finding_cull_has_z_chain() {
        crate::init();
        // A position with a genuine slider → square → Y → Z chain:
        //   R on a1, pawn on a4 (Y), pawn on a6 (Z), enumerate on a2 (square).
        // Slider (R@a1) sees a2 (direct threat). Y = a4 (first past a2 on
        // rank-file going up). Z = a6 (first past Y). The Z delta MUST
        // be emitted when push_threats_for_piece is called for the piece
        // at a2 appearing (or disappearing).
        //
        // Use a white knight on a2 as the subject (so we trigger a real
        // section 2 walk; any piece works since section 2 is about
        // sliders seeing `square`).
        let b = crate::board::Board::from_fen("4k3/8/P7/8/P7/8/N7/R3K3 w - - 0 1");
        let mut deltas: Vec<RawThreatDelta> = Vec::new();
        push_threats_for_piece(
            &mut deltas,
            &b.pieces, &b.colors, &b.mailbox,
            b.occupied(), b.colors[WHITE as usize],
            colored_piece(WHITE, KNIGHT), WHITE, KNIGHT,
            8, true,  // a2 = 8
        );

        // Expect: section 2 emits slider R@a1 → N@a2 direct threat.
        // AND: the Z-level delta for (R@a1, pawn@a6) is emitted (the
        // x-ray target that flips when a2 gets a piece).
        //
        // Sanity check: the rook at a1 should appear as an attacker to
        // a2 in the emitted deltas.
        let r = colored_piece(WHITE, ROOK) as u8;
        let has_r_to_a2 = deltas.iter().any(|d| {
            d.attacker_cp() == r && d.from_sq() == 0 && d.to_sq() == 8
        });
        assert!(has_r_to_a2, "expected R@a1 → square a2 direct threat in deltas");

        // The Z-chain: rook's Y is a4 (sq 24), Z is a6 (sq 40).
        // The Z delta is a rook-to-a6 entry with `add = !true = false`
        // (because the piece at square is "appearing", Z is "lost").
        let has_r_to_a6 = deltas.iter().any(|d| {
            d.attacker_cp() == r && d.from_sq() == 0 && d.to_sq() == 40
        });
        assert!(has_r_to_a6,
            "expected Z-level delta (R@a1 → pawn@a6) — Z-finding cull must NOT fire here.\n\
             deltas: {:?}",
            deltas.iter().map(|d| (d.attacker_cp(), d.from_sq(), d.victim_cp(), d.to_sq(), d.add())).collect::<Vec<_>>()
        );
    }
}

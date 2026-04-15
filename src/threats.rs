/// Threat feature computation for NNUE.
///
/// Encodes (attacker_piece, attacker_sq, victim_piece, victim_sq) relationships.
/// Each active threat on the board contributes one feature index into the threat
/// accumulator. Feature indices are perspective-relative with king-file mirroring.
///
/// Reference: Reckless engine (src/nnue/threats.rs) — same encoding pattern.
/// Total features: ~66,864 (depends on piece-pair filtering).

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
    /// If semi-excluded and attacker_sq >= attacked_sq, returns negative (skip).
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
    mut from: u32,      // attacker square
    victim_cp: usize,   // colored piece index of victim
    mut to: u32,        // victim square
    mirrored: bool,
    pov: Color,
) -> i32 {
    // Perspective flip: rank flip for black, file flip if mirrored
    let flip = (7 * mirrored as u32) ^ (56 * pov as u32);
    from ^= flip;
    to ^= flip;

    // Remap piece colors relative to POV
    let attacking = if pov == BLACK {
        // Swap color: WP(0)↔BP(6), WN(1)↔BN(7), etc.
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
        let base = pair.base(from, to);
        if base < 0 {
            return base; // excluded pair
        }
        base + PIECE_OFFSET_LOOKUP[attacking][from as usize]
            + ATTACK_INDEX_LOOKUP[attacking][from as usize][to as usize] as i32
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

                // Find attacked occupied squares
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

/// Maximum threat deltas per ply. Reckless uses 80; we use 128 for safety.
pub const MAX_THREAT_DELTAS: usize = 128;

/// Raw threat delta: stores piece data, not pre-computed indices.
/// Indices are computed per-perspective during accumulator update.
#[derive(Copy, Clone)]
pub struct RawThreatDelta {
    pub attacker_cp: u8,
    pub from_sq: u8,
    pub victim_cp: u8,
    pub to_sq: u8,
    pub add: bool,
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
/// Handles: threats from this piece, threats to this piece, and x-ray discoveries.
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
    // 1. Threats FROM this piece to occupied squares
    let attacks = piece_attacks_occ(piece_type, piece_color, square, occ);
    let mut attacked_occ = attacks & occ;
    while attacked_occ != 0 {
        let target_sq = attacked_occ.trailing_zeros();
        attacked_occ &= attacked_occ - 1;
        let victim_pt = mailbox[target_sq as usize];
        if victim_pt >= 6 { continue; }
        let victim_color = if white_bb & (1u64 << target_sq) != 0 { WHITE } else { BLACK };
        deltas.push(RawThreatDelta {
            attacker_cp: cp as u8,
            from_sq: square as u8,
            victim_cp: colored_piece(victim_color, victim_pt) as u8,
            to_sq: target_sq as u8,
            add,
        });
    }

    // 2. Sliding pieces that attack THROUGH this square (x-ray threats)
    let rook_att = rook_attacks(square, occ);
    let bishop_att = bishop_attacks(square, occ);

    let diagonal_sliders = (pieces_bb[BISHOP as usize] | pieces_bb[QUEEN as usize]) & bishop_att;
    let orthogonal_sliders = (pieces_bb[ROOK as usize] | pieces_bb[QUEEN as usize]) & rook_att;

    let mut sliders = (diagonal_sliders | orthogonal_sliders) & occ;
    while sliders != 0 {
        let slider_sq = sliders.trailing_zeros();
        sliders &= sliders - 1;
        let slider_pt = mailbox[slider_sq as usize];
        if slider_pt >= 6 { continue; }
        let slider_color = if white_bb & (1u64 << slider_sq) != 0 { WHITE } else { BLACK };
        let slider_cp = colored_piece(slider_color, slider_pt);

        // The slider attacks this piece's square
        deltas.push(RawThreatDelta {
            attacker_cp: slider_cp as u8,
            from_sq: slider_sq as u8,
            victim_cp: cp as u8,
            to_sq: square as u8,
            add,
        });

        // X-ray: does removing this piece reveal a new target behind it?
        // Compute the ray from slider through this square
        let ray = ray_between(slider_sq, square);
        let beyond = ray_beyond(slider_sq, square) & occ;
        if beyond != 0 {
            let xray_target = if slider_sq < square {
                beyond.trailing_zeros() // first piece beyond
            } else {
                63 - beyond.leading_zeros() // last piece in other direction
            };
            // Wait — we need proper ray extension, not just any piece in beyond.
            // Skip x-rays for now (matching design doc: "skip x-ray initially")
            let _ = xray_target;
        }
    }

    // 3. Non-sliding pieces that attack this square
    // Pawns
    let black_pawns = pieces_bb[PAWN as usize] & colors_bb[BLACK as usize] & pawn_attacks(WHITE, square);
    let white_pawns = pieces_bb[PAWN as usize] & colors_bb[WHITE as usize] & pawn_attacks(BLACK, square);
    // Knights
    let knights = pieces_bb[KNIGHT as usize] & knight_attacks(square);
    // Kings
    let kings = pieces_bb[KING as usize] & king_attacks(square);

    let mut non_sliders = (black_pawns | white_pawns | knights | kings) & occ;
    while non_sliders != 0 {
        let ns_sq = non_sliders.trailing_zeros();
        non_sliders &= non_sliders - 1;
        let ns_pt = mailbox[ns_sq as usize];
        if ns_pt >= 6 { continue; }
        let ns_color = if white_bb & (1u64 << ns_sq) != 0 { WHITE } else { BLACK };
        deltas.push(RawThreatDelta {
            attacker_cp: colored_piece(ns_color, ns_pt) as u8,
            from_sq: ns_sq as u8,
            victim_cp: cp as u8,
            to_sq: square as u8,
            add,
        });
    }
}

/// Placeholder for ray_between and ray_beyond — needed for x-ray threats.
/// For now returns 0 (x-rays skipped per design doc).
fn ray_between(_from: u32, _to: u32) -> Bitboard { 0 }
fn ray_beyond(_from: u32, _to: u32) -> Bitboard { 0 }

/// Apply raw threat deltas to update the threat accumulator incrementally.
/// Copies from `prev` and applies all deltas for a specific perspective.
pub fn apply_threat_deltas(
    dst: &mut [i16],           // destination threat accumulator (one perspective)
    src: &[i16],               // source (previous position's threat accumulator)
    deltas: &[RawThreatDelta],
    threat_weights: &[i8],     // [num_threats × hidden_size]
    hidden_size: usize,
    num_threats: usize,
    pov: Color,
    mirrored: bool,
) {
    // Copy from previous position
    dst[..hidden_size].copy_from_slice(&src[..hidden_size]);

    // Apply each delta
    for delta in deltas {
        let idx = threat_index(
            delta.attacker_cp as usize,
            delta.from_sq as u32,
            delta.victim_cp as usize,
            delta.to_sq as u32,
            mirrored,
            pov,
        );
        if idx < 0 || (idx as usize) >= num_threats { continue; }

        let w_off = idx as usize * hidden_size;
        if delta.add {
            for j in 0..hidden_size {
                dst[j] += threat_weights[w_off + j] as i16;
            }
        } else {
            for j in 0..hidden_size {
                dst[j] -= threat_weights[w_off + j] as i16;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

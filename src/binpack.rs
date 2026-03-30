/// Stockfish BINP-format binpack writer.
///
/// Format: chunks of [BINP magic (4) + chunk_size (4)] + data.
/// Each chain: stem (34 bytes) + movetext (variable-length bit-packed moves).
///
/// Stem layout:
///   CompressedPosition (24 bytes): occ bitboard + nibble-packed pieces
///   CompressedMove (2 bytes): type:2 | from:6 | to:6 | promo:2 (big-endian)
///   Score (2 bytes): signedToUnsigned encoded (big-endian)
///   PlyResult (2 bytes): result:2 | ply:14 (big-endian)
///   Rule50 (2 bytes): big-endian
///   NumPlies (2 bytes): number of continuation moves (big-endian)
///
/// Movetext: hierarchical piece+destination encoding using minimum bits.

use crate::board::Board;
use crate::types::*;
use crate::attacks::*;
use std::io::Write;

/// A single training sample: position + move + score + result + ply.
pub struct TrainSample {
    pub board: Board,
    pub mv: Move,
    pub score: i16,
    pub result: i16,  // -1 = black wins, 0 = draw, 1 = white wins
    pub ply: u16,
}

/// A chain of training samples from a single game.
/// The first sample is the stem (anchor position).
/// Subsequent samples are continuations (encoded as movetext deltas).
pub struct TrainChain {
    pub samples: Vec<TrainSample>,
}

/// Bit writer for variable-length encoding.
struct BitWriter {
    data: Vec<u8>,
    bit_pos: usize,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { data: Vec::with_capacity(256), bit_pos: 0 }
    }

    fn write_bit(&mut self, bit: u8) {
        let byte_idx = self.bit_pos / 8;
        let bit_idx = self.bit_pos % 8;
        if byte_idx >= self.data.len() {
            self.data.push(0);
        }
        if bit != 0 {
            self.data[byte_idx] |= 1 << (7 - bit_idx); // MSB-first, matching Stockfish
        }
        self.bit_pos += 1;
    }

    fn write_bits(&mut self, val: u32, n: usize) {
        // MSB-first: write most significant bit of value first
        for i in (0..n).rev() {
            self.write_bit(((val >> i) & 1) as u8);
        }
    }

    /// Pad to byte boundary.
    fn flush(&mut self) {
        while self.bit_pos % 8 != 0 {
            self.write_bit(0);
        }
    }

    /// Write VLE (variable-length encoding) with block size 4.
    /// Matches Stockfish's writeVle16.
    fn write_vle(&mut self, mut v: u16) {
        const BLOCK_SIZE: u32 = 4;
        let mask = (1u16 << BLOCK_SIZE) - 1;
        loop {
            let block = v & mask;
            v >>= BLOCK_SIZE;
            if v > 0 {
                // More blocks to come: set extension bit
                self.write_bits((block | (1 << BLOCK_SIZE)) as u32, (BLOCK_SIZE + 1) as usize);
            } else {
                // Last block: no extension bit
                self.write_bits(block as u32, (BLOCK_SIZE + 1) as usize);
                break;
            }
        }
    }

    fn bytes(&self) -> &[u8] {
        &self.data
    }
}

/// Minimum bits needed to represent values in [0, max_val).
fn used_bits(max_val: usize) -> usize {
    if max_val <= 1 { return 0; }
    (usize::BITS - (max_val - 1).leading_zeros()) as usize
}

/// signedToUnsigned: encode i16 as u16 for delta coding.
fn signed_to_unsigned(a: i16) -> u16 {
    let mut r = a as u16;
    if r & 0x8000 != 0 {
        r ^= 0x7FFF;
    }
    (r << 1) | (r >> 15) // rotate left by 1
}

/// Compress a Board into 24-byte BINP CompressedPosition.
fn compress_position(board: &Board) -> [u8; 24] {
    let mut data = [0u8; 24];

    // Build occupancy bitboard
    let occ = board.occupied();

    // Write occupancy as big-endian
    data[0] = (occ >> 56) as u8;
    data[1] = (occ >> 48) as u8;
    data[2] = (occ >> 40) as u8;
    data[3] = (occ >> 32) as u8;
    data[4] = (occ >> 24) as u8;
    data[5] = (occ >> 16) as u8;
    data[6] = (occ >> 8) as u8;
    data[7] = occ as u8;

    // Nibble-pack pieces in LSB-first order of occupancy
    let mut nibble_idx = 0;
    let mut tmp_occ = occ;
    while tmp_occ != 0 {
        let sq = tmp_occ.trailing_zeros() as u8;
        tmp_occ &= tmp_occ - 1;

        let pt = board.piece_type_at(sq);
        let color = if board.colors[WHITE as usize] & (1u64 << sq) != 0 { WHITE } else { BLACK };

        // Determine nibble value
        let nibble = if pt == KING && color == BLACK && board.side_to_move == BLACK {
            15u8 // black king + black to move
        } else if pt == PAWN {
            // Check if this pawn has an EP square behind it
            if board.ep_square != NO_SQUARE {
                let rank = sq >> 3;
                if rank == 3 && color == WHITE && board.ep_square == sq - 8 {
                    12 // EP pawn (white)
                } else if rank == 4 && color == BLACK && board.ep_square == sq + 8 {
                    12 // EP pawn (black)
                } else {
                    color * 1 // normal pawn: 0=white, 1=black
                }
            } else {
                color * 1
            }
        } else if pt == ROOK {
            // Check castling rights
            if color == WHITE && ((sq == 0 && board.castling & CASTLE_WQ != 0) || (sq == 7 && board.castling & CASTLE_WK != 0)) {
                13 // white castling rook
            } else if color == BLACK && ((sq == 56 && board.castling & CASTLE_BQ != 0) || (sq == 63 && board.castling & CASTLE_BK != 0)) {
                14 // black castling rook
            } else {
                6 + color // normal rook: 6=white, 7=black
            }
        } else {
            // Standard piece nibble: pawn=0/1, knight=2/3, bishop=4/5, rook=6/7, queen=8/9, king=10/11
            pt * 2 + color
        };

        let byte_idx = nibble_idx / 2;
        if nibble_idx % 2 == 0 {
            data[8 + byte_idx] |= nibble;
        } else {
            data[8 + byte_idx] |= nibble << 4;
        }
        nibble_idx += 1;
    }

    data
}

/// Compress a move into 2-byte BINP CompressedMove (big-endian).
/// Type: 0=Normal, 1=Promotion, 2=Castle, 3=EnPassant
/// For castling: to = rook square (not king destination).
fn compress_move(mv: Move, _board: &Board) -> [u8; 2] {
    let from = move_from(mv) as u16;
    let to = move_to(mv) as u16;
    let flags = move_flags(mv);

    let (move_type, actual_to, promo) = if flags == FLAG_EN_PASSANT {
        (3u16, to, 0u16)
    } else if flags == FLAG_CASTLE {
        // BINP uses rook square as destination
        let rook_sq = if to == 6 { 7u16 }      // white kingside: g1 → h1 rook
            else if to == 2 { 0u16 }             // white queenside: c1 → a1 rook
            else if to == 62 { 63u16 }           // black kingside: g8 → h8 rook
            else { 56u16 };                       // black queenside: c8 → a8 rook
        (2, rook_sq, 0)
    } else if is_promotion(mv) {
        let promo_idx = match promotion_piece_type(mv) {
            KNIGHT => 0u16,
            BISHOP => 1,
            ROOK => 2,
            QUEEN => 3,
            _ => 3,
        };
        (1, to, promo_idx)
    } else {
        (0, to, 0)
    };

    let packed = (move_type << 14) | (from << 8) | (actual_to << 2) | promo;
    [(packed >> 8) as u8, packed as u8]
}

/// Get the nth set bit in a bitboard (0-indexed).
fn nth_set_bit(bb: u64, n: usize) -> u8 {
    let mut b = bb;
    for _ in 0..n {
        b &= b - 1; // clear lowest set bit
    }
    b.trailing_zeros() as u8
}

/// Encode a move in BINP movetext format using hierarchical piece+destination encoding.
/// Writes variable-length bits to the BitWriter.
fn encode_move(bw: &mut BitWriter, mv: Move, board: &Board) {
    let us = board.side_to_move;
    let our_pieces = board.colors[us as usize];
    let their_pieces = board.colors[flip_color(us) as usize];
    let occupied = board.occupied();
    let from = move_from(mv);
    let to = move_to(mv);
    let flags = move_flags(mv);
    let pt = board.piece_type_at(from);

    // 1. Encode piece selection: index of 'from' among our pieces
    let num_our = our_pieces.count_ones() as usize;
    let mut piece_id = 0usize;
    let mut tmp = our_pieces;
    while tmp != 0 {
        let sq = tmp.trailing_zeros() as u8;
        if sq == from { break; }
        piece_id += 1;
        tmp &= tmp - 1;
    }
    bw.write_bits(piece_id as u32, used_bits(num_our));

    match pt {
        PAWN => {
            let (forward, promo_rank, start_rank): (i8, u8, u8) = if us == WHITE {
                (8, 6, 1)
            } else {
                (-8, 1, 6)
            };

            // Compute pawn destinations
            let mut destinations = 0u64;
            let attack_targets = their_pieces | if board.ep_square != NO_SQUARE {
                1u64 << board.ep_square
            } else { 0 };

            destinations |= pawn_attacks(us, from as u32) & attack_targets;

            let sq_fwd = (from as i8 + forward) as u8;
            if sq_fwd < 64 && occupied & (1u64 << sq_fwd) == 0 {
                destinations |= 1u64 << sq_fwd;
                let sq_fwd2 = (sq_fwd as i8 + forward) as u8;
                if (from >> 3) == start_rank && sq_fwd2 < 64 && occupied & (1u64 << sq_fwd2) == 0 {
                    destinations |= 1u64 << sq_fwd2;
                }
            }

            let dest_count = destinations.count_ones() as usize;

            if (from >> 3) == promo_rank {
                // Promotion: moveId = destIndex * 4 + promoIndex
                let dest_idx = {
                    let mut idx = 0;
                    let mut d = destinations;
                    while d != 0 {
                        let sq = d.trailing_zeros() as u8;
                        if sq == to { break; }
                        idx += 1;
                        d &= d - 1;
                    }
                    idx
                };
                let promo_idx = if is_promotion(mv) {
                    match promotion_piece_type(mv) {
                        KNIGHT => 0u32,
                        BISHOP => 1,
                        ROOK => 2,
                        QUEEN => 3,
                        _ => 3,
                    }
                } else { 3 };
                let move_id = dest_idx * 4 + promo_idx;
                bw.write_bits(move_id, used_bits(dest_count * 4));
            } else {
                // Normal pawn move: index of 'to' among destinations
                let dest_idx = {
                    let mut idx = 0u32;
                    let mut d = destinations;
                    while d != 0 {
                        let sq = d.trailing_zeros() as u8;
                        if sq == to { break; }
                        idx += 1;
                        d &= d - 1;
                    }
                    idx
                };
                bw.write_bits(dest_idx, used_bits(dest_count));
            }
        }
        KING => {
            // King encoding: normal attack squares + castling (separate counts)
            let attacks = king_attacks(from as u32) & !our_pieces;
            let attacks_count = attacks.count_ones() as usize;

            // Count castling options (ordered: queenside first, then kingside)
            let (our_castle_qs, our_castle_ks) = if us == WHITE {
                (board.castling & CASTLE_WQ != 0, board.castling & CASTLE_WK != 0)
            } else {
                (board.castling & CASTLE_BQ != 0, board.castling & CASTLE_BK != 0)
            };
            let num_castlings = our_castle_qs as usize + our_castle_ks as usize;

            if flags == FLAG_CASTLE {
                // Castling: moveId = attacks_count + castling_index
                // Queenside = index 0 (if available), kingside = next
                let is_kingside = to == 6 || to == 62; // g1 or g8
                let castle_idx = if is_kingside {
                    if our_castle_qs { 1 } else { 0 } // kingside is second if queenside exists
                } else {
                    0 // queenside is always first
                };
                bw.write_bits((attacks_count + castle_idx) as u32, used_bits(attacks_count + num_castlings));
            } else {
                // Normal king move: index of 'to' in attacks bitboard
                let dest_idx = {
                    let mut idx = 0u32;
                    let mut d = attacks;
                    while d != 0 {
                        let sq = d.trailing_zeros() as u8;
                        if sq == to { break; }
                        idx += 1;
                        d &= d - 1;
                    }
                    idx
                };
                bw.write_bits(dest_idx, used_bits(attacks_count + num_castlings));
            }
        }
        KNIGHT => {
            let destinations = knight_attacks(from as u32) & !our_pieces;
            let dest_count = destinations.count_ones() as usize;
            let dest_idx = {
                let mut idx = 0u32;
                let mut d = destinations;
                while d != 0 {
                    let sq = d.trailing_zeros() as u8;
                    if sq == to { break; }
                    idx += 1;
                    d &= d - 1;
                }
                idx
            };
            bw.write_bits(dest_idx, used_bits(dest_count));
        }
        BISHOP => {
            let destinations = bishop_attacks(from as u32, occupied) & !our_pieces;
            let dest_count = destinations.count_ones() as usize;
            let dest_idx = {
                let mut idx = 0u32;
                let mut d = destinations;
                while d != 0 {
                    let sq = d.trailing_zeros() as u8;
                    if sq == to { break; }
                    idx += 1;
                    d &= d - 1;
                }
                idx
            };
            bw.write_bits(dest_idx, used_bits(dest_count));
        }
        ROOK => {
            let destinations = rook_attacks(from as u32, occupied) & !our_pieces;
            let dest_count = destinations.count_ones() as usize;
            let dest_idx = {
                let mut idx = 0u32;
                let mut d = destinations;
                while d != 0 {
                    let sq = d.trailing_zeros() as u8;
                    if sq == to { break; }
                    idx += 1;
                    d &= d - 1;
                }
                idx
            };
            bw.write_bits(dest_idx, used_bits(dest_count));
        }
        QUEEN => {
            let destinations = queen_attacks(from as u32, occupied) & !our_pieces;
            let dest_count = destinations.count_ones() as usize;
            let dest_idx = {
                let mut idx = 0u32;
                let mut d = destinations;
                while d != 0 {
                    let sq = d.trailing_zeros() as u8;
                    if sq == to { break; }
                    idx += 1;
                    d &= d - 1;
                }
                idx
            };
            bw.write_bits(dest_idx, used_bits(dest_count));
        }
        _ => {}
    }
}

/// BINP chunk writer. Buffers chains and writes complete chunks.
pub struct BinpackWriter<W: Write> {
    writer: W,
    chunk_buf: Vec<u8>,
    chains_written: u64,
    positions_written: u64,
}

impl<W: Write> BinpackWriter<W> {
    pub fn new(writer: W) -> Self {
        BinpackWriter {
            writer,
            chunk_buf: Vec::with_capacity(1 << 20), // 1MB buffer
            chains_written: 0,
            positions_written: 0,
        }
    }

    /// Write a complete chain (game) to the binpack.
    /// Each sample in the chain must have the board set to the position AT that move.
    pub fn write_chain(&mut self, chain: &TrainChain) -> std::io::Result<()> {
        if chain.samples.is_empty() { return Ok(()); }

        let stem = &chain.samples[0];
        let num_plies = (chain.samples.len() - 1) as u16;

        // Stem: CompressedPosition (24) + CompressedMove (2) + Score (2) +
        //        PlyResult (2) + Rule50 (2) + NumPlies (2) = 34 bytes
        let comp_pos = compress_position(&stem.board);
        self.chunk_buf.extend_from_slice(&comp_pos);

        let comp_move = compress_move(stem.mv, &stem.board);
        self.chunk_buf.extend_from_slice(&comp_move);

        let score_enc = signed_to_unsigned(stem.score);
        self.chunk_buf.push((score_enc >> 8) as u8);
        self.chunk_buf.push(score_enc as u8);

        let result_enc = signed_to_unsigned(stem.result) & 3;
        let ply_result = (result_enc << 14) | (stem.ply & 0x3FFF);
        self.chunk_buf.push((ply_result >> 8) as u8);
        self.chunk_buf.push(ply_result as u8);

        let rule50 = stem.board.halfmove;
        self.chunk_buf.push((rule50 >> 8) as u8);
        self.chunk_buf.push(rule50 as u8);

        self.chunk_buf.push((num_plies >> 8) as u8);
        self.chunk_buf.push(num_plies as u8);

        self.positions_written += 1;

        // Movetext: encode move + VLE score delta (matching Stockfish format)
        // Score delta: VLE of signedToUnsigned(plyScore - m_lastScore)
        // m_lastScore starts as -stem.score, alternates sign each ply
        if num_plies > 0 {
            let mut bw = BitWriter::new();
            let mut m_last_score = -(stem.score as i16);

            for i in 1..chain.samples.len() {
                let sample = &chain.samples[i];
                encode_move(&mut bw, sample.mv, &sample.board);

                // VLE score delta
                let ply_score = sample.score;
                let delta = signed_to_unsigned(ply_score - m_last_score);
                bw.write_vle(delta);
                m_last_score = -ply_score;

                self.positions_written += 1;
            }
            bw.flush();
            self.chunk_buf.extend_from_slice(bw.bytes());
        }

        self.chains_written += 1;

        // Flush chunk if buffer is large enough (target ~1MB chunks)
        if self.chunk_buf.len() >= 1 << 20 {
            self.flush_chunk()?;
        }

        Ok(())
    }

    /// Flush current buffer as a BINP chunk.
    fn flush_chunk(&mut self) -> std::io::Result<()> {
        if self.chunk_buf.is_empty() { return Ok(()); }
        // Write BINP magic
        self.writer.write_all(b"BINP")?;
        // Write chunk size (little-endian u32)
        let size = self.chunk_buf.len() as u32;
        self.writer.write_all(&size.to_le_bytes())?;
        // Write chunk data
        self.writer.write_all(&self.chunk_buf)?;
        self.chunk_buf.clear();
        Ok(())
    }

    /// Finish writing — flush remaining data.
    pub fn finish(&mut self) -> std::io::Result<()> {
        self.flush_chunk()?;
        self.writer.flush()
    }

    pub fn chains_written(&self) -> u64 { self.chains_written }
    pub fn positions_written(&self) -> u64 { self.positions_written }
}

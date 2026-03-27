/// Syzygy tablebase probing via shakmaty-syzygy.
///
/// WDL probes at interior nodes (requires halfmove == 0).
/// DTZ probes at root for best tablebase move.

use shakmaty::{Chess, FromSetup, CastlingMode};
use shakmaty::fen::Fen;
use shakmaty_syzygy::{Tablebase, AmbiguousWdl, Dtz, MaybeRounded};

use crate::board::Board;

/// Wrapper around shakmaty-syzygy Tablebase.
pub struct SyzygyTB {
    tb: Tablebase<Chess>,
    max_pieces: usize,
}

impl SyzygyTB {
    /// Initialize tablebases from a directory path.
    pub fn new(path: &str) -> Result<Self, String> {
        let mut tb = Tablebase::new();
        tb.add_directory(path).map_err(|e| format!("Syzygy init: {}", e))?;

        let max_pieces = tb.max_pieces();
        eprintln!("info string Syzygy tablebases loaded: {} pieces from {}", max_pieces, path);

        Ok(SyzygyTB { tb, max_pieces })
    }

    /// Maximum number of pieces supported.
    pub fn max_pieces(&self) -> usize {
        self.max_pieces
    }

    /// Probe WDL for an interior node. Returns Some(wdl_score) or None.
    /// wdl_score: positive = winning, 0 = draw, negative = losing.
    /// Only valid when halfmove clock is 0 (no 50-move rule complications).
    pub fn probe_wdl(&self, board: &Board) -> Option<i32> {
        if crate::bitboard::popcount(board.occupied()) as usize > self.max_pieces {
            return None;
        }

        let chess = board_to_shakmaty(board)?;
        match self.tb.probe_wdl(&chess) {
            Ok(wdl) => Some(ambiguous_wdl_to_score(wdl)),
            Err(_) => None,
        }
    }

    /// Probe DTZ at root to find the best tablebase move.
    /// Returns Some((best_move_uci, wdl_score)) or None.
    pub fn probe_root(&self, board: &Board) -> Option<(String, i32)> {
        if crate::bitboard::popcount(board.occupied()) as usize > self.max_pieces {
            return None;
        }

        let chess = board_to_shakmaty(board)?;
        // Get all legal moves with their DTZ
        match self.tb.best_move(&chess) {
            Ok(Some((m, dtz))) => {
                // Format as UCI (e.g. "b2c2" not "Rb2-c2")
                let uci = shakmaty::uci::UciMove::from_standard(m);
                let wdl = dtz_to_wdl_score(dtz);
                Some((uci.to_string(), wdl))
            }
            _ => None,
        }
    }
}

/// Convert our Board to a shakmaty Chess position via FEN.
fn board_to_shakmaty(board: &Board) -> Option<Chess> {
    let fen_str = board.to_fen();
    let fen: Fen = fen_str.parse().ok()?;
    Chess::from_setup(fen.into_setup(), CastlingMode::Standard).ok()
}

fn ambiguous_wdl_to_score(wdl: AmbiguousWdl) -> i32 {
    match wdl {
        AmbiguousWdl::Win => 20000,
        AmbiguousWdl::MaybeLoss => -1,
        AmbiguousWdl::CursedWin | AmbiguousWdl::MaybeWin => 1,
        AmbiguousWdl::Draw => 0,
        AmbiguousWdl::BlessedLoss => -1,
        AmbiguousWdl::Loss => -20000,
    }
}

fn dtz_to_wdl_score(dtz: MaybeRounded<Dtz>) -> i32 {
    let d = dtz.ignore_rounding();
    if d.0 > 0 { 20000 }
    else if d.0 < 0 { -20000 }
    else { 0 }
}

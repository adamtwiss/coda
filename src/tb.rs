/// Syzygy tablebase probing via shakmaty-syzygy.
///
/// WDL probes at interior nodes (requires halfmove == 0).
/// DTZ probes at root for best tablebase move.

use shakmaty::{Chess, FromSetup, CastlingMode};
use shakmaty::fen::Fen;
use shakmaty_syzygy::{Tablebase, AmbiguousWdl, Dtz, MaybeRounded};

use crate::board::Board;
use crate::tb_cache::TbCache;

/// Default cache size in MB. CCRL convention allows engines one
/// probe-result cache. Override via `setoption name TBHash value N`.
pub const DEFAULT_TB_HASH_MB: usize = 16;

/// Wrapper around shakmaty-syzygy Tablebase.
pub struct SyzygyTB {
    tb: Tablebase<Chess>,
    max_pieces: usize,
    cache: TbCache,
}

impl SyzygyTB {
    /// Initialize tablebases from a directory path. Uses the default
    /// cache size; call `with_cache_mb` to customise.
    pub fn new(path: &str) -> Result<Self, String> {
        Self::new_with_cache(path, DEFAULT_TB_HASH_MB)
    }

    /// Initialize tablebases with a specific cache size in MB (0 disables).
    pub fn new_with_cache(path: &str, cache_mb: usize) -> Result<Self, String> {
        let mut tb = Tablebase::new();
        tb.add_directory(path).map_err(|e| format!("Syzygy init: {}", e))?;

        let max_pieces = tb.max_pieces();
        let cache = TbCache::new(cache_mb);
        eprintln!("info string Syzygy tablebases loaded: {} pieces from {}, cache {} MB",
                  max_pieces, path, cache.size_mb());

        Ok(SyzygyTB { tb, max_pieces, cache })
    }

    /// Maximum number of pieces supported.
    pub fn max_pieces(&self) -> usize {
        self.max_pieces
    }

    /// Clear the probe cache (called on ucinewgame).
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Probe WDL for an interior node. Returns Some(wdl_score) or None.
    /// wdl_score: positive = winning, 0 = draw, negative = losing.
    /// Only valid when halfmove clock is 0 (no 50-move rule complications).
    pub fn probe_wdl(&self, board: &Board) -> Option<i32> {
        if crate::bitboard::popcount(board.occupied()) as usize > self.max_pieces {
            return None;
        }

        // Native-hash cache check first — avoids the Board→Chess translation
        // and the shakmaty decompression path when the slot is valid.
        // Halfmove is part of the cache key (C2 fix): shakmaty's probe_wdl
        // returns different results across halfmove, so we must cache per
        // halfmove to avoid serving a stale cursed-win/blessed-loss answer.
        if let Some(wdl) = self.cache.probe(board.hash, board.halfmove) {
            return Some(wdl);
        }

        let chess = board_to_shakmaty(board)?;
        match self.tb.probe_wdl(&chess) {
            Ok(wdl) => {
                let score = ambiguous_wdl_to_score(wdl);
                self.cache.store(board.hash, board.halfmove, score);
                Some(score)
            }
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
    // shakmaty-syzygy DTZ convention: negative = side to move wins,
    // positive = side to move loses. Invert for our score convention
    // (positive = good for side to move).
    let d = dtz.ignore_rounding();
    if d.0 < 0 { 20000 }
    else if d.0 > 0 { -20000 }
    else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use std::path::Path;

    fn tb_path() -> Option<String> {
        // Check common tablebase locations
        for path in &["/tablebases", "/home/adam/chess/syzygy", "/syzygy"] {
            if Path::new(path).exists() {
                return Some(path.to_string());
            }
        }
        None
    }

    fn make_tb() -> Option<SyzygyTB> {
        let path = tb_path()?;
        SyzygyTB::new(&path).ok()
    }

    #[test]
    fn tb_root_probe_signs() {
        let tb = match make_tb() {
            Some(tb) => tb,
            None => { eprintln!("Skipping TB test: no tablebases found"); return; }
        };

        // KQP vs K: White winning
        // White to move → positive (STM winning)
        let board = Board::from_fen("4k3/8/8/8/8/4P3/8/3QK3 w - - 0 1");
        let (_, wdl) = tb.probe_root(&board).expect("probe should succeed");
        assert!(wdl > 0, "KQP vs K, White to move: expected positive, got {}", wdl);

        // Black to move → negative (STM losing)
        let board = Board::from_fen("4k3/8/8/8/8/4P3/8/3QK3 b - - 0 1");
        let (_, wdl) = tb.probe_root(&board).expect("probe should succeed");
        assert!(wdl < 0, "KQP vs K, Black to move: expected negative, got {}", wdl);

        // KRP vs KR: Black winning (the original bug position)
        // Black to move → positive (STM winning)
        let board = Board::from_fen("5R2/8/8/8/4r3/4p3/4k3/2K5 b - - 0 71");
        let (_, wdl) = tb.probe_root(&board).expect("probe should succeed");
        assert!(wdl > 0, "KRP vs KR, Black to move (winning): expected positive, got {}", wdl);

        // White to move → negative (STM losing)
        let board = Board::from_fen("5R2/8/8/8/4r3/4p3/4k3/2K5 w - - 0 71");
        let (_, wdl) = tb.probe_root(&board).expect("probe should succeed");
        assert!(wdl < 0, "KRP vs KR, White to move (losing): expected negative, got {}", wdl);

        // KBN vs K: White winning
        let board = Board::from_fen("4k3/8/8/8/8/8/8/2BNK3 w - - 0 1");
        let (_, wdl) = tb.probe_root(&board).expect("probe should succeed");
        assert!(wdl > 0, "KBN vs K, White to move: expected positive, got {}", wdl);

        // KR vs KR: drawn (rooks can't capture each other)
        let board = Board::from_fen("4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1");
        let (_, wdl) = tb.probe_root(&board).expect("probe should succeed");
        assert!(wdl == 0, "KR vs KR (drawn): expected 0, got {}", wdl);
    }

    #[test]
    fn tb_wdl_probe_signs() {
        let tb = match make_tb() {
            Some(tb) => tb,
            None => { eprintln!("Skipping TB test: no tablebases found"); return; }
        };

        // KQP vs K: White winning, White to move → positive
        let board = Board::from_fen("4k3/8/8/8/8/4P3/8/3QK3 w - - 0 1");
        let wdl = tb.probe_wdl(&board).expect("WDL probe should succeed");
        assert!(wdl > 0, "WDL KQP vs K, White to move: expected positive, got {}", wdl);

        // KQP vs K: White winning, Black to move → negative
        let board = Board::from_fen("4k3/8/8/8/8/4P3/8/3QK3 b - - 0 1");
        let wdl = tb.probe_wdl(&board).expect("WDL probe should succeed");
        assert!(wdl < 0, "WDL KQP vs K, Black to move: expected negative, got {}", wdl);

        // KRP vs KR: Black winning, Black to move → positive
        let board = Board::from_fen("5R2/8/8/8/4r3/4p3/4k3/2K5 b - - 0 1");
        let wdl = tb.probe_wdl(&board).expect("WDL probe should succeed");
        assert!(wdl > 0, "WDL KRP vs KR, Black to move: expected positive, got {}", wdl);

        // KR vs KR: drawn
        let board = Board::from_fen("4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1");
        let wdl = tb.probe_wdl(&board).expect("WDL probe should succeed");
        assert!(wdl == 0, "WDL KR vs KR (drawn): expected 0, got {}", wdl);
    }
}

#![allow(dead_code)]
// Intentional patterns, not defects — kept so `clippy -D warnings` stays a
// usable regression gate without churning hot code:
#![allow(clippy::needless_range_loop)] // manual indexing mirrors matmul math in SIMD/NNUE paths
#![allow(clippy::too_many_arguments)] // hot inference/search signatures
#![allow(clippy::identity_op)] // `color * 1` etc. document an encoding pattern
#![allow(clippy::type_complexity)] // factoring SIMD tuple types out hurts locality
#![allow(clippy::manual_memcpy)] // explicit element loops in hot accumulator paths
#![allow(clippy::manual_clamp)] // .max().min() avoids clamp's max<min panic semantics
#![allow(clippy::unusual_byte_groupings)] // hex/bit literals grouped by field, not by 4s
#![allow(clippy::unnecessary_cast)] // explicit pointer casts document intrinsic intent
#![allow(clippy::if_same_then_else)] // distinct conditions kept separate for documentation
#![allow(clippy::manual_strip)] // index-slice form reads clearer in the EPD parser
#![allow(clippy::doc_lazy_continuation)] // doc-comment markdown wrapping, cosmetic

// Local scratch buffers use MaybeUninit to skip hot-path zeroing. These macros
// keep the raw pointer plus initialized-prefix slice pattern consistent.
macro_rules! scratch_ptr {
    ($storage:expr, $ty:ty) => {
        $storage.as_mut_ptr() as *mut $ty
    };
}

macro_rules! scratch_slice {
    ($ptr:expr, $len:expr) => {
        unsafe { std::slice::from_raw_parts($ptr, $len) }
    };
    (mut $ptr:expr, $len:expr) => {
        unsafe { std::slice::from_raw_parts_mut($ptr, $len) }
    };
}

mod types;
mod bitboard;
mod zobrist;
mod attacks;
mod setwise;
mod board;
mod movegen;
mod eval;
mod see;
mod tt;
mod movepicker;
mod search;
mod thread_pool;
mod uci;
mod epd;
pub mod nnue;
pub mod book;
pub mod tb;
pub mod tb_cache;
pub mod datagen;
pub mod nnue_export;
pub mod bullet_convert;
mod cuckoo;
pub mod threats;
pub mod threat_accum;
pub mod sparse_l1;
pub mod nnue_simd;

use board::Board;
use movegen::{perft, perft_divide};
use clap::{Parser, Subcommand};

use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        attacks::init_attacks();
        bitboard::init_bitboards();
        zobrist::init_zobrist();
        board::init_castle_masks();
        search::init_lmr();
        cuckoo::init_cuckoo();
        nnue::init_nnue();
        threats::init_threats();
    });
}

/// Coda Chess Engine — Chess Optimised, Developed Agentically
#[derive(Parser)]
#[command(name = "coda", about = "Coda Chess Engine", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// NNUE network file path
    #[arg(long = "nnue", short = 'n', global = true)]
    nnue: Option<String>,

    /// Opening book file path
    #[arg(long = "book", short = 'b')]
    book: Option<String>,

    /// DIAGNOSTIC ONLY — load nets even when they mismatch Coda's inference
    /// configuration. Default is refuse-to-load so mismatches can't silently degrade
    /// SPRT / OB / Lichess games. Use only when deliberately probing a
    /// mismatched net. Also available as UCI option `LoadAnyway`.
    #[arg(long = "load-anyway", global = true)]
    load_anyway: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Search benchmark (default depth 12, 48 positions)
    Bench {
        /// Search depth
        #[arg(default_value_t = 12)]
        depth: i32,
        /// Number of threads
        #[arg(long = "threads", short = 't', default_value_t = 1)]
        threads: usize,
        /// Override tunable defaults for this bench invocation only.
        /// Format: NAME=VALUE (repeatable). Diagnostic use for stage-1
        /// loose-knob perturbation audits.
        #[arg(long = "set", value_name = "NAME=VALUE")]
        set: Vec<String>,
    },
    /// Pathology bench: run known-pathological positions, flag any
    /// whose tree exceeds `--threshold` nodes. Slow / optional —
    /// surfaces tail-distribution blow-ups (eval-driven search
    /// non-convergence, extension cascades) that SPRT cannot detect.
    /// Exit code = number of positions over threshold.
    BenchPathology {
        /// Search depth
        #[arg(default_value_t = 8)]
        depth: i32,
        /// Node threshold per position; flag if exceeded
        #[arg(long = "threshold", default_value_t = 5_000_000)]
        threshold: u64,
    },
    /// NNUE micro-benchmark: pure inference throughput (no search).
    /// `--mode fresh` uses the debug scalar `force_recompute` path.
    /// `--mode refresh` uses the production SIMD `materialize` path
    /// with an invalidated Finny cache (mirrors a king-bucket-cross
    /// first-touch in search).
    /// `--mode incremental` makes one legal move per iter and pops,
    /// measuring realistic incremental cost. `make-unmake-all` cycles
    /// through each position's legal moves instead of timing only move 0.
    EvalBench {
        /// Search-bench positions are used if no EPD is supplied.
        #[arg(long, default_value = "")]
        epd: String,
        /// Number of positions to load (0 = all)
        #[arg(long, default_value_t = 8)]
        positions: usize,
        /// Repetitions per position
        #[arg(long, default_value_t = 100_000)]
        reps: u64,
        /// `fresh` | `refresh` | `incremental` | `make-unmake` | `make-unmake-all`
        #[arg(long, default_value = "fresh")]
        mode: String,
        /// Disable AVX-512 & VNNI for this run (forces AVX-2 SIMD path)
        #[arg(long, default_value_t = false)]
        no_avx512: bool,
        /// Disable threat-delta generation in make/unmake eval-bench modes.
        #[arg(long, default_value_t = false)]
        no_threat_deltas: bool,
    },
    /// Run EPD test suite
    Epd {
        /// EPD file path
        #[arg(default_value = "testdata/wac.epd")]
        path: String,
        /// Time per position (ms)
        #[arg(short = 't', long = "time", default_value_t = 100)]
        time: u64,
        /// Maximum positions (0 = all)
        #[arg(short = 'm', long = "max", default_value_t = 0)]
        max: usize,
    },
    /// Inspect binpack training data (ply distribution, score stats)
    /// Dump binpack positions verbatim as TSV: ply, score, move (UCI), fen — one row per
    /// stored position, in file order, with NO filtering whatsoever.
    ///
    /// Exists because every other binpack tool here is blind to skip-mark
    /// filtering. `sample-positions` silently drops |score|>2000 and in-check
    /// positions; `inspect-binpack` and `chain-stats` aggregate. Stockfish-style
    /// filtering MARKS positions with an out-of-range score sentinel rather than
    /// deleting rows, so row counts and ply histograms are identical before and
    /// after filtering — only the score field moves. Verifying filtered data
    /// therefore needs a dump that preserves marked rows exactly where they are.
    DumpBinpack {
        /// Input binpack file
        #[arg(short, long)]
        input: String,
        /// Output TSV path ("-" for stdout)
        #[arg(short, long, default_value = "-")]
        output: String,
        /// Max positions to read (0 = all)
        #[arg(long, default_value_t = 0)]
        count: usize,
    },
    /// Split a binpack into N parts at BLOCK boundaries, for sharding work
    /// across machines.
    ///
    /// The format is `File = Block*`, `Block = "BINP" + u32le(size) + data`, so
    /// every block boundary is a valid split point and every part is itself a
    /// valid binpack. Merging is plain concatenation (`cat a b > c`) — no tool
    /// needed — because a concatenation of Blocks is still a File.
    ///
    /// Splitting at an arbitrary BYTE offset instead corrupts the tail part and
    /// makes readers fault partway through, which is why this walks headers.
    SplitBinpack {
        /// Input binpack file
        #[arg(short, long)]
        input: String,
        /// Output prefix; parts are written as <prefix>.000, <prefix>.001, ...
        #[arg(short, long)]
        output: String,
        /// Number of parts
        #[arg(long, default_value_t = 2)]
        parts: usize,
    },
    InspectBinpack {
        /// Input binpack file
        #[arg(short = 'i', long = "input")]
        input: String,
        /// Max positions to read (0 = all)
        #[arg(long = "count", default_value_t = 1000000)]
        count: u64,
    },
    /// Report average chain (game-fragment) length of a binpack — singletons
    /// (avg ~1, e.g. import-tsv outlier sets) vs proper game chains (avg ~20-40).
    /// Bullet's interleave loader (since 2026-06-30) weights files by sampled
    /// position density, not raw bytes, so low-chain files are no longer
    /// over-weighted — but density still varies run-to-run with file
    /// composition, so use this to sanity-check expected exposure before/after
    /// changing a file's chain structure. Chain boundary = ply discontinuity.
    ChainStats {
        /// Input binpack file
        #[arg(short = 'i', long = "input")]
        input: String,
        /// Number of chains to read (0 = whole file)
        #[arg(short = 'c', long = "chains", default_value_t = 100)]
        chains: u64,
    },
    /// Perft with divide
    Perft {
        /// Search depth
        #[arg(default_value_t = 5)]
        depth: u32,
        /// FEN position (default: startpos)
        #[arg(trailing_var_arg = true)]
        fen: Vec<String>,
    },
    /// Perft benchmark suite (6 standard positions)
    PerftBench,
    /// Patch a .nnue file's header flag bits in-place. Primary use case:
    /// mark an already-converted net as HL-CReLU (sets bit 5 on extended_kb
    /// nets). Refuses to operate on nets without extended_kb since bit 5
    /// means consensus_buckets on legacy nets.
    PatchNet {
        /// Input .nnue file (modified in place unless --output given)
        #[arg(long, short = 'i')]
        input: String,
        /// Output path (default: overwrite input)
        #[arg(long, short = 'o')]
        output: Option<String>,
        /// Set the HL-CReLU bit (bit 5 when extended_kb=1)
        #[arg(long)]
        set_hl_crelu: bool,
        /// Clear the HL-CReLU bit
        #[arg(long)]
        clear_hl_crelu: bool,
        /// Set the FT-SCReLU bit (bit 0). Retrofit for historical nets that
        /// were converted without `--screlu` but should have had it set for
        /// consistency with the convert-bullet recipe.
        #[arg(long)]
        set_ft_screlu: bool,
        /// Clear the FT-SCReLU bit (bit 0)
        #[arg(long)]
        clear_ft_screlu: bool,
        /// Just print current flags, make no changes
        #[arg(long)]
        inspect: bool,
    },
    /// Dump SPSA tune spec (name,int,default,min,max,c_end,r_end) from tunables
    TuneSpec {
        /// SPSA r_end (default 0.002)
        #[arg(long, default_value_t = 0.002)]
        r_end: f32,
        /// Emit only `core: true` tunables (curated subset for routine retunes)
        #[arg(long, default_value_t = false)]
        core: bool,
    },
    /// Download NNUE net from net.txt URL
    FetchNet,
    /// C8 audit fuzzer: diff Coda's current enumerate_threats (physical-
    /// frame semi-exclusion) against a bf-frame reference matching Bullet
    /// training, to surface same-type-pair mismatches introduced when real
    /// STM is Black.
    FuzzThreats {
        /// Number of random positions to generate
        #[arg(long, default_value_t = 10000)]
        count: usize,
        /// Random seed (0 = time-based)
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Max half-moves from startpos per random position
        #[arg(long, default_value_t = 40)]
        max_plies: usize,
        /// Print first N mismatches in detail
        #[arg(long, default_value_t = 5)]
        verbose: usize,
        /// Use the post-C8fix-2 Bullet reference (`map_features` 62931d1)
        /// instead of the pre-fix bf-frame reference. Expected mismatch
        /// count: 0.
        #[arg(long, default_value_t = false)]
        postfix: bool,
    },
    /// NNUE network health check (uses --nnue/-n flag or auto-discovers)
    CheckNet {},
    /// Measure threat-matrix row sparsity in a v9 .nnue file.
    /// Reports zero-row count, near-zero rows, and distribution of row
    /// max-abs weights. Used for sparsity-probe diagnosis.
    MeasureNetSparsity {
        /// Input .nnue file
        #[arg(long, short = 'i')]
        input: String,
    },
    /// Dump per-layer weight statistics for a .nnue file. Diagnostic for
    /// comparing trained nets — surfaces collapsed/stuck layers and dead
    /// neurons. Per-king-bucket and per-output-bucket breakdowns optional.
    DumpNetStats {
        /// Input .nnue file
        #[arg(long, short = 'i')]
        input: String,
        /// Print per-output-bucket stats for L1/L2/output
        #[arg(long)]
        per_bucket: bool,
        /// Print per-king-bucket stats for PSQ
        #[arg(long)]
        per_kb: bool,
        /// Print histogram bins for each layer
        #[arg(long)]
        histogram: bool,
    },
    /// Generate training data (SF binpack format)
    Datagen {
        /// Output binpack file
        #[arg(long, short = 'o', default_value = "data.binpack")]
        output: String,
        /// Search depth
        #[arg(long, short = 'd', default_value_t = 8)]
        depth: i32,
        /// Number of games
        #[arg(long, short = 'g', default_value_t = 1000)]
        games: usize,
        /// Number of threads
        #[arg(long, short = 't', default_value_t = 1)]
        threads: usize,
        /// Hash table size (MB)
        #[arg(long, default_value_t = 16)]
        hash: usize,
        /// Blunder rate (0.0 - 1.0)
        #[arg(long, default_value_t = 0.0)]
        blunder: f64,
        /// Force-capture rate (0.0 - 1.0): probability of playing a random capture
        /// instead of best move, creating material-imbalanced games for training diversity
        #[arg(long, default_value_t = 0.0)]
        force_captures: f64,
        /// Source EPD for material removal mode
        #[arg(long)]
        epd: Option<String>,
    },
    /// Sample positions from binpack to EPD
    SamplePositions {
        /// Input binpack file
        #[arg(long, short = 'i')]
        input: String,
        /// Output EPD file
        #[arg(long, short = 'o', default_value = "positions.epd")]
        output: String,
        /// Number of positions to sample
        #[arg(long, default_value_t = 1_000_000)]
        count: usize,
        /// Sample rate (0.0 = keep all until count)
        #[arg(long, default_value_t = 0.0)]
        rate: f64,
        /// Also emit the stored STM-POV label as a second tab column (fen<TAB>score)
        #[arg(long, default_value_t = false)]
        scores: bool,
    },
    /// Import a TSV of (fen, score) rows into an sfbinpack binpack for fine-tuning.
    /// Score = STM-POV cp (SF deep eval); WDL derived from score; mate-ish dropped.
    ImportTsv {
        /// Input TSV (tab-separated; first line is a header)
        #[arg(long, short = 'i')]
        input: String,
        /// Output binpack file
        #[arg(long, short = 'o', default_value = "imported.binpack")]
        output: String,
        /// 0-based column index of the FEN
        #[arg(long, default_value_t = 15)]
        fen_col: usize,
        /// 0-based column index of the STM-POV score (cp)
        #[arg(long, default_value_t = 10)]
        score_col: usize,
        /// Drop positions with |score| above this (mate-ish filter)
        #[arg(long, default_value_t = 2000)]
        max_abs: i32,
        /// Upsample each kept position N times
        #[arg(long, default_value_t = 1)]
        repeat: usize,
    },
    /// Import a per-position game TSV (fen, uci, score, result, ply) into an
    /// sfbinpack binpack — every position, in game order, with actual move (for
    /// chain compression), actual WDL and ply, NO filtering (the loader filters).
    /// Feed a game-TSV export (one game per line, converted from PGN); input "-" reads stdin.
    ImportGameTsv {
        /// Input TSV ("-" for stdin); cols: fen, uci, score, result, ply (no header)
        #[arg(long, short = 'i')]
        input: String,
        /// Output binpack file
        #[arg(long, short = 'o', default_value = "games.binpack")]
        output: String,
    },
    /// Import datagen PGN -> binpack natively (replaces pgn_to_game_tsv.py | import-game-tsv)
    ImportPgn {
        /// Input PGN ("-" for stdin; e.g. bzcat *.pgn.bz2 | coda import-pgn -i -)
        #[arg(long, short = 'i')]
        input: String,
        /// Output binpack file
        #[arg(long, short = 'o', default_value = "games.binpack")]
        output: String,
    },
    /// Evaluate positions from binpack to measure eval scale distribution
    EvalDist {
        /// Input binpack file
        #[arg(long, short = 'i')]
        input: String,
        /// Number of positions to evaluate
        #[arg(long, short = 'c', default_value_t = 1_000_000)]
        count: usize,
        /// Optional CSV dump: `fen,white_result,coda_eval_white_cp` per kept
        /// position (for cross-engine eval-quality benchmarking).
        #[arg(long)]
        csv: Option<String>,
        /// Quiet-filter positions (skip captures/promotions/in-check) to
        /// match how NNUE is consumed. Off by default (scale measurement
        /// wants all positions); on for eval-quality CSV export.
        #[arg(long, default_value_t = false)]
        quiet_only: bool,
        /// Shard `k/N`: process only entries whose global index % N == k.
        /// Run N copies (k=0..N-1) in parallel to scan a whole binpack across
        /// cores. Implies scan-to-EOF (ignores --count) and streaming CSV
        /// (no in-memory distribution stats — required for billion-scale).
        #[arg(long)]
        shard: Option<String>,
        /// Only emit CSV rows where |coda_eval - lc0| >= this (cp). 0 = off.
        /// Coda's eval is ~LC0-scale by construction, so this is the cheap
        /// pre-filter for the eval-blindspot harvest (the "150" of 150/80).
        #[arg(long, default_value_t = 0)]
        min_error: i32,
        /// Only emit CSV rows where |lc0| <= this (cp). 0 = off. Restricts to
        /// the balanced/non-decided band; applied before the NNUE eval so it
        /// also saves compute on out-of-band positions.
        #[arg(long, default_value_t = 0)]
        max_abs_lc0: i32,
    },
    /// Batch static-eval a FEN list: one FEN per line (extra tab-separated
    /// columns after the FEN are ignored), emits `fen<TAB>eval_stm_cp` keyed
    /// by FEN. Exists so corpus tooling never scrapes UCI `eval` output by
    /// line order (known misalignment trap, see
    /// docs/overrate_eval_investigation_2026-06-30.md).
    EvalFens {
        /// Input file: one FEN per line, or TSV with FEN in column 0
        #[arg(long, short = 'i')]
        input: String,
        /// Output TSV path ("-" for stdout)
        #[arg(long, short = 'o', default_value = "-")]
        output: String,
    },
    /// Count material-class frequency + per-class label (score) distribution
    /// across a binpack. Read-only. Diagnoses endgame-class coverage vs
    /// mislabeling in training data (undersampled -> low count; mislabeled ->
    /// high mean|score| on a drawish class).
    BinpackMaterial {
        /// Input .binpack path
        #[arg(long, short = 'i')]
        input: String,
        /// Max positions to scan (0 = whole file)
        #[arg(long, default_value_t = 0)]
        max: usize,
    },
    /// Dump threat features for a FEN position (for cross-checking with Bullet)
    DumpThreats {
        /// FEN string
        #[arg(default_value = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")]
        fen: String,
    },
    /// Build a per-feature hit histogram across an EPD corpus.
    /// Enumerates threats from both POVs at each position and counts firings
    /// per feature index. Used to identify dead/cold/hot features for
    /// encoding cleanup and importance-ordered FT row layout.
    ProfileThreats {
        /// Input EPD file (one FEN per line)
        #[arg(long, short = 'i')]
        input: String,
        /// Output CSV (idx,hits) sorted by hits desc
        #[arg(long, short = 'o', default_value = "threat_hits.csv")]
        output: String,
        /// Limit to first N positions (0 = all)
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Optional NNUE net — correlate hit counts with per-row weight magnitudes
        #[arg(long)]
        net: Option<String>,
    },
    /// Show statistics for a binpack file
    BinpackStats {
        /// Input binpack file
        #[arg(long, short = 'i')]
        input: String,
    },
    /// Convert Bullet quantised.bin to .nnue
    ConvertBullet {
        /// Input quantised.bin path
        #[arg(long, short = 'i')]
        input: String,
        /// Output .nnue path
        #[arg(long, short = 'o', default_value = "net.nnue")]
        output: String,
        /// Use SCReLU activation
        #[arg(long)]
        screlu: bool,
        /// Use pairwise multiplication
        #[arg(long)]
        pairwise: bool,
        /// L1 hidden layer size (0 = no hidden layers, v5)
        #[arg(long, default_value_t = 0)]
        hidden: usize,
        /// L2 hidden layer size
        #[arg(long, default_value_t = 0)]
        hidden2: usize,
        /// Use int8 quantization for L1
        #[arg(long)]
        int8l1: bool,
        /// Bucketed hidden layers (output buckets baked into L1/L2)
        #[arg(long)]
        bucketed_hidden: bool,
        /// Override FT size (auto-detected if 0)
        #[arg(long, default_value_t = 0)]
        ft_size: usize,
        /// Hidden layers quantised as i16 (GoChess-style, default is f32)
        #[arg(long)]
        int16_hidden: bool,
        /// Dual L1 activation (CReLU+SCReLU concat, v8 format)
        #[arg(long)]
        dual: bool,
        /// Dual L2 activation (CReLU+SCReLU concat on L2's output, v11 format).
        /// Implies --dual; the output affine reads 2*l2_size.
        #[arg(long)]
        dual_l2: bool,
        /// [LEGACY] Use consensus king bucket layout (fine-near, coarse-far).
        /// Equivalent to --kb-layout consensus. Ignored if --kb-layout is set explicitly.
        #[arg(long)]
        consensus_buckets: bool,
        /// King bucket layout: uniform | consensus.
        /// Both default to kb-count 16.
        #[arg(long, default_value = "")]
        kb_layout: String,
        /// King bucket count. Overrides the default for the chosen kb-layout
        /// (only useful for experimental configs — normally derived from layout).
        #[arg(long, default_value_t = 0)]
        kb_count: usize,
        /// Number of threat features (0 = no threats, v9 format)
        #[arg(long, default_value_t = 0)]
        threats: usize,
        /// Source output bucket count (default 8, set to 2 for 2-bucket nets)
        #[arg(long, default_value_t = 8)]
        output_buckets: usize,
        /// Hidden-layer activation is CReLU (default SCReLU). Sets header
        /// bit 5 on extended_kb nets so the engine auto-configures
        /// HiddenActivation=crelu at load time.
        #[arg(long)]
        hl_crelu: bool,
    },
    /// Convert .nnue to Bullet checkpoint (for transfer learning)
    ConvertCheckpoint {
        /// Output checkpoint directory
        #[arg(long, short = 'o', default_value = "v7_checkpoint")]
        output: String,
        /// Feature transformer size
        #[arg(long, default_value_t = 1024)]
        ft: usize,
        /// L1 hidden layer size
        #[arg(long, default_value_t = 16)]
        l1: usize,
        /// L2 hidden layer size
        #[arg(long, default_value_t = 32)]
        l2: usize,
    },
}

/// One-line ISA / build diagnostic. Answers "why is NPS low on this host?":
/// reports whether this is a DEBUG (unoptimized) build and which SIMD kernels
/// the CPU will select at runtime. A scalar/debug build shows `eval=SCALAR` or
/// `[DEBUG…]`. Printed to stderr at startup (visible in GUI/CCRL engine logs,
/// safe for the UCI stdout protocol) and to stdout at the top of `bench`.
fn isa_banner() -> String {
    let profile = if cfg!(debug_assertions) {
        "DEBUG unoptimized ~5-10x slow!"
    } else {
        "release"
    };
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = std::is_x86_feature_detected!("avx2");
        let avx512f = std::is_x86_feature_detected!("avx512f");
        let avx512bw = std::is_x86_feature_detected!("avx512bw");
        let avx512vnni = std::is_x86_feature_detected!("avx512vnni");
        let avxvnni = std::is_x86_feature_detected!("avxvnni");
        let bmi2 = std::is_x86_feature_detected!("bmi2");
        let eval = if avx512f && avx512bw {
            "avx512"
        } else if avx2 {
            "avx2"
        } else {
            "SCALAR"
        };
        let movegen = if attacks::using_pext() { "pext" } else { "magic" };
        format!(
            "Coda ISA [{profile}] eval={eval} movegen={movegen} \
cpu[avx2={avx2} avx512f={avx512f} avx512bw={avx512bw} avx512vnni={avx512vnni} avxvnni={avxvnni} bmi2={bmi2}] \
build-floor[avx2={a2} avx512f={a5}]",
            a2 = cfg!(target_feature = "avx2"),
            a5 = cfg!(target_feature = "avx512f"),
        )
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        format!("Coda ISA [{profile}] (non-x86_64 arch)")
    }
}

fn main() {
    // Restore default SIGPIPE behavior so writes to a closed stdout
    // (e.g. cutechess killed the engine mid-search/mid-print between
    // games at very-short TC) terminate the process cleanly instead
    // of panicking inside `println!`. Without this, fastchess/OB
    // workers see a `failed printing to stdout: Broken pipe` panic,
    // the game is recorded as "abandoned" at PlyCount=0, and the
    // SPSA tune (or SPRT) accumulates dozens-to-hundreds of phantom
    // results. Diagnosed via tune #1417 (2026-05-22).
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    init();

    // OpenBench datagen invokes the *dev* engine as
    //   ./coda "genfens N seed S book B" quit
    // where argv[1] is the entire genfens command string (with spaces).
    // Intercept it before clap, which would otherwise reject it as an unknown
    // subcommand. Emits `info string genfens <FEN>` lines for OB's opening-book
    // generation (see datagen::run_genfens). Runs after init() so the magic
    // attack tables are ready for move generation.
    {
        let raw: Vec<String> = std::env::args().collect();
        if raw.get(1).map_or(false, |a| a.starts_with("genfens")) {
            datagen::run_genfens(&raw[1]);
            return;
        }
    }

    let cli = Cli::parse();

    // ISA/build banner to stderr — visible in GUI/CCRL engine logs without
    // touching the UCI stdout protocol. The one-line answer to "why is NPS low
    // on this host?" (debug build, or scalar SIMD fallback).
    eprintln!("{}", isa_banner());

    if cli.load_anyway {
        crate::nnue::LOAD_ANYWAY.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!("WARNING: --load-anyway set; training/inference mismatches will NOT refuse load");
    }

    match cli.command {
        None => {
            // UCI mode (default)
            let nnue_ref = cli.nnue.as_deref();
            let book_ref = cli.book.as_deref();
            uci::uci_loop_with_nnue(nnue_ref, book_ref);
        }

        Some(Commands::Bench { depth, threads, set }) => {
            println!("{}", isa_banner());
            let nnue_path = cli.nnue.as_deref();
            // Apply per-invocation tunable overrides (--set NAME=VALUE).
            let mut lmr_c_changed = false;
            for kv in &set {
                let (name, value_str) = match kv.split_once('=') {
                    Some(parts) => parts,
                    None => { eprintln!("--set ignored (no '='): {}", kv); continue; }
                };
                let value: i32 = match value_str.trim().parse() {
                    Ok(v) => v,
                    Err(_) => { eprintln!("--set ignored (bad value): {}", kv); continue; }
                };
                let mut matched = false;
                for (pname, atomic, _d, min, max, _c, _is_core) in search::tunable_params() {
                    if pname == name.trim() {
                        let clamped = value.max(min).min(max);
                        atomic.store(clamped, std::sync::atomic::Ordering::Relaxed);
                        eprintln!("--set {} = {} (clamped from {} into [{}, {}])", pname, clamped, value, min, max);
                        if pname.starts_with("LMR_C") { lmr_c_changed = true; }
                        matched = true;
                        break;
                    }
                }
                if !matched { eprintln!("--set ignored (unknown tunable): {}", name); }
            }
            // LMR_C_QUIET / LMR_C_CAP are baked into the LMR reduction table
            // at startup. Without rebuilding it, --set updates the atomic but
            // search keeps reading the old cached table. Mirror UCI setoption
            // behavior in src/uci.rs.
            if lmr_c_changed { search::init_lmr(); }
            let start = std::time::Instant::now();
            if threads <= 1 {
                let nodes = search::bench(depth, nnue_path);
                let elapsed = start.elapsed();
                let nps = if elapsed.as_secs_f64() > 0.0 {
                    (nodes as f64 / elapsed.as_secs_f64()) as u64
                } else { 0 };
                println!("\n{} nodes {} nps", nodes, nps);
            } else {
                let nnue_owned = nnue_path.map(|s| s.to_string());
                // Thread 0 runs with stats printed (info lines, per-position
                // progress, EBF/FMC summary); other threads run silent to
                // avoid interleaved output. Final aggregate line printed at
                // the end.
                let handles: Vec<_> = (0..threads).enumerate().map(|(i, _)| {
                    let np = nnue_owned.clone();
                    std::thread::spawn(move || {
                        if i == 0 {
                            search::bench(depth, np.as_deref())
                        } else {
                            search::bench_silent(depth, np.as_deref())
                        }
                    })
                }).collect();
                let mut total_nodes = 0u64;
                for h in handles {
                    total_nodes += h.join().unwrap();
                }
                let elapsed = start.elapsed();
                let nps = if elapsed.as_secs_f64() > 0.0 {
                    (total_nodes as f64 / elapsed.as_secs_f64()) as u64
                } else { 0 };
                println!("\n{} nodes {} nps ({} threads)", total_nodes, nps, threads);
            }
        }

        Some(Commands::BenchPathology { depth, threshold }) => {
            let nnue_path = cli.nnue.as_deref();
            let over = search::bench_pathology(depth, threshold, nnue_path);
            std::process::exit(over.min(127) as i32);
        }

        Some(Commands::EvalBench { epd, positions, reps, mode, no_avx512, no_threat_deltas }) => {
            use crate::{board::Board, nnue::{NNUENet, NNUEAccumulator}, eval::evaluate_nnue, threat_accum::ThreatStack};
            let nnue_path = cli.nnue.as_deref().expect("EvalBench requires --nnue/-n");
            let mut net = NNUENet::load(nnue_path).expect("failed to load NNUE");
            if no_avx512 {
                // Force-disable AVX-512 dispatch so kernels fall through to
                // AVX-2 paths. Matches the SIMD-consistency test pattern.
                net.has_avx512 = false;
                net.has_avx512_vnni = false;
                println!("info string eval-bench: AVX-512 & VNNI disabled (AVX-2 path only)");
            }
            let mut acc = NNUEAccumulator::new(net.hidden_size);
            let mut tstack = ThreatStack::new(net.hidden_size);
            tstack.active = net.has_threats;

            let bench_positions: Vec<String> = if epd.is_empty() {
                // Same positions as `coda bench` (49-position list, see search::BENCH_POSITIONS).
                crate::search::BENCH_POSITIONS.iter().map(|s| s.to_string()).collect()
            } else {
                let text = std::fs::read_to_string(&epd).expect("failed to read EPD");
                let iter = text.lines().filter(|l| !l.trim().is_empty());
                let mut out: Vec<String> = Vec::new();
                for line in iter {
                    let parts: Vec<&str> = line.split_whitespace().take(4).collect();
                    if parts.len() < 4 { continue; }
                    out.push(format!("{} 0 1", parts.join(" ")));
                    if positions > 0 && out.len() >= positions { break; }
                }
                out
            };
            let n_positions = if positions > 0 { positions.min(bench_positions.len()) } else { bench_positions.len() };
            let positions_slice = &bench_positions[..n_positions];

            // Warm-up: ensure AVX/VNNI-aware code paths are jitted / caches populated.
            for fen in positions_slice {
                let board = Board::from_fen(fen);
                acc.force_recompute(&net, &board);
                let _ = evaluate_nnue(&board, &net, &mut acc, &tstack);
            }

            let start = std::time::Instant::now();
            let mut sum: i64 = 0;
            let mut count: u64 = 0;
            let mut threat_delta_total: u64 = 0;
            let mut threat_delta_max: usize = 0;
            let mut move_bucket_counts = [0u64; 5];
            let mut move_bucket_deltas = [0u64; 5];
            let mut move_corpus_positions = 0u64;
            let mut move_corpus_legal_moves = 0u64;
            const MOVE_BUCKET_NAMES: [&str; 5] = ["quiet", "capture", "promotion", "castle", "ep"];
            if mode == "fresh" {
                for fen in positions_slice {
                    let board = Board::from_fen(fen);
                    for _ in 0..reps {
                        // Force-recompute both perspectives from scratch: no Finny
                        // benefit, no incremental shortcuts. Pure forward-pass cost.
                        // NB: this currently uses the DEBUG scalar path in
                        // `force_recompute`. Real production first-touch cost is
                        // measured by `--mode refresh` below.
                        acc.force_recompute(&net, &board);
                        let s = evaluate_nnue(&board, &net, &mut acc, &tstack);
                        sum = sum.wrapping_add(s as i64);
                        count += 1;
                    }
                }
            } else if mode == "refresh" {
                for fen in positions_slice {
                    let board = Board::from_fen(fen);
                    for _ in 0..reps {
                        // Production "fresh" path: invalidate Finny + current
                        // accumulator, then call `materialize` which routes
                        // through `refresh_accumulator` + `finny_batch_apply`
                        // (SIMD). Mirrors what happens on a king-bucket cross
                        // in real search when Finny has no cached state.
                        acc.invalidate_for_bench();
                        if tstack.active {
                            tstack.reset_for_bench();
                        }
                        acc.materialize(&net, &board);
                        let s = evaluate_nnue(&board, &net, &mut acc, &tstack);
                        sum = sum.wrapping_add(s as i64);
                        count += 1;
                    }
                }
            } else if mode == "incremental" {
                use crate::movegen::generate_legal_moves;
                use crate::search::build_dirty_piece;
                use crate::types::{flip_color, move_from, move_to};
                for fen in positions_slice {
                    let mut board = Board::from_fen(fen);
                    acc.force_recompute(&net, &board);
                    if tstack.active {
                        tstack.ensure_computed(&net.threat_weights, net.num_threat_features, &board);
                    }
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { continue; }
                    // Pick the first legal quiet we can make/unmake cleanly.
                    let mv = legal.get(0);
                    for _ in 0..reps {
                        let us = board.side_to_move;
                        let them = flip_color(us);
                        let moved_pt = board.piece_type_at(move_from(mv));
                        let captured_pt = board.piece_type_at(move_to(mv));
                        let dirty = build_dirty_piece(mv, us, them, moved_pt, captured_pt, &net);
                        if !board.make_move(mv) { break; }
                        acc.push(dirty);
                        let s = evaluate_nnue(&board, &net, &mut acc, &tstack);
                        sum = sum.wrapping_add(s as i64);
                        board.unmake_move();
                        acc.pop();
                        count += 1;
                    }
                }
            } else if mode == "make-unmake" || mode == "make-unmake-all" {
                // Times make_move + unmake_move. No NNUE evaluation happens.
                // With threat-delta generation enabled (as in v9 production
                // search), the dominant cost inside make_move is
                // `push_threats_for_piece` + deltas fanout. Disabling
                // threat deltas (measured separately) shows the base
                // move-make cost alone.
                use crate::movegen::generate_legal_moves;
                use crate::types::{
                    move_flags, move_to, is_promotion, FLAG_CASTLE, FLAG_EN_PASSANT,
                    NO_PIECE_TYPE,
                };
                let cycle_all_moves = mode == "make-unmake-all";
                for fen in positions_slice {
                    let mut board = Board::from_fen(fen);
                    // v9 production path: threat deltas always on.
                    board.generate_threat_deltas = net.has_threats && !no_threat_deltas;
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { continue; }
                    move_corpus_positions += 1;
                    move_corpus_legal_moves += legal.len as u64;
                    for rep in 0..reps {
                        let mv_idx = if cycle_all_moves {
                            (rep as usize) % legal.len
                        } else {
                            0
                        };
                        let mv = legal.get(mv_idx);
                        let bucket = if move_flags(mv) == FLAG_EN_PASSANT {
                            4
                        } else if move_flags(mv) == FLAG_CASTLE {
                            3
                        } else if is_promotion(mv) {
                            2
                        } else if board.piece_type_at(move_to(mv)) != NO_PIECE_TYPE {
                            1
                        } else {
                            0
                        };
                        if !board.make_move(mv) { break; }
                        let delta_count = board.threat_deltas.len();
                        threat_delta_total += delta_count as u64;
                        threat_delta_max = threat_delta_max.max(delta_count);
                        move_bucket_counts[bucket] += 1;
                        move_bucket_deltas[bucket] += delta_count as u64;
                        board.unmake_move();
                        count += 1;
                    }
                }
            } else {
                eprintln!("Unknown --mode: {} (expected 'fresh' | 'refresh' | 'incremental' | 'make-unmake' | 'make-unmake-all')", mode);
                return;
            }
            let secs = start.elapsed().as_secs_f64();
            let evs = count as f64 / secs;
            // In make-unmake mode the "mean_eval" field is meaningless
            // (no eval was called). Report 0 so the CSV is consistent.
            println!(
                "evalbench mode={} positions={} reps/pos={} total_evals={} elapsed={:.3}s  evals/sec={:.0}  mean_eval={}",
                mode, n_positions, reps, count, secs, evs,
                sum / count.max(1) as i64,
            );
            if mode == "make-unmake" || mode == "make-unmake-all" {
                let avg_deltas = threat_delta_total as f64 / count.max(1) as f64;
                let avg_legal = move_corpus_legal_moves as f64 / move_corpus_positions.max(1) as f64;
                println!(
                    "evalbench move_corpus active_positions={} legal_moves={} avg_legal_moves_per_position={:.2}",
                    move_corpus_positions, move_corpus_legal_moves, avg_legal,
                );
                println!(
                    "evalbench threat_deltas total={} avg_per_move={:.2} max_per_move={}",
                    threat_delta_total, avg_deltas, threat_delta_max,
                );
                for i in 0..MOVE_BUCKET_NAMES.len() {
                    if move_bucket_counts[i] == 0 { continue; }
                    let avg = move_bucket_deltas[i] as f64 / move_bucket_counts[i] as f64;
                    println!(
                        "evalbench move_bucket name={} count={} avg_deltas={:.2}",
                        MOVE_BUCKET_NAMES[i], move_bucket_counts[i], avg,
                    );
                }
                #[cfg(feature = "profile-threats")]
                {
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                    crate::threats::thr_stats::report();
                }
            }
        }

        Some(Commands::Epd { path, time, max }) => {
            epd::run_epd(&path, time, max, cli.nnue.as_deref());
        }

        Some(Commands::DumpBinpack { input, output, count }) => {
            run_dump_binpack(&input, &output, count);
        }

        Some(Commands::SplitBinpack { input, output, parts }) => {
            run_split_binpack(&input, &output, parts);
        }

        Some(Commands::InspectBinpack { input, count }) => {
            use sfbinpack::CompressedTrainingDataEntryReader;
            let file = std::fs::File::open(&input).expect("Failed to open binpack");
            let mut reader = CompressedTrainingDataEntryReader::new(file).expect("Failed to read binpack");

            let mut ply_hist = vec![0u64; 500];
            let mut total = 0u64;
            let mut score_zero = 0u64;
            let mut in_check = 0u64;
            let mut is_capture = 0u64;
            let mut is_promo = 0u64;
            let mut score_sum = 0i64;
            let mut piece_count_hist = vec![0u64; 33];
            // Score range histogram
            let mut score_ranges = [0u64; 10]; // <100, <500, <1000, <2000, <5000, <10000, <20000, <30000, >=30000, =32001
            let limit = if count == 0 { u64::MAX } else { count };

            while reader.has_next() && total < limit {
                let entry = reader.next();
                total += 1;
                let ply = entry.ply as usize;
                if ply < 500 { ply_hist[ply] += 1; }
                if entry.score == 0 { score_zero += 1; }
                let abs_score = entry.score.unsigned_abs() as u32;
                score_sum += abs_score as i64;
                match abs_score {
                    0..=99 => score_ranges[0] += 1,
                    100..=499 => score_ranges[1] += 1,
                    500..=999 => score_ranges[2] += 1,
                    1000..=1999 => score_ranges[3] += 1,
                    2000..=4999 => score_ranges[4] += 1,
                    5000..=9999 => score_ranges[5] += 1,
                    10000..=19999 => score_ranges[6] += 1,
                    20000..=29999 => score_ranges[7] += 1,
                    30000..=32000 => score_ranges[8] += 1,
                    _ => score_ranges[9] += 1, // 32001+ (mate scores?)
                }

                let stm = entry.pos.side_to_move();
                if entry.pos.is_checked(stm) { in_check += 1; }

                // Check if destination has a piece (capture)
                let dest_piece = entry.pos.piece_at(entry.mv.to());
                if dest_piece != sfbinpack::chess::piece::Piece::NONE {
                    is_capture += 1;
                }
                if entry.mv.mtype() == sfbinpack::chess::r#move::MoveType::Promotion {
                    is_promo += 1;
                }

                // Count pieces for output bucket analysis
                let pc = entry.pos.occupied().count() as usize;
                if pc < 33 { piece_count_hist[pc] += 1; }
            }

            println!("Inspected {} positions from {}", total, input);
            println!("\n=== Ply Distribution ===");
            for ply in 0..60 {
                if ply_hist[ply] > 0 {
                    println!("  ply {:3}: {:8} ({:.2}%)", ply, ply_hist[ply],
                        100.0 * ply_hist[ply] as f64 / total as f64);
                }
            }
            let below_16: u64 = ply_hist[..16].iter().sum();
            let above_16: u64 = ply_hist[16..].iter().sum();
            println!("\n  ply < 16:  {:8} ({:.1}%)", below_16, 100.0 * below_16 as f64 / total as f64);
            println!("  ply >= 16: {:8} ({:.1}%)", above_16, 100.0 * above_16 as f64 / total as f64);

            println!("\n=== Position Stats ===");
            println!("  Score = 0: {:8} ({:.1}%)", score_zero, 100.0 * score_zero as f64 / total as f64);
            println!("  In check:  {:8} ({:.1}%)", in_check, 100.0 * in_check as f64 / total as f64);
            println!("  Captures:  {:8} ({:.1}%)", is_capture, 100.0 * is_capture as f64 / total as f64);
            println!("  Promos:    {:8} ({:.1}%)", is_promo, 100.0 * is_promo as f64 / total as f64);
            println!("  Avg |score|: {:.1}", score_sum as f64 / total as f64);
            println!("\n=== Score Distribution ===");
            let labels = ["    0-99", "  100-499", "  500-999", "1000-1999", "2000-4999",
                         "5000-9999", "10K-19999", "20K-29999", "30K-32000", "  >=32001"];
            for (i, label) in labels.iter().enumerate() {
                println!("  |score| {}: {:8} ({:.1}%)", label, score_ranges[i],
                    100.0 * score_ranges[i] as f64 / total as f64);
            }

            println!("\n=== Piece Count (Output Bucket proxy) ===");
            for pc in (2..33).rev() {
                if piece_count_hist[pc] > 0 {
                    let bucket = std::cmp::min(7, (32 - pc) / 4); // approximate MaterialCount bucketing
                    println!("  {} pieces (bucket ~{}): {:8} ({:.2}%)", pc, bucket,
                        piece_count_hist[pc], 100.0 * piece_count_hist[pc] as f64 / total as f64);
                }
            }
        }

        Some(Commands::ChainStats { input, chains }) => {
            use sfbinpack::CompressedTrainingDataEntryReader;
            let file_size = std::fs::metadata(&input).map(|m| m.len()).unwrap_or(0);
            let file = std::fs::File::open(&input).expect("Failed to open binpack");
            let mut reader = CompressedTrainingDataEntryReader::new(file).expect("Failed to read binpack");
            let target = if chains == 0 { u64::MAX } else { chains };
            let mut lens: Vec<u32> = Vec::new();
            let mut cur: u32 = 0;
            let mut prev_ply: i64 = -1000;
            while reader.has_next() && (lens.len() as u64) < target {
                let e = reader.next();
                let ply = e.ply as i64;
                if cur == 0 {
                    cur = 1;
                } else if ply == prev_ply + 1 {
                    cur += 1;
                } else {
                    lens.push(cur);
                    cur = 1;
                }
                prev_ply = ply;
            }
            if cur > 0 { lens.push(cur); }
            let n = lens.len().max(1);
            let total: u64 = lens.iter().map(|&x| x as u64).sum();
            let avg = total as f64 / n as f64;
            let singletons = lens.iter().filter(|&&x| x == 1).count();
            let maxl = lens.iter().copied().max().unwrap_or(0);
            let minl = lens.iter().copied().min().unwrap_or(0);
            println!("File: {} ({:.2} GB)", input, file_size as f64 / 1_073_741_824.0);
            println!("Chains read:       {}", n);
            println!("Positions in them: {}", total);
            println!("Avg chain length:  {:.2}   (min {}, max {})", avg, minl, maxl);
            println!("Singletons (len=1): {} / {} ({:.1}%)", singletons, n,
                100.0 * singletons as f64 / n as f64);
            println!("(singletons/short chains => byte-heavy => over-weighted by size-weighted interleave)");
        }

        Some(Commands::Perft { depth, fen }) => {
            let fen_str = if fen.is_empty() {
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()
            } else {
                fen.join(" ")
            };
            let mut board = Board::from_fen(&fen_str);
            println!("Position: {}", fen_str);
            println!("Depth: {}", depth);
            println!();
            let start = std::time::Instant::now();
            let nodes = perft_divide(&mut board, depth);
            let elapsed = start.elapsed();
            let nps = if elapsed.as_secs_f64() > 0.0 {
                (nodes as f64 / elapsed.as_secs_f64()) as u64
            } else { 0 };
            println!("\nNodes: {}", nodes);
            println!("Time: {:.3}s", elapsed.as_secs_f64());
            println!("NPS: {}", nps);
        }

        Some(Commands::PerftBench) => {
            run_perft_bench();
        }

        Some(Commands::PatchNet { input, output, set_hl_crelu, clear_hl_crelu,
                                   set_ft_screlu, clear_ft_screlu, inspect }) => {
            if set_hl_crelu && clear_hl_crelu {
                eprintln!("Error: --set-hl-crelu and --clear-hl-crelu are mutually exclusive");
                std::process::exit(1);
            }
            if set_ft_screlu && clear_ft_screlu {
                eprintln!("Error: --set-ft-screlu and --clear-ft-screlu are mutually exclusive");
                std::process::exit(1);
            }
            let mut data = match std::fs::read(&input) {
                Ok(d) => d,
                Err(e) => { eprintln!("Error reading {}: {}", input, e); std::process::exit(1); }
            };
            // Header layout: magic(4) + version(4) + flags(1) + ...
            if data.len() < 9 {
                eprintln!("Error: file too short to be a .nnue"); std::process::exit(1);
            }
            let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
            let flags = data[8];
            let extended_kb = flags & 128 != 0;
            let bit5 = flags & 32 != 0;
            let bit0 = flags & 1 != 0;
            let bit5_desc = if extended_kb {
                format!("hl_crelu={}", bit5)
            } else {
                format!("consensus_buckets={} (legacy)", bit5)
            };
            println!("{}: version={} flags=0b{:08b} extended_kb={} ft_screlu={} bit5: {}",
                input, version, flags, extended_kb, bit0, bit5_desc);
            if inspect { return; }
            let patching_bit5 = set_hl_crelu || clear_hl_crelu;
            let patching_bit0 = set_ft_screlu || clear_ft_screlu;
            if !patching_bit5 && !patching_bit0 {
                eprintln!("Nothing to change. Pass --set-hl-crelu / --clear-hl-crelu / \
                    --set-ft-screlu / --clear-ft-screlu, or --inspect.");
                return;
            }
            if patching_bit5 && !extended_kb {
                eprintln!("Error: cannot patch bit 5 on legacy net (extended_kb=0 means \
                    bit 5 is consensus_buckets). Regenerate the net with --kb-layout or \
                    a non-16 kb_count to enable extended_kb.");
                std::process::exit(1);
            }
            let mut new_flags = flags;
            if set_hl_crelu { new_flags |= 32; }
            if clear_hl_crelu { new_flags &= !32; }
            if set_ft_screlu { new_flags |= 1; }
            if clear_ft_screlu { new_flags &= !1; }
            if new_flags == flags {
                println!("(no change — flags already in requested state)");
                return;
            }
            data[8] = new_flags;
            let dest = output.as_deref().unwrap_or(&input);
            if let Err(e) = std::fs::write(dest, &data) {
                eprintln!("Error writing {}: {}", dest, e); std::process::exit(1);
            }
            let mut ops = Vec::<&str>::new();
            if set_hl_crelu { ops.push("set hl_crelu"); }
            if clear_hl_crelu { ops.push("clear hl_crelu"); }
            if set_ft_screlu { ops.push("set ft_screlu"); }
            if clear_ft_screlu { ops.push("clear ft_screlu"); }
            println!("Patched: {} flags 0b{:08b} → 0b{:08b} ({})",
                dest, flags, new_flags, ops.join(" + "));
        }

        Some(Commands::TuneSpec { r_end, core }) => {
            for (name, _, default, min, max, c_end, is_core) in search::tunable_params() {
                if core && !is_core { continue; }
                println!("{}, int, {}, {}, {}, {}, {}", name, default, min, max, c_end, r_end);
            }
        }

        Some(Commands::FetchNet) => {
            run_fetch_net();
        }

        Some(Commands::FuzzThreats { count, seed, max_plies, verbose, postfix }) => {
            run_fuzz_threats(count, seed, max_plies, verbose, postfix);
        }

        Some(Commands::CheckNet {}) => {
            match &cli.nnue {
                Some(path) => run_check_net(path),
                None => {
                    eprintln!("Error: no NNUE net specified. Use --nnue/-n <path>");
                    std::process::exit(1);
                }
            }
        }

        Some(Commands::MeasureNetSparsity { input }) => {
            run_measure_net_sparsity(&input);
        }

        Some(Commands::DumpNetStats { input, per_bucket, per_kb, histogram }) => {
            run_dump_net_stats(&input, per_bucket, per_kb, histogram);
        }

        Some(Commands::Datagen { output, depth, games, threads, hash, blunder, force_captures, epd }) => {
            let nnue_path = cli.nnue.unwrap_or_default();
            let mode = if let Some(epd_path) = epd {
                datagen::DatagenMode::Material { source_epd: epd_path }
            } else {
                datagen::DatagenMode::SelfPlay { blunder_rate: blunder, force_capture_rate: force_captures }
            };
            let config = datagen::DatagenConfig {
                nnue_path,
                output_path: output,
                mode,
                depth,
                num_games: games,
                threads,
                hash_mb: hash,
            };
            datagen::run_datagen(&config);
        }

        Some(Commands::DumpThreats { fen }) => {
            let mut board = board::Board::new();
            board.set_fen(&fen);
            let occ = board.colors[0] | board.colors[1];
            let piece_names = ["P", "N", "B", "R", "Q", "K"];
            let color_names = ["w", "b"];
            let sq_name = |sq: u32| -> String {
                let file = (b'a' + (sq % 8) as u8) as char;
                let rank = (b'1' + (sq / 8) as u8) as char;
                format!("{}{}", file, rank)
            };

            for pov in [types::WHITE, types::BLACK] {
                let king_sq = (board.pieces[types::KING as usize] & board.colors[pov as usize]).trailing_zeros();
                let mirrored = (king_sq % 8) >= 4;
                println!("# POV={} king={} mirrored={}", color_names[pov as usize], sq_name(king_sq), mirrored);

                let mut features: Vec<(usize, String)> = Vec::new();
                threats::enumerate_threats(
                    &board.pieces, &board.colors, &board.mailbox,
                    occ, pov, mirrored,
                    |feat_idx| {
                        features.push((feat_idx, String::new()));
                    },
                );

                // Complete feature-index set (incl x-ray) for cross-implementation diffing
                // against Bullet's map_features. The detail loop below prints DIRECT threats
                // only, so this IDX line is the authoritative per-POV set.
                let mut idxs: Vec<usize> = features.iter().map(|(i, _)| *i).collect();
                idxs.sort_unstable();
                idxs.dedup();
                println!("IDX {}", idxs.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));

                // PSQ (HalfKA) feature-index set for this POV, consensus kb
                // layout, for cross-implementation diffing against Bullet's
                // ChessBucketsMirrored. The
                // threat IDX gate above says nothing about this half of the
                // input — it has its own mirror/bucket/perspective machinery.
                let (kb_tab, km_tab) = nnue::compute_king_buckets(nnue::KbLayout::Consensus);
                let mut psq: Vec<usize> = Vec::new();
                for color in [types::WHITE, types::BLACK] {
                    for pt in 0..6u8 {
                        let mut bb = board.pieces[pt as usize] & board.colors[color as usize];
                        while bb != 0 {
                            let sq = bb.trailing_zeros() as u8;
                            bb &= bb - 1;
                            psq.push(nnue::halfka_index_with(
                                &kb_tab, &km_tab, pov, king_sq as u8, color, pt, sq));
                        }
                    }
                }
                psq.sort_unstable();
                println!("PSQ {}", psq.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));

                // Re-enumerate with detail for printing
                let white_bb = board.colors[types::WHITE as usize];
                for color in [types::WHITE, types::BLACK] {
                    for pt in 0..6u8 {
                        let mut bb = board.pieces[pt as usize] & board.colors[color as usize];
                        while bb != 0 {
                            let sq = bb.trailing_zeros();
                            bb &= bb - 1;
                            let attacks = threats::piece_attacks_occ(pt, color, sq, occ);
                            let mut attacked_occ = attacks & occ;
                            while attacked_occ != 0 {
                                let target_sq = attacked_occ.trailing_zeros();
                                attacked_occ &= attacked_occ - 1;
                                let victim_pt = board.mailbox[target_sq as usize];
                                if victim_pt >= 6 { continue; }
                                let victim_color = if white_bb & (1u64 << target_sq) != 0 { types::WHITE } else { types::BLACK };
                                let cp = threats::colored_piece(color, pt);
                                let vcp = threats::colored_piece(victim_color, victim_pt);
                                let idx = threats::threat_index(cp, sq, vcp, target_sq, mirrored, pov);
                                if idx >= 0 {
                                    println!("{} {}{}({}) {} {}{}({}) → index {}",
                                        color_names[color as usize], piece_names[pt as usize], sq_name(sq), cp,
                                        color_names[victim_color as usize], piece_names[victim_pt as usize], sq_name(target_sq), vcp,
                                        idx);
                                }
                            }
                        }
                    }
                }
                println!();
            }
        }

        Some(Commands::ProfileThreats { input, output, limit, net }) => {
            run_profile_threats(&input, &output, limit, net.as_deref());
        }

        Some(Commands::BinpackStats { input }) => {
            run_binpack_stats(&input);
        }

        Some(Commands::SamplePositions { input, output, count, rate, scores }) => {
            run_sample_positions(&input, &output, count, rate, scores);
        }

        Some(Commands::ImportTsv { input, output, fen_col, score_col, max_abs, repeat }) => {
            datagen::import_tsv(&input, &output, fen_col, score_col, max_abs, repeat);
        }

        Some(Commands::ImportPgn { input, output }) => {
            datagen::import_pgn(&input, &output);
        }
        Some(Commands::ImportGameTsv { input, output }) => {
            datagen::import_game_tsv(&input, &output);
        }

        Some(Commands::EvalDist { input, count, csv, quiet_only, shard, min_error, max_abs_lc0 }) => {
            run_eval_dist(&input, count, &cli.nnue, &csv, quiet_only, &shard, min_error, max_abs_lc0);
        }

        Some(Commands::EvalFens { input, output }) => {
            run_eval_fens(&input, &output, &cli.nnue);
        }

        Some(Commands::BinpackMaterial { input, max }) => {
            run_binpack_material(&input, max);
        }

        Some(Commands::ConvertBullet { input, output, screlu, pairwise, hidden, hidden2, int8l1, bucketed_hidden, ft_size, int16_hidden, dual, dual_l2, consensus_buckets, kb_layout, kb_count, threats, output_buckets, hl_crelu }) => {
            // Resolve king bucket layout and count. Explicit --kb-layout wins;
            // --consensus-buckets is the legacy path for 16-bucket consensus.
            let layout = if !kb_layout.is_empty() {
                match bullet_convert::KbLayout::from_name(&kb_layout) {
                    Ok(l) => l,
                    Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
                }
            } else if consensus_buckets {
                bullet_convert::KbLayout::Consensus
            } else {
                bullet_convert::KbLayout::Uniform
            };
            let count = if kb_count > 0 { kb_count } else { layout.default_count() };

            let result = if hidden > 0 {
                bullet_convert::convert_v7(&input, &output, screlu, pairwise, hidden, hidden2, int8l1, bucketed_hidden, ft_size, int16_hidden, dual, dual_l2, layout, count, threats, hl_crelu)
            } else {
                bullet_convert::convert_v5(&input, &output, screlu, pairwise, output_buckets, layout, count)
            };
            if let Err(e) = result {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }

        Some(Commands::ConvertCheckpoint { output, ft, l1, l2 }) => {
            let nnue_path = cli.nnue.as_deref().expect("ConvertCheckpoint requires --nnue/-n");
            if let Err(e) = nnue_export::nnue_to_bullet_checkpoint(nnue_path, &output, ft, l1, l2) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn run_perft_bench() {
    let positions = [
        ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324u64),
        ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690),
        ("pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083),
        ("pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292),
        ("pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194),
        ("pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551),
    ];

    let start = std::time::Instant::now();
    let mut total_nodes = 0u64;
    let mut all_passed = true;

    for (name, fen, depth, expected) in &positions {
        let mut board = Board::from_fen(fen);
        let nodes = perft(&mut board, *depth);
        total_nodes += nodes;
        let passed = nodes == *expected;
        if !passed { all_passed = false; }
        println!("{:10} depth {} : {:>12} {} (expected {})",
            name, depth, nodes, if passed { "OK" } else { "FAIL" }, expected);
    }

    let elapsed = start.elapsed();
    let nps = (total_nodes as f64 / elapsed.as_secs_f64()) as u64;
    println!("\nTotal: {} nodes in {:.3}s = {} NPS", total_nodes, elapsed.as_secs_f64(), nps);
    if all_passed {
        println!("All perft tests PASSED");
    } else {
        println!("Some perft tests FAILED");
        std::process::exit(1);
    }
}

fn run_fuzz_threats(count: usize, seed: u64, max_plies: usize, verbose: usize, postfix: bool) {
    use board::Board;
    use threats::{
        enumerate_threats, enumerate_threats_bullet_postfix_ref,
        enumerate_threats_bullet_ref, num_threat_features,
    };
    use types::{WHITE, BLACK, KING};

    // Threat tables are initialised by init_engine() — ensure that ran.
    // (main.rs calls it once at startup; fuzz-threats goes through that
    // path because the subcommand dispatch is after init.)
    let nf = num_threat_features();
    assert!(nf > 0, "threat tables not initialised");

    // Deterministic PRNG — xorshift64.
    let mut state: u64 = if seed == 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap()
            .as_nanos() as u64
    } else { seed };
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut total_positions = 0usize;
    let mut total_mismatches = 0usize;
    let mut mismatches_by_stm = [0usize; 2];
    let mut printed = 0usize;

    for _ in 0..count {
        let mut board = Board::startpos();
        let plies = (next() as usize) % max_plies.max(1);
        for _ in 0..plies {
            let moves = crate::movegen::generate_legal_moves(&board);
            if moves.len == 0 { break; }
            let pick = (next() as usize) % moves.len;
            board.make_move(moves.get(pick));
            // Skip positions where only king is left (rare terminal)
            if (board.pieces[KING as usize]).count_ones() < 2 { break; }
        }

        total_positions += 1;
        let real_stm = board.side_to_move;
        let occ = board.colors[0] | board.colors[1];

        for pov in [WHITE, BLACK] {
            let our_k_sq = (board.pieces[KING as usize] & board.colors[pov as usize]).trailing_zeros();
            let mirrored = (our_k_sq & 7) >= 4;

            let mut set_coda = std::collections::BTreeSet::<usize>::new();
            enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, pov, mirrored,
                |idx| { set_coda.insert(idx); },
            );

            let mut set_ref = std::collections::BTreeSet::<usize>::new();
            if postfix {
                enumerate_threats_bullet_postfix_ref(
                    &board.pieces, &board.colors, &board.mailbox,
                    occ, pov, mirrored, real_stm,
                    |idx| { set_ref.insert(idx); },
                );
            } else {
                enumerate_threats_bullet_ref(
                    &board.pieces, &board.colors, &board.mailbox,
                    occ, pov, mirrored, real_stm,
                    |idx| { set_ref.insert(idx); },
                );
            }

            if set_coda != set_ref {
                total_mismatches += 1;
                mismatches_by_stm[real_stm as usize] += 1;
                if printed < verbose {
                    let only_coda: Vec<_> = set_coda.difference(&set_ref).collect();
                    let only_ref: Vec<_> = set_ref.difference(&set_coda).collect();
                    println!("MISMATCH: fen={} pov={} real_stm={} coda_only={} ref_only={}",
                        board.to_fen(),
                        if pov == WHITE { "W" } else { "B" },
                        if real_stm == WHITE { "W" } else { "B" },
                        only_coda.len(), only_ref.len());
                    for i in only_coda.iter().take(3) { println!("  only in coda: {}", i); }
                    for i in only_ref.iter().take(3) { println!("  only in ref:  {}", i); }
                    printed += 1;
                }
            }
        }
    }

    let total_evals = total_positions * 2;
    let ref_label = if postfix { "post-C8fix-2 Bullet (62931d1)" } else { "pre-C8fix-2 Bullet (bf-frame)" };
    println!("Fuzz complete (reference: {}).", ref_label);
    println!("  positions fuzzed  : {}", total_positions);
    println!("  evals compared    : {} (2 povs per position)", total_evals);
    println!("  mismatches        : {} ({:.2}%)", total_mismatches,
        100.0 * total_mismatches as f64 / total_evals.max(1) as f64);
    println!("  by real STM       : W={} B={}", mismatches_by_stm[0], mismatches_by_stm[1]);
}

fn run_fetch_net() {
    let net_txt = std::path::Path::new("net.txt");
    if !net_txt.exists() {
        eprintln!("Error: net.txt not found in current directory");
        std::process::exit(1);
    }
    let url = std::fs::read_to_string(net_txt).unwrap().trim().to_string();
    if url.is_empty() {
        eprintln!("Error: net.txt is empty");
        std::process::exit(1);
    }
    let fname = url.rsplit('/').next().unwrap_or("net.nnue");
    let out_path = std::path::Path::new(fname);
    if out_path.exists() {
        println!("{} already exists, skipping download", fname);
        return;
    }
    println!("Downloading {} ...", url);
    // C8 audit LIKELY #48: add `-f` (--fail) so curl exits non-zero on
    // HTTP 4xx/5xx instead of capturing the error body as the output
    // file. Previously a 404/500 response was silently saved as the
    // "net" and curl returned success; the subsequent load would fail
    // cryptically or — worse — truncate a partial download into what
    // looked like a valid file. Also validate magic bytes after download.
    let output = std::process::Command::new("curl")
        .args(["-fsL", &url, "-o", fname])
        .status();
    match output {
        Ok(status) if status.success() => {
            let size = std::fs::metadata(fname).map(|m| m.len()).unwrap_or(0);
            // Magic-byte positive check: Coda .nnue files start with the
            // 4-byte magic 'EUNN' (= 0x4E4E5545 little-endian). Reject
            // anything that doesn't match — covers HTML error pages and
            // partial / wrong downloads. Original audit fix used a
            // negative ASCII-alpha check, but the magic is itself ASCII
            // alpha so it falsely rejected every valid NNUE.
            const NNUE_MAGIC: &[u8; 4] = b"EUNN";
            if let Ok(bytes) = std::fs::read(fname) {
                if bytes.len() < 16 || &bytes[0..4] != NNUE_MAGIC {
                    eprintln!("Error: downloaded file doesn't look like a .nnue (first bytes: {:?}; expected magic 'EUNN')",
                        &bytes.iter().take(16).collect::<Vec<_>>());
                    let _ = std::fs::remove_file(fname);
                    std::process::exit(1);
                }
            }
            println!("Downloaded {} ({} bytes)", fname, size);
        }
        _ => {
            eprintln!("Error: failed to download {}", url);
            // Remove any partial file to avoid confusing the next invocation.
            let _ = std::fs::remove_file(fname);
            std::process::exit(1);
        }
    }
}

fn run_check_net(net_path: &str) {
    use nnue::{NNUENet, NNUEAccumulator};

    let net = match NNUENet::load(net_path) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Error loading net: {}", e);
            std::process::exit(1);
        }
    };

    let mut arch = format!("FT={}", net.hidden_size);
    if net.l1_size > 0 { arch += &format!(" L1={}", net.l1_size); }
    if net.l2_size > 0 { arch += &format!(" L2={}", net.l2_size); }
    if net.use_screlu { arch += " SCReLU"; }
    if net.use_pairwise { arch += " pairwise"; }
    if net.l1_scale == 64 { arch += " int8L1"; }
    println!("Net: {} ({})", net_path, arch);
    println!();

    let eval_fen = |fen: &str| -> i32 {
        let board = Board::from_fen(fen);
        let h = net.hidden_size;
        let mut acc = NNUEAccumulator::new(h);
        // Full recompute for both perspectives
        let piece_count = board.occupied().count_ones();
        acc.force_recompute(&net, &board);
        // C8 audit LIKELY #20: for v9 threat nets, use forward_with_threats
        // so the displayed eval actually consumes the threat features.
        // `net.forward()` with a v9 net zeroes the threat half → displayed
        // eval is arbitrarily wrong. Build a one-shot threat stack for the
        // position.
        if net.has_threats {
            let mut ts = crate::threat_accum::ThreatStack::new(h);
            ts.active = true;
            ts.ensure_computed(&net.threat_weights, net.num_threat_features, &board);
            net.forward_with_threats(&acc, board.side_to_move, piece_count, &ts)
        } else {
            net.forward(&acc, board.side_to_move, piece_count)
        }
    };

    // === Eval positions ===
    let startpos = eval_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let miss_pawn = eval_fen("rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1");
    let miss_knight = eval_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1");
    // Imbalances: White missing queen, Black missing rook (White should be losing)
    let q_vs_r = eval_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBN1 w Qkq - 0 1");
    // Endgame: KR vs K (White winning)
    let eg_rook_up = eval_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1");
    // Endgame: KR vs K (Black winning)
    let eg_rook_down = eval_fen("r3k3/8/8/8/8/8/8/4K3 w - - 0 1");

    println!("{:<22} {:>8}", "Position", "Score");
    println!("{:<22} {:>8}", "--------", "--------");
    println!("{:<22} {:>8}", "startpos", startpos);
    println!("{:<22} {:>8}", "miss pawn (W)", miss_pawn);
    println!("{:<22} {:>8}", "miss knight (W)", miss_knight);
    println!("{:<22} {:>8}", "Q vs R (W down Q+R)", q_vs_r);
    println!("{:<22} {:>8}", "EG rook up (W)", eg_rook_up);
    println!("{:<22} {:>8}", "EG rook down (W)", eg_rook_down);
    println!();

    // === Health checks (relative, not absolute) ===
    let mut pass = 0;
    let mut fail = 0;

    let check = |name: &str, ok: bool, pass: &mut i32, fail: &mut i32| {
        if ok { *pass += 1; print!("  PASS"); } else { *fail += 1; print!("  FAIL"); }
        println!("  {}", name);
    };

    println!("Health checks:");
    // 1. Startpos should be near zero (not wildly off)
    check("startpos near zero (|score| < 150)", startpos.abs() < 150, &mut pass, &mut fail);
    // 2. Missing a pawn should be worse than startpos
    check("miss pawn < startpos", miss_pawn < startpos, &mut pass, &mut fail);
    // 3. Missing a knight should be worse than missing a pawn
    check("miss knight < miss pawn", miss_knight < miss_pawn, &mut pass, &mut fail);
    // 4. Queen vs rook imbalance: side down the queen should be losing
    check("Q-vs-R imbalance negative", q_vs_r < 0, &mut pass, &mut fail);
    // 5. EG rook up should be positive (winning)
    check("EG rook up positive", eg_rook_up > 0, &mut pass, &mut fail);
    // 6. EG rook down should be negative (losing)
    check("EG rook down negative", eg_rook_down < 0, &mut pass, &mut fail);
    // 7. EG rook up and down should have opposite signs and similar magnitude
    check("EG symmetry (opposite signs)", eg_rook_up > 0 && eg_rook_down < 0, &mut pass, &mut fail);
    // 8. Differentiation: not all scores identical (collapsed net detection)
    let scores = [startpos, miss_pawn, miss_knight, q_vs_r, eg_rook_up, eg_rook_down];
    let all_same = scores.windows(2).all(|w| (w[0] - w[1]).abs() < 10);
    check("scores differentiated (not collapsed)", !all_same, &mut pass, &mut fail);

    println!();
    if fail == 0 {
        println!("All {} checks passed.", pass);
    } else {
        println!("{} passed, {} FAILED.", pass, fail);
    }
}

/// Measure threat-matrix row sparsity in a v9+ .nnue file.
/// Used for L1-regularisation probe diagnosis. Prints zero-row count,
/// near-zero rows (|max_abs| ≤ 1), and histogram of row max-abs weights.
fn run_measure_net_sparsity(net_path: &str) {
    use crate::nnue::NNUENet;
    let net = match NNUENet::load(net_path) {
        Ok(n) => n,
        Err(e) => { eprintln!("load error: {}", e); std::process::exit(1); }
    };
    let num_features = net.num_threat_features;
    if num_features == 0 {
        println!("Net has no threat features (not a v9+ net).");
        return;
    }
    let h = net.hidden_size;
    let mut zero_rows = 0usize;
    let mut near_zero_rows = 0usize;  // |max| <= 1
    let mut hist: [usize; 6] = [0; 6];  // 0, 1, 2-3, 4-7, 8-31, 32+
    let mut nonzero_weights = 0u64;
    let mut total_abs_sum = 0u64;
    for r in 0..num_features {
        let row = &net.threat_weights[r * h..(r + 1) * h];
        let max_abs = row.iter().map(|&w| (w as i32).abs()).max().unwrap_or(0);
        if max_abs == 0 { zero_rows += 1; }
        if max_abs <= 1 { near_zero_rows += 1; }
        let bucket = match max_abs {
            0 => 0, 1 => 1, 2..=3 => 2, 4..=7 => 3, 8..=31 => 4, _ => 5,
        };
        hist[bucket] += 1;
        for &w in row {
            if w != 0 { nonzero_weights += 1; }
            total_abs_sum += (w as i32).unsigned_abs() as u64;
        }
    }
    let total_rows = num_features;
    let total_weights = (num_features * h) as u64;
    println!("=== Threat matrix sparsity: {} ===", net_path);
    println!("Rows:  {} total, hidden_size={}", total_rows, h);
    println!("Zero rows:      {} ({:.2}%)", zero_rows,
        100.0 * zero_rows as f64 / total_rows as f64);
    println!("Near-zero rows: {} ({:.2}%)  [|max| <= 1]", near_zero_rows,
        100.0 * near_zero_rows as f64 / total_rows as f64);
    println!("Row-max-abs histogram:");
    println!("  0       : {}", hist[0]);
    println!("  1       : {}", hist[1]);
    println!("  2-3     : {}", hist[2]);
    println!("  4-7     : {}", hist[3]);
    println!("  8-31    : {}", hist[4]);
    println!("  32+     : {}", hist[5]);
    println!("Weight-level sparsity:");
    println!("  Zero weights:    {} / {} ({:.2}%)",
        total_weights - nonzero_weights, total_weights,
        100.0 * (total_weights - nonzero_weights) as f64 / total_weights as f64);
    println!("  Mean |weight|:   {:.3}", total_abs_sum as f64 / total_weights as f64);
    let matrix_bytes = (num_features * h) as u64;  // i8 = 1 byte per weight
    let compact_bytes = ((total_rows - zero_rows) * h) as u64;
    println!("Matrix size: {} MB raw, {} MB compact (post zero-row compaction)",
        matrix_bytes / (1024 * 1024), compact_bytes / (1024 * 1024));
}

/// Per-layer weight statistics. All inputs treated as i64 internally for
/// safety when summing over millions of weights.
struct LayerStats {
    n: u64,
    mean: f64,
    abs_mean: f64,
    stdev: f64,
    min: i64,
    max: i64,
    p50_abs: i64,
    p99_abs: i64,
    pct_zero: f64,
    dead_rows: usize,    // rows where all weights are zero (only if rows > 1)
    near_dead_rows: usize, // rows where max|w| <= 1
    total_rows: usize,
}

fn compute_stats(weights: &[i64], row_size: usize) -> LayerStats {
    let n = weights.len() as u64;
    if n == 0 {
        return LayerStats {
            n: 0, mean: 0.0, abs_mean: 0.0, stdev: 0.0, min: 0, max: 0,
            p50_abs: 0, p99_abs: 0, pct_zero: 0.0,
            dead_rows: 0, near_dead_rows: 0, total_rows: 0,
        };
    }
    let mut sum: i128 = 0;
    let mut abs_sum: i128 = 0;
    let mut min: i64 = i64::MAX;
    let mut max: i64 = i64::MIN;
    let mut zero_count: u64 = 0;
    for &w in weights {
        sum += w as i128;
        abs_sum += w.unsigned_abs() as i128;
        if w < min { min = w; }
        if w > max { max = w; }
        if w == 0 { zero_count += 1; }
    }
    let mean = sum as f64 / n as f64;
    let abs_mean = abs_sum as f64 / n as f64;
    // stdev: second pass for accuracy
    let mut sq_sum: f64 = 0.0;
    for &w in weights {
        let d = w as f64 - mean;
        sq_sum += d * d;
    }
    let stdev = (sq_sum / n as f64).sqrt();
    // percentiles of |w|
    let mut abs_vals: Vec<i64> = weights.iter().map(|&w| w.abs()).collect();
    abs_vals.sort_unstable();
    let p50_abs = abs_vals[abs_vals.len() / 2];
    let p99_idx = ((abs_vals.len() as f64 * 0.99) as usize).min(abs_vals.len() - 1);
    let p99_abs = abs_vals[p99_idx];
    let pct_zero = 100.0 * zero_count as f64 / n as f64;
    let total_rows = if row_size > 0 { weights.len() / row_size } else { 0 };
    let mut dead_rows = 0usize;
    let mut near_dead_rows = 0usize;
    if row_size > 0 {
        for r in 0..total_rows {
            let row = &weights[r * row_size..(r + 1) * row_size];
            let mx = row.iter().map(|w| w.abs()).max().unwrap_or(0);
            if mx == 0 { dead_rows += 1; }
            if mx <= 1 { near_dead_rows += 1; }
        }
    }
    LayerStats {
        n, mean, abs_mean, stdev, min, max, p50_abs, p99_abs, pct_zero,
        dead_rows, near_dead_rows, total_rows,
    }
}

fn print_stats_line(label: &str, s: &LayerStats, dtype: &str) {
    let row_info = if s.total_rows > 0 {
        format!(" rows={} dead={} near_dead={}",
            s.total_rows, s.dead_rows, s.near_dead_rows)
    } else { String::new() };
    println!("  {:<32} n={:>9} {:<4} mean={:+8.3} |mean|={:7.3} stdev={:8.3} min={:>6} max={:>6} |p50|={:>4} |p99|={:>5} zero={:5.2}%{}",
        label, s.n, dtype,
        s.mean, s.abs_mean, s.stdev, s.min, s.max,
        s.p50_abs, s.p99_abs, s.pct_zero, row_info);
}

fn print_histogram(label: &str, weights: &[i64]) {
    if weights.is_empty() { return; }
    let max_abs = weights.iter().map(|w| w.abs()).max().unwrap_or(0);
    if max_abs == 0 {
        println!("  {} histogram: ALL ZERO", label);
        return;
    }
    // Log-spaced bins by |w|: 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-127, 128-255, 256+
    let mut bins = [0u64; 10];
    for &w in weights {
        let a = w.unsigned_abs();
        let bin = if a == 0 { 0 }
                  else if a == 1 { 1 }
                  else if a <= 3 { 2 }
                  else if a <= 7 { 3 }
                  else if a <= 15 { 4 }
                  else if a <= 31 { 5 }
                  else if a <= 63 { 6 }
                  else if a <= 127 { 7 }
                  else if a <= 255 { 8 }
                  else { 9 };
        bins[bin] += 1;
    }
    let n = weights.len() as f64;
    let labels = ["0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256+"];
    let bin_str: Vec<String> = (0..10).map(|i| {
        format!("{}:{:.1}%", labels[i], 100.0 * bins[i] as f64 / n)
    }).collect();
    println!("  {} histogram: {}", label, bin_str.join("  "));
}

fn run_dump_net_stats(net_path: &str, per_bucket: bool, per_kb: bool, histogram: bool) {
    use crate::nnue::NNUENet;
    let net = match NNUENet::load(net_path) {
        Ok(n) => n,
        Err(e) => { eprintln!("load error: {}", e); std::process::exit(1); }
    };
    println!("=== {} ===", net_path);
    println!("hidden_size={} num_king_buckets={} num_threat_features={} has_threats={}",
        net.hidden_size, net.num_king_buckets, net.num_threat_features, net.has_threats);
    println!("l1_per_bucket={} l2_per_bucket={} bucketed_hidden={} l1_scale={} screlu={} pairwise={} dual_l1={}",
        net.l1_per_bucket, net.l2_per_bucket, net.bucketed_hidden, net.l1_scale,
        net.use_screlu, net.use_pairwise, net.dual_l1);

    let h = net.hidden_size;
    let kb = net.num_king_buckets;
    let psq_per_bucket = 768 * h;

    // l0w PSQ — overall
    let psq_w: Vec<i64> = net.input_weights.iter().map(|&w| w as i64).collect();
    let s = compute_stats(&psq_w, h); // row = (kb, piece, sq) → h weights
    println!("\n[l0w PSQ — i16 @ QA={}]", 255i32);
    print_stats_line("all", &s, "i16");
    if histogram { print_histogram("l0w_psq", &psq_w); }
    if per_kb {
        for k in 0..kb {
            let block = &psq_w[k * psq_per_bucket..(k + 1) * psq_per_bucket];
            let s = compute_stats(block, h);
            print_stats_line(&format!("kb {}", k), &s, "i16");
        }
    }

    // l0b
    let l0b: Vec<i64> = net.input_biases.iter().map(|&w| w as i64).collect();
    let s = compute_stats(&l0b, 0);
    println!("\n[l0b — i16]");
    print_stats_line("all", &s, "i16");

    // threats
    if net.has_threats && net.num_threat_features > 0 {
        let tw: Vec<i64> = net.threat_weights.iter().map(|&w| w as i64).collect();
        let s = compute_stats(&tw, h);
        println!("\n[l0w THREATS — i8 @ QA={}, post-clamp]", 255i32);
        print_stats_line("all", &s, "i8");
        if histogram { print_histogram("threats", &tw); }
    }

    // L1 weights — bucketed_hidden means [l1_input × (BUCKETS × l1_per_bucket)]
    if net.l1_per_bucket > 0 {
        let l1_input = if net.use_pairwise { h } else { 2 * h };
        let bl1 = net.l1_size; // already includes BUCKETS factor
        let l1w: Vec<i64> = net.l1_weights.iter().map(|&w| w as i64).collect();
        let s = compute_stats(&l1w, bl1); // row = one input, bl1 cols
        println!("\n[l1w — i16 (i8 quantised), input={} × bl1={}]", l1_input, bl1);
        print_stats_line("all", &s, "i16");
        if histogram { print_histogram("l1w", &l1w); }
        if per_bucket && net.bucketed_hidden {
            let buckets = bl1 / net.l1_per_bucket;
            for b in 0..buckets {
                // Per-bucket slice: every row, columns [b*L1 .. (b+1)*L1]
                let mut col: Vec<i64> = Vec::with_capacity(l1_input * net.l1_per_bucket);
                for r in 0..l1_input {
                    let row_off = r * bl1;
                    col.extend_from_slice(&l1w[row_off + b * net.l1_per_bucket .. row_off + (b + 1) * net.l1_per_bucket]);
                }
                let s = compute_stats(&col, net.l1_per_bucket);
                print_stats_line(&format!("bucket {}", b), &s, "i16");
            }
        }
        let l1b: Vec<i64> = net.l1_biases.iter().map(|&w| w as i64).collect();
        let s = compute_stats(&l1b, net.l1_per_bucket);
        println!("\n[l1b — i16, total={}]", bl1);
        print_stats_line("all", &s, "i16");
        if per_bucket && net.bucketed_hidden {
            let buckets = bl1 / net.l1_per_bucket;
            for b in 0..buckets {
                let s = compute_stats(&l1b[b * net.l1_per_bucket .. (b + 1) * net.l1_per_bucket], 0);
                print_stats_line(&format!("bucket {}", b), &s, "i16");
            }
        }
    }

    // L2 weights — float
    if net.l2_per_bucket > 0 {
        let bl2 = net.l2_size;
        let l2_input = if net.dual_l1 { net.l1_per_bucket * 2 } else { net.l1_per_bucket };
        // l2_weights_f scale: divided by qa_l1 already
        let l2w_scaled: Vec<i64> = net.l2_weights_f.iter().map(|&w| (w * 1000.0).round() as i64).collect();
        let s = compute_stats(&l2w_scaled, bl2);
        println!("\n[l2w — f32 (×1000 for stat display), input={} × bl2={}]", l2_input, bl2);
        print_stats_line("all", &s, "milli");
        if histogram { print_histogram("l2w_milli", &l2w_scaled); }
        if per_bucket && net.bucketed_hidden {
            let buckets = bl2 / net.l2_per_bucket;
            for b in 0..buckets {
                let mut col: Vec<i64> = Vec::with_capacity(l2_input * net.l2_per_bucket);
                for r in 0..l2_input {
                    let row_off = r * bl2;
                    col.extend_from_slice(&l2w_scaled[row_off + b * net.l2_per_bucket .. row_off + (b + 1) * net.l2_per_bucket]);
                }
                let s = compute_stats(&col, net.l2_per_bucket);
                print_stats_line(&format!("bucket {}", b), &s, "milli");
            }
        }
        let l2b_scaled: Vec<i64> = net.l2_biases_f.iter().map(|&w| (w * 1000.0).round() as i64).collect();
        let s = compute_stats(&l2b_scaled, net.l2_per_bucket);
        println!("\n[l2b — f32 (×1000), total={}]", bl2);
        print_stats_line("all", &s, "milli");
        if per_bucket && net.bucketed_hidden {
            let buckets = bl2 / net.l2_per_bucket;
            for b in 0..buckets {
                let s = compute_stats(&l2b_scaled[b * net.l2_per_bucket .. (b + 1) * net.l2_per_bucket], 0);
                print_stats_line(&format!("bucket {}", b), &s, "milli");
            }
        }
    }

    // Output weights — per output bucket (8)
    let out_width = if net.l2_per_bucket > 0 { net.l2_per_bucket }
                    else if net.l1_per_bucket > 0 { net.l1_per_bucket }
                    else if net.use_pairwise { h } else { 2 * h };
    let outw: Vec<i64> = net.output_weights.iter().map(|&w| w as i64).collect();
    let s = compute_stats(&outw, out_width);
    println!("\n[output_weights — i16 @ QB=64, BUCKETS={} × out_width={}]",
        crate::nnue::NNUE_OUTPUT_BUCKETS, out_width);
    print_stats_line("all", &s, "i16");
    if histogram { print_histogram("output_w", &outw); }
    if per_bucket {
        for b in 0..crate::nnue::NNUE_OUTPUT_BUCKETS {
            let s = compute_stats(&outw[b * out_width .. (b + 1) * out_width], 0);
            print_stats_line(&format!("bucket {}", b), &s, "i16");
        }
    }

    // Output bias
    println!("\n[output_bias — i32 @ QA*QB]");
    for b in 0..crate::nnue::NNUE_OUTPUT_BUCKETS {
        println!("  bucket {} = {:>10} ({:+.4} cp scaled)",
            b, net.output_bias[b],
            net.output_bias[b] as f64 / (255.0 * 64.0));
    }
}

fn run_profile_threats(input: &str, output: &str, limit: usize, net_path: Option<&str>) {
    use std::io::{BufRead, Write};

    let total_features = threats::num_threat_features();
    println!("Threat features: {}", total_features);
    println!("Reading positions from {}", input);

    let file = std::fs::File::open(input)
        .unwrap_or_else(|_| panic!("Failed to open {}", input));
    let reader = std::io::BufReader::new(file);

    let mut hits: Vec<u64> = vec![0; total_features];
    let mut positions = 0usize;
    let mut total_firings: u64 = 0;
    let mut max_per_pos: u32 = 0;

    for line in reader.lines() {
        let line = match line { Ok(l) => l, Err(_) => continue };
        let line = line.trim();
        if line.is_empty() { continue; }

        let mut board = board::Board::new();
        board.set_fen(line);

        let occ = board.colors[0] | board.colors[1];
        let mut pos_firings: u32 = 0;

        for pov in [types::WHITE, types::BLACK] {
            let king_bb = board.pieces[types::KING as usize] & board.colors[pov as usize];
            if king_bb == 0 { continue; }
            let king_sq = king_bb.trailing_zeros();
            let mirrored = (king_sq % 8) >= 4;

            threats::enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, pov, mirrored,
                |feat_idx| {
                    if feat_idx < total_features {
                        hits[feat_idx] += 1;
                        pos_firings += 1;
                    }
                },
            );
        }

        total_firings += pos_firings as u64;
        if pos_firings > max_per_pos { max_per_pos = pos_firings; }

        positions += 1;
        if positions.is_multiple_of(100_000) {
            eprint!("\r  processed {} positions, {} firings", positions, total_firings);
        }
        if limit > 0 && positions >= limit { break; }
    }
    eprintln!();

    if positions == 0 {
        println!("No positions processed");
        return;
    }

    let dead = hits.iter().filter(|&&h| h == 0).count();
    let cold = hits.iter().filter(|&&h| h > 0 && h < 10).count();
    let avg_per_pos = total_firings as f64 / positions as f64;
    let avg_per_feature = total_firings as f64 / total_features as f64;

    println!();
    println!("=== Position summary ===");
    println!("Positions processed:   {}", positions);
    println!("Total threat firings:  {}", total_firings);
    println!("Avg firings/position:  {:.2}  (max {})", avg_per_pos, max_per_pos);
    println!();
    println!("=== Feature usage ===");
    println!("Total feature indices: {}", total_features);
    println!("Dead (0 hits):         {} ({:.1}%)", dead, 100.0 * dead as f64 / total_features as f64);
    println!("Cold (1-9 hits):       {} ({:.1}%)", cold, 100.0 * cold as f64 / total_features as f64);
    println!("Avg firings/feature:   {:.1}", avg_per_feature);

    let mut idx_by_hits: Vec<(usize, u64)> = hits.iter().enumerate().map(|(i, &h)| (i, h)).collect();
    idx_by_hits.sort_by(|a, b| b.1.cmp(&a.1));

    println!();
    println!("=== Concentration ===");
    for &pct in &[1.0_f64, 5.0, 10.0, 20.0, 50.0] {
        let k = ((total_features as f64) * pct / 100.0) as usize;
        let sum: u64 = idx_by_hits.iter().take(k).map(|(_, h)| *h).sum();
        let share = 100.0 * sum as f64 / total_firings.max(1) as f64;
        println!("Top {:>4.1}% ({:>5} features): {:>6.2}% of all firings",
            pct, k, share);
    }

    println!();
    println!("=== Top 20 hottest features ===");
    println!("{:>6}  {:>12}  {:>8}", "idx", "hits", "share%");
    for &(idx, h) in idx_by_hits.iter().take(20) {
        let share = 100.0 * h as f64 / total_firings.max(1) as f64;
        println!("{:>6}  {:>12}  {:>7.3}%", idx, h, share);
    }

    // Build reverse map: feature idx → (attacker_cp, victim_cp).
    // Enumerate (attacker_cp, attacker_sq, victim_cp, victim_sq, mirrored)
    // with pov=WHITE (the encoding's canonical perspective).
    println!();
    println!("=== Building reverse map (idx → piece-pair) ===");
    let mut idx_to_pair: Vec<Option<(u8, u8)>> = vec![None; total_features];
    let mut indices_per_pair: [[u32; 12]; 12] = [[0; 12]; 12];

    // Brute-force enumeration over all (a_cp, a_sq, v_cp, v_sq, mir) to map every
    // reachable idx to its owning (a_cp, v_cp) pair. We don't restrict to attack patterns
    // because the encoder's ATTACK_INDEX_LOOKUP produces non-negative idx even for
    // (from, to) outside the empty-board attack set; some indices in the allocated
    // address space are only reachable via tuples that don't correspond to legal attacks
    // (encoder allocates per-piece × per-channel slots regardless of attack-pattern fit).
    for attacker_cp in 0..12u8 {
        let attacker_pt = attacker_cp % 6;
        for attacker_sq in 0..64u32 {
            if attacker_pt == 0 && !(8..56).contains(&attacker_sq) { continue; }
            for victim_cp in 0..12u8 {
                let victim_pt = victim_cp % 6;
                for victim_sq in 0..64u32 {
                    if victim_sq == attacker_sq { continue; }
                    if victim_pt == 0 && !(8..56).contains(&victim_sq) { continue; }
                    for mirrored in [false, true] {
                        let idx = threats::threat_index(
                            attacker_cp as usize, attacker_sq,
                            victim_cp as usize, victim_sq,
                            mirrored, types::WHITE,
                        );
                        if idx >= 0 && (idx as usize) < total_features {
                            let u = idx as usize;
                            if idx_to_pair[u].is_none() {
                                idx_to_pair[u] = Some((attacker_cp, victim_cp));
                                indices_per_pair[attacker_cp as usize][victim_cp as usize] += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    let mapped = idx_to_pair.iter().filter(|x| x.is_some()).count();
    println!("Reverse-mapped: {}/{} indices ({} unmapped)",
        mapped, total_features, total_features - mapped);

    // Aggregate dead/cold/hits per (attacker_cp, victim_cp) pair
    let mut pair_dead: [[u32; 12]; 12] = [[0; 12]; 12];
    let mut pair_cold: [[u32; 12]; 12] = [[0; 12]; 12];
    let mut pair_hits: [[u64; 12]; 12] = [[0; 12]; 12];
    let mut unmapped_dead = 0u32;
    let mut unmapped_total = 0u32;
    for idx in 0..total_features {
        let h = hits[idx];
        match idx_to_pair[idx] {
            Some((a, v)) => {
                pair_hits[a as usize][v as usize] += h;
                if h == 0 {
                    pair_dead[a as usize][v as usize] += 1;
                } else if h < 10 {
                    pair_cold[a as usize][v as usize] += 1;
                }
            }
            None => {
                unmapped_total += 1;
                if h == 0 { unmapped_dead += 1; }
            }
        }
    }

    let pn = ["wP","wN","wB","wR","wQ","wK","bP","bN","bB","bR","bQ","bK"];

    println!();
    println!("=== Dead-feature breakdown by piece-pair ===");
    println!("(rows: attacker, cols: victim. Each cell = dead/total | cell%dead)");
    println!("(white-perspective canonical encoding; black-perspective tuples remap to same indices)");
    print!("{:>4}", "");
    for v in 0..12 { print!("  {:>10}", pn[v]); }
    println!();
    for a in 0..12 {
        print!("{:>4}", pn[a]);
        for v in 0..12 {
            let total = indices_per_pair[a][v];
            let dead = pair_dead[a][v];
            if total == 0 {
                print!("  {:>10}", "—");
            } else {
                print!("  {:>4}/{:<5}", dead, total);
            }
        }
        println!();
    }

    // Top-K dead pairs by absolute count
    let mut pair_summary: Vec<(u8, u8, u32, u32, u64)> = Vec::new();
    for a in 0..12 {
        for v in 0..12 {
            let total = indices_per_pair[a][v];
            if total > 0 {
                pair_summary.push((a as u8, v as u8, indices_per_pair[a][v], pair_dead[a][v], pair_hits[a][v]));
            }
        }
    }
    pair_summary.sort_by(|p, q| q.3.cmp(&p.3));

    println!();
    println!("=== Top 20 piece-pairs by absolute dead-feature count ===");
    println!("{:>4} → {:>4}   {:>8} {:>8} {:>8}   {:>14}",
        "attk", "vctm", "total", "dead", "%dead", "total_hits");
    for &(a, v, total, dead, h) in pair_summary.iter().take(20) {
        let pct = 100.0 * dead as f64 / total as f64;
        println!("{:>4} → {:>4}   {:>8} {:>8} {:>7.1}%   {:>14}",
            pn[a as usize], pn[v as usize], total, dead, pct, h);
    }

    // Fully-dead pairs (100% dead)
    println!();
    println!("=== Fully-dead piece-pairs (100% of features for this pair never fire) ===");
    let mut full_dead_count = 0u32;
    for &(a, v, total, dead, _h) in &pair_summary {
        if total > 0 && dead == total {
            println!("  {} → {}: {} dead features", pn[a as usize], pn[v as usize], total);
            full_dead_count += total;
        }
    }
    println!("Total features in fully-dead pairs: {} ({:.1}% of all features)",
        full_dead_count, 100.0 * full_dead_count as f64 / total_features as f64);

    if unmapped_total > 0 {
        println!();
        println!("Note: {} unmapped indices ({} dead) — phantom address-space slots", unmapped_total, unmapped_dead);
        println!("  the encoder allocates per piece × channel but no game state can produce.");
        println!("  These are CLEANLY DROPPABLE — no Elo risk. ({:.1}% of all features)",
            100.0 * unmapped_total as f64 / total_features as f64);
    }

    // Optional: correlate hits with NNUE row weight magnitudes
    if let Some(path) = net_path {
        use crate::nnue::NNUENet;
        match NNUENet::load(path) {
            Err(e) => {
                println!();
                println!("Failed to load net {} for magnitude correlation: {}", path, e);
            }
            Ok(net) => {
                if net.num_threat_features != total_features {
                    println!();
                    println!("Net threat-feature count {} != current encoding {}; skipping correlation",
                        net.num_threat_features, total_features);
                } else {
                    let h = net.hidden_size;
                    let mut row_l1: Vec<u32> = vec![0; total_features];
                    let mut row_max: Vec<u32> = vec![0; total_features];
                    for r in 0..total_features {
                        let row = &net.threat_weights[r * h..(r + 1) * h];
                        let mut l1 = 0u32;
                        let mut mx = 0u32;
                        for &w in row {
                            let a = (w as i32).unsigned_abs();
                            l1 += a;
                            if a > mx { mx = a; }
                        }
                        row_l1[r] = l1;
                        row_max[r] = mx;
                    }

                    println!();
                    println!("=== Hits × weight-magnitude correlation: {} ===", path);

                    // Confusion matrix: alive-by-hits × alive-by-magnitude
                    // alive-by-hits: hits >= 10
                    // alive-by-magnitude: max_abs >= 4
                    let mut hh_mh = 0u32; // high-hits, high-mag
                    let mut hh_lm = 0u32; // high-hits, low-mag
                    let mut lh_mh = 0u32; // low-hits, high-mag
                    let mut lh_lm = 0u32; // low-hits, low-mag
                    let mut zero_hits_zero_mag = 0u32;
                    let mut zero_hits_nonzero_mag = 0u32;
                    let mut nonzero_hits_zero_mag = 0u32;
                    for r in 0..total_features {
                        let h = hits[r];
                        let m = row_max[r];
                        if h == 0 {
                            if m == 0 { zero_hits_zero_mag += 1; }
                            else { zero_hits_nonzero_mag += 1; }
                        } else if m == 0 {
                            nonzero_hits_zero_mag += 1;
                        }
                        let alive_h = h >= 10;
                        let alive_m = m >= 4;
                        match (alive_h, alive_m) {
                            (true, true)   => hh_mh += 1,
                            (true, false)  => hh_lm += 1,
                            (false, true)  => lh_mh += 1,
                            (false, false) => lh_lm += 1,
                        }
                    }

                    println!("Pure agreement:");
                    println!("  zero hits AND zero |max|:                {} ({:.1}% of features)",
                        zero_hits_zero_mag, 100.0 * zero_hits_zero_mag as f64 / total_features as f64);
                    println!("  zero hits but NONZERO |max|:             {} (rows trained but never fire)",
                        zero_hits_nonzero_mag);
                    println!("  NONZERO hits but zero |max|:             {} (rows fire but lasso killed weights)",
                        nonzero_hits_zero_mag);
                    println!();
                    println!("Alive thresholds: hits>=10, |max|>=4");
                    println!("                       |max|>=4   |max|<4");
                    println!("  hits>=10        :    {:>8}   {:>8}", hh_mh, hh_lm);
                    println!("  hits<10         :    {:>8}   {:>8}", lh_mh, lh_lm);
                    let agreement = hh_mh + lh_lm;
                    let disagree  = hh_lm + lh_mh;
                    println!("  Agreement: {} ({:.1}%)  Disagree: {} ({:.1}%)",
                        agreement, 100.0 * agreement as f64 / total_features as f64,
                        disagree,  100.0 * disagree  as f64 / total_features as f64);

                    // Importance score: hits × L1
                    let mut score: Vec<(usize, u128, u64, u32)> = (0..total_features).map(|r| {
                        let s = hits[r] as u128 * row_l1[r] as u128;
                        (r, s, hits[r], row_l1[r])
                    }).collect();
                    score.sort_by(|a, b| b.1.cmp(&a.1));

                    println!();
                    println!("=== Concentration by importance (hits × |row|_1) ===");
                    let total_score: u128 = score.iter().map(|(_, s, _, _)| s).sum();
                    for &pct in &[1.0_f64, 5.0, 10.0, 20.0] {
                        let k = ((total_features as f64) * pct / 100.0) as usize;
                        let sum: u128 = score.iter().take(k).map(|(_, s, _, _)| *s).sum();
                        let share = 100.0 * sum as f64 / total_score.max(1) as f64;
                        println!("Top {:>4.1}% ({:>5} features): {:>6.2}% of importance",
                            pct, k, share);
                    }

                    println!();
                    println!("=== Top 20 by importance score ===");
                    println!("{:>6}  {:>12}  {:>10}  {:>14}", "idx", "hits", "row_L1", "score(hits×L1)");
                    for &(idx, s, h, l1) in score.iter().take(20) {
                        println!("{:>6}  {:>12}  {:>10}  {:>14}", idx, h, l1, s);
                    }
                }
            }
        }
    }

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(output).unwrap_or_else(|_| panic!("Failed to create {}", output))
    );
    writeln!(out, "idx,hits").expect("write");
    for (idx, h) in &idx_by_hits {
        writeln!(out, "{},{}", idx, h).expect("write");
    }
    println!();
    println!("CSV written to {}", output);
}

fn run_binpack_stats(input: &str) {
    use sfbinpack::CompressedTrainingDataEntryReader;

    let file_size = std::fs::metadata(input)
        .unwrap_or_else(|_| panic!("Failed to stat {}", input))
        .len();
    println!("File: {} ({:.2} GB)", input, file_size as f64 / 1_073_741_824.0);

    let file = std::fs::File::open(input).unwrap_or_else(|_| panic!("Failed to open {}", input));
    let mut reader = CompressedTrainingDataEntryReader::new(file)
        .unwrap_or_else(|_| panic!("Failed to parse binpack {}", input));

    let mut total_positions = 0u64;
    let mut score_sum = 0i64;
    let mut score_abs_sum = 0u64;
    let mut score_max = 0i16;
    let mut score_min = 0i16;
    let mut results = [0u64; 3]; // loss/draw/win
    let mut in_check = 0u64;
    let mut score_buckets = [0u64; 10]; // <100, <200, <500, <1000, <2000, <5000, <10000, <20000, <30000, 30000+
    // Score-anomaly filter retention. Same formula as bullet/coda_v9_768_threats.rs
    // `--max-score-anomaly`: for draws anomaly = |score|; for non-draws anomaly = max(0, -sign(result)*score).
    let anomaly_thresholds: [i32; 9] = [500, 1000, 2000, 5000, 10000, 15000, 20000, 30000, 50000];
    let mut anomaly_dropped = [0u64; 9];
    let mut anomaly_draw_dropped = [0u64; 9];
    let start = std::time::Instant::now();

    while reader.has_next() {
        let entry = reader.next();
        total_positions += 1;

        let score = entry.score;
        score_sum += score as i64;
        score_abs_sum += score.unsigned_abs() as u64;
        if score > score_max { score_max = score; }
        if score < score_min { score_min = score; }

        match entry.result {
            -1 => results[0] += 1,
            0 => results[1] += 1,
            1 => results[2] += 1,
            _ => {}
        }

        if entry.pos.is_checked(entry.pos.side_to_move()) {
            in_check += 1;
        }

        let abs = score.unsigned_abs();
        let bucket = if abs < 100 { 0 } else if abs < 200 { 1 } else if abs < 500 { 2 }
            else if abs < 1000 { 3 } else if abs < 2000 { 4 } else if abs < 5000 { 5 }
            else if abs < 10000 { 6 } else if abs < 20000 { 7 } else if abs < 30000 { 8 }
            else { 9 };
        score_buckets[bucket] += 1;

        // Score-anomaly: for draws = |score|; for non-draws = max(0, -sign(result)*score).
        let anomaly_abs: u32 = match entry.result {
            0 => score.unsigned_abs() as u32,
            r => {
                let sign = (r as i32).signum();
                ((-sign * score as i32).max(0)) as u32
            }
        };
        for (i, &t) in anomaly_thresholds.iter().enumerate() {
            if anomaly_abs > t as u32 {
                anomaly_dropped[i] += 1;
                if entry.result == 0 { anomaly_draw_dropped[i] += 1; }
            }
        }

        if total_positions.is_multiple_of(10_000_000) {
            let elapsed = start.elapsed().as_secs_f64();
            eprint!("\r{:.0}M positions scanned ({:.0}M/s)...",
                total_positions as f64 / 1e6, total_positions as f64 / elapsed / 1e6);
        }
    }
    eprintln!();

    let elapsed = start.elapsed().as_secs_f64();
    let bytes_per_pos = file_size as f64 / total_positions as f64;
    let avg_score = score_sum as f64 / total_positions as f64;
    let avg_abs_score = score_abs_sum as f64 / total_positions as f64;

    println!("\n=== Binpack Statistics ===");
    println!("Positions:      {:>12} ({:.2}M)", total_positions, total_positions as f64 / 1e6);
    println!("Bytes/position: {:>12.1}", bytes_per_pos);
    println!("Scan time:      {:>12.1}s ({:.1}M pos/s)", elapsed, total_positions as f64 / elapsed / 1e6);
    println!();
    println!("Score range:    {} to {}", score_min, score_max);
    println!("Avg score:      {:.1}", avg_score);
    println!("Avg |score|:    {:.1}", avg_abs_score);
    println!("In check:       {:>12} ({:.1}%)", in_check, in_check as f64 / total_positions as f64 * 100.0);
    println!();
    println!("Results: W {:.1}%  D {:.1}%  L {:.1}%",
        results[2] as f64 / total_positions as f64 * 100.0,
        results[1] as f64 / total_positions as f64 * 100.0,
        results[0] as f64 / total_positions as f64 * 100.0);
    println!();
    println!("Score distribution:");
    let labels = ["<100", "<200", "<500", "<1000", "<2000", "<5000", "<10000", "<20000", "<30000", "30000+"];
    for (i, label) in labels.iter().enumerate() {
        let pct = score_buckets[i] as f64 / total_positions as f64 * 100.0;
        let bar = "#".repeat((pct * 2.0) as usize);
        println!("  {:>8}: {:>10} ({:>5.1}%) {}", label, score_buckets[i], pct, bar);
    }
    println!();
    println!("Score-anomaly filter retention (per `--max-score-anomaly`):");
    println!("  threshold      dropped    drop%   retain%    of which draws");
    for (i, &t) in anomaly_thresholds.iter().enumerate() {
        let dropped = anomaly_dropped[i];
        let drop_pct = dropped as f64 / total_positions as f64 * 100.0;
        let retain_pct = 100.0 - drop_pct;
        let draws = anomaly_draw_dropped[i];
        let draw_share = if dropped > 0 { draws as f64 / dropped as f64 * 100.0 } else { 0.0 };
        println!("  {:>9}    {:>10} ({:>5.2}%) ({:>5.2}%)    {:>10} ({:>5.1}%)",
            t, dropped, drop_pct, retain_pct, draws, draw_share);
    }
}

/// Verbatim binpack -> TSV dump (ply, score, move, fen). No filtering, sampling
/// no reordering: row N of the output is position N of the file.
///
/// Deliberately does NOT reuse `run_sample_positions`' loop, which drops
/// |score|>2000 and in-check positions. Those drops make it impossible to
/// compare a filtered binpack against its unfiltered source, because filtering
/// works by writing an out-of-range sentinel score and the sampler discards
/// exactly the rows that carry the sentinel.
fn run_dump_binpack(input: &str, output: &str, count: usize) {
    use sfbinpack::CompressedTrainingDataEntryReader;
    use std::io::Write;

    let file = std::fs::File::open(input)
        .unwrap_or_else(|_| panic!("Failed to open {}", input));
    let mut reader = CompressedTrainingDataEntryReader::new(file)
        .unwrap_or_else(|_| panic!("Failed to parse binpack {}", input));

    let stdout = std::io::stdout();
    let mut out: Box<dyn Write> = if output == "-" {
        Box::new(std::io::BufWriter::new(stdout.lock()))
    } else {
        Box::new(std::io::BufWriter::new(
            std::fs::File::create(output)
                .unwrap_or_else(|_| panic!("Failed to create {}", output)),
        ))
    };

    // fen is written LAST so downstream splits can bound the field count; the
    // move is included because the Stockfish-side filter rules (capture /
    // promotion) key on the stored move, not just the position.
    let mut n = 0usize;
    while reader.has_next() {
        let entry = reader.next();
        let fen = entry.pos.fen().unwrap_or_default();
        if writeln!(out, "{}\t{}\t{}\t{}", entry.ply, entry.score, entry.mv.as_uci(), fen).is_err() {
            break; // downstream closed (e.g. `| head`)
        }
        n += 1;
        if count > 0 && n >= count { break; }
    }
    let _ = out.flush();
}

/// Split a binpack at block boundaries into `parts` roughly equal pieces.
///
/// Walks the `"BINP" + u32le(size)` headers to collect every legal cut point,
/// then picks the ones closest to even byte splits. Each output is a standalone
/// valid binpack; `cat` them back together to recover the original exactly.
fn run_split_binpack(input: &str, output: &str, parts: usize) {
    use std::io::{Read, Seek, SeekFrom, Write};

    if parts < 1 {
        eprintln!("--parts must be >= 1");
        std::process::exit(2);
    }
    let mut f = std::fs::File::open(input)
        .unwrap_or_else(|_| panic!("Failed to open {}", input));
    let total = f.metadata().expect("stat failed").len();

    // Collect block start offsets by walking headers.
    let mut starts: Vec<u64> = Vec::new();
    let mut off: u64 = 0;
    let mut hdr = [0u8; 8];
    loop {
        if off >= total { break; }
        f.seek(SeekFrom::Start(off)).expect("seek failed");
        if f.read_exact(&mut hdr).is_err() { break; }
        if &hdr[0..4] != b"BINP" {
            eprintln!("WARNING: no BINP magic at offset {} — stopping scan (file truncated or not a binpack)", off);
            break;
        }
        let size = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as u64;
        starts.push(off);
        off += 8 + size;
    }
    if starts.is_empty() {
        eprintln!("No BINP blocks found in {}", input);
        std::process::exit(2);
    }
    let scanned_end = off.min(total);
    println!("{} blocks, {} bytes scanned (file {} bytes)", starts.len(), scanned_end, total);
    if scanned_end < total {
        println!("NOTE: {} trailing bytes are not part of a complete block and will be dropped",
                 total - scanned_end);
    }

    let n = parts.min(starts.len());
    if n < parts {
        println!("NOTE: only {} blocks available, producing {} parts", starts.len(), n);
    }
    // Choose cut indices closest to even byte boundaries.
    let mut cuts: Vec<usize> = vec![0];
    for i in 1..n {
        let target = scanned_end * (i as u64) / (n as u64);
        let idx = starts.partition_point(|&o| o < target).max(1).min(starts.len() - 1);
        if idx > *cuts.last().unwrap() { cuts.push(idx); }
    }
    cuts.push(starts.len());

    for i in 0..cuts.len() - 1 {
        let begin = starts[cuts[i]];
        let end = if cuts[i + 1] < starts.len() { starts[cuts[i + 1]] } else { scanned_end };
        let path = format!("{}.{:03}", output, i);
        let mut out = std::io::BufWriter::new(
            std::fs::File::create(&path).unwrap_or_else(|_| panic!("Failed to create {}", path)));
        f.seek(SeekFrom::Start(begin)).expect("seek failed");
        let mut remaining = end - begin;
        let mut buf = vec![0u8; 8 << 20];
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            f.read_exact(&mut buf[..want]).expect("read failed");
            out.write_all(&buf[..want]).expect("write failed");
            remaining -= want as u64;
        }
        out.flush().expect("flush failed");
        println!("  {} : bytes {}..{} ({} blocks, {:.2} GB)",
                 path, begin, end, cuts[i + 1] - cuts[i],
                 (end - begin) as f64 / 1073741824.0);
    }
}

fn run_sample_positions(input: &str, output: &str, n: usize, sample_rate: f64, scores: bool) {
    use sfbinpack::CompressedTrainingDataEntryReader;
    use std::io::Write;

    println!("Sampling ~{} positions from {}", n, input);
    let file = std::fs::File::open(input).unwrap_or_else(|_| panic!("Failed to open {}", input));
    let reader = CompressedTrainingDataEntryReader::new(file)
        .unwrap_or_else(|_| panic!("Failed to parse binpack {}", input));

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(output).unwrap_or_else(|_| panic!("Failed to create {}", output))
    );

    let mut count = 0usize;
    let mut total = 0usize;
    let mut rng_state = 0xDEADBEEF42u64;
    let mut reader = reader;

    while reader.has_next() {
        let entry = reader.next();
        total += 1;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;

        if entry.score.unsigned_abs() > 2000 { continue; }
        if entry.pos.is_checked(entry.pos.side_to_move()) { continue; }

        let keep = if sample_rate > 0.0 {
            let rand_val = (rng_state >> 11) as f64 / ((1u64 << 53) as f64);
            rand_val < sample_rate
        } else {
            true
        };

        if keep {
            let fen = entry.pos.fen().unwrap_or_default();
            if scores {
                writeln!(out, "{}\t{}", fen, entry.score).expect("write failed");
            } else {
                writeln!(out, "{}", fen).expect("write failed");
            }
            count += 1;
            if count >= n { break; }
        }

        if total.is_multiple_of(10_000_000) {
            println!("  scanned {}M positions, sampled {}", total / 1_000_000, count);
        }
    }

    println!("Sampled {} positions from {} scanned → {}", count, total, output);
}

#[allow(clippy::too_many_arguments)]
fn run_eval_fens(input: &str, output: &str, nnue_path: &Option<String>) {
    use std::io::{BufRead as _, Write as _};

    let mut info = search::SearchInfo::new(1);
    info.silent = true;
    if let Some(path) = nnue_path {
        info.load_nnue(path).expect("Failed to load NNUE");
    } else if !info.auto_discover_nnue() {
        eprintln!("Error: No NNUE net. Use -n <path>");
        return;
    }

    let file = std::fs::File::open(input).unwrap_or_else(|_| panic!("Failed to open {}", input));
    let reader = std::io::BufReader::new(file);
    let mut out: Box<dyn std::io::Write> = if output == "-" {
        Box::new(std::io::BufWriter::new(std::io::stdout()))
    } else {
        Box::new(std::io::BufWriter::new(
            std::fs::File::create(output).unwrap_or_else(|_| panic!("Failed to create {}", output)),
        ))
    };

    let (net, acc) = match (&info.nnue_net, &mut info.nnue_acc) {
        (Some(net), Some(acc)) => (net, acc),
        _ => { eprintln!("Error: NNUE not initialised"); return; }
    };

    let mut n = 0usize;
    for line in reader.lines() {
        let line = line.expect("read error");
        let fen = line.split('\t').next().unwrap_or("").trim();
        if fen.is_empty() || fen.starts_with('#') { continue; }
        let board = board::Board::from_fen(fen);

        let mut ts = crate::threat_accum::ThreatStack::new(net.hidden_size);
        ts.active = net.has_threats;
        if ts.active {
            ts.ensure_computed(&net.threat_weights, net.num_threat_features, &board);
        }
        // Each FEN is an unrelated board: full recompute, same as eval-dist.
        acc.force_recompute(net, &board);
        let score = eval::evaluate_nnue(&board, net, acc, &ts);
        writeln!(out, "{}\t{}", fen, score).unwrap();
        n += 1;
    }
    out.flush().unwrap();
    eprintln!("eval-fens: {} positions evaluated", n);
}

/// Classify a position's material signature from the FEN piece-placement
/// field, splitting bishop endings into opposite- vs same-coloured. Focused on
/// the drawish minor-piece endgame classes (OCB+pawns etc.).
fn classify_material(fen: &str) -> &'static str {
    let placement = fen.split(' ').next().unwrap_or("");
    let mut cnt = [0i32; 128];
    let (mut rank, mut file) = (7i32, 0i32);
    let (mut wb_light, mut wb_dark, mut bb_light, mut bb_dark) = (0i32, 0i32, 0i32, 0i32);
    for ch in placement.chars() {
        match ch {
            '/' => { rank -= 1; file = 0; }
            '1'..='8' => { file += ch as i32 - '0' as i32; }
            _ => {
                let sq_light = (rank + file) % 2 == 1; // a1 (0,0)=dark
                match ch {
                    'B' => { if sq_light { wb_light += 1 } else { wb_dark += 1 } }
                    'b' => { if sq_light { bb_light += 1 } else { bb_dark += 1 } }
                    _ => {}
                }
                if (ch as usize) < 128 { cnt[ch as usize] += 1; }
                file += 1;
            }
        }
    }
    let g = |c: char| cnt[c as usize];
    let (wp, bp) = (g('P'), g('p'));
    let (wn, wb, wr, wq) = (g('N'), g('B'), g('R'), g('Q'));
    let (bn, bb, br, bq) = (g('n'), g('b'), g('r'), g('q'));
    let (w_minor, b_minor) = (wn + wb, bn + bb);
    if wq > 0 || bq > 0 { return "Q-endgame"; }
    if wr > 0 || br > 0 {
        return if w_minor == 0 && b_minor == 0 { "rook-ending" } else { "rook+minor" };
    }
    if w_minor == 0 && b_minor == 0 {
        return if wp > 0 || bp > 0 { "pawns-only" } else { "KvK" };
    }
    // No rooks/queens, at least one minor -> minor endgame family.
    if wb == 1 && bb == 1 && wn == 0 && bn == 0 {
        let w_light = wb_light == 1;
        let b_light = bb_light == 1;
        let _ = (wb_dark, bb_dark);
        return if w_light != b_light { "OCB+pawns" } else { "SCB+pawns" };
    }
    if w_minor + b_minor == 1 { return "single-minor+pawns"; }
    if wb == 0 && bb == 0 { return "knight(s)+pawns"; }
    "minor-mix+pawns"
}

/// Read-only material-class histogram + per-class label distribution over a
/// binpack. See the BinpackMaterial command doc.
fn run_binpack_material(input: &str, max: usize) {
    use sfbinpack::CompressedTrainingDataEntryReader;
    use std::collections::HashMap;

    let file = std::fs::File::open(input).unwrap_or_else(|_| panic!("Failed to open {}", input));
    let mut reader = CompressedTrainingDataEntryReader::new(file)
        .unwrap_or_else(|_| panic!("Failed to parse binpack {}", input));

    let mut count: HashMap<&'static str, u64> = HashMap::new();
    let mut abs_sum: HashMap<&'static str, i64> = HashMap::new();
    let mut dec150: HashMap<&'static str, u64> = HashMap::new();
    let mut dec300: HashMap<&'static str, u64> = HashMap::new();
    let mut total: u64 = 0;
    let mut kept: u64 = 0;

    while reader.has_next() && (max == 0 || kept < max as u64) {
        let entry = reader.next();
        total += 1;
        if entry.pos.is_checked(entry.pos.side_to_move()) { continue; }
        if entry.score.unsigned_abs() > 10000 { continue; }
        let fen = match entry.pos.fen() { Ok(f) => f, Err(_) => continue };
        kept += 1;
        let cls = classify_material(&fen);
        let s = (entry.score as i64).abs();
        *count.entry(cls).or_insert(0) += 1;
        *abs_sum.entry(cls).or_insert(0) += s;
        if s > 150 { *dec150.entry(cls).or_insert(0) += 1; }
        if s > 300 { *dec300.entry(cls).or_insert(0) += 1; }
    }

    let mut classes: Vec<(&'static str, u64)> = count.iter().map(|(k, v)| (*k, *v)).collect();
    classes.sort_by(|a, b| b.1.cmp(&a.1));
    println!("binpack: {}", input);
    println!("scanned {} kept {} (skipped in-check/extreme)", total, kept);
    println!("{:<20} {:>12} {:>7} {:>10} {:>9} {:>9}", "class", "count", "pct", "mean|scr|", "%>150cp", "%>300cp");
    for (cls, c) in &classes {
        let mean = *abs_sum.get(cls).unwrap_or(&0) as f64 / *c as f64;
        let d150 = 100.0 * *dec150.get(cls).unwrap_or(&0) as f64 / *c as f64;
        let d300 = 100.0 * *dec300.get(cls).unwrap_or(&0) as f64 / *c as f64;
        let pct = 100.0 * *c as f64 / kept.max(1) as f64;
        println!("{:<20} {:>12} {:>6.2}% {:>10.0} {:>8.1}% {:>8.1}%", cls, c, pct, mean, d150, d300);
    }
}

fn run_eval_dist(input: &str, n: usize, nnue_path: &Option<String>, csv: &Option<String>,
                 quiet_only: bool, shard: &Option<String>, min_error: i32, max_abs_lc0: i32) {
    use sfbinpack::CompressedTrainingDataEntryReader;
    use std::io::Write as _;

    // Parse `--shard k/N`. Present => stream-scan the whole binpack (ignore
    // --count) and skip in-memory stats, so N copies can cover the file across
    // cores without OOMing on billions of rows.
    let shard_kn: Option<(usize, usize)> = shard.as_ref().map(|s| {
        let (k, nn) = s.split_once('/').expect("--shard must be k/N");
        let k: usize = k.trim().parse().expect("--shard k must be an integer");
        let nn: usize = nn.trim().parse().expect("--shard N must be an integer");
        assert!(nn > 0 && k < nn, "--shard requires 0 <= k < N");
        (k, nn)
    });
    let scan_all = shard_kn.is_some();
    if scan_all && csv.is_none() {
        eprintln!("Error: --shard requires --csv (streaming harvest mode)");
        return;
    }

    // Load NNUE
    let mut info = search::SearchInfo::new(1);
    info.silent = true;
    if let Some(path) = nnue_path {
        info.load_nnue(path).expect("Failed to load NNUE");
    } else if !info.auto_discover_nnue() {
        eprintln!("Error: No NNUE net. Use -n <path>");
        return;
    }

    // Optional CSV: `fen,white_result,coda_eval_white_cp` per kept position,
    // for a cross-engine eval-quality benchmark.
    let mut csv_w = csv.as_ref().map(|p| {
        let f = std::fs::File::create(p).unwrap_or_else(|_| panic!("Failed to create CSV {}", p));
        let mut w = std::io::BufWriter::new(f);
        // lc0_score = the binpack's LC0 WDL-calibrated eval (white POV) — the
        // strong-oracle ground truth (game RESULT is too drawish on T80
        // self-play to discriminate static evals; corr ≈ 0).
        // game_id: globally-consistent game index (incremented on every
        // ply-continuation break). Lets the harvest dedupe to <=1 position per
        // game — same-game positions share the eval blind spot, so keeping
        // several from one game would correlate the test set and inflate signal.
        writeln!(w, "fen,white_result,coda_eval_white_cp,lc0_score_white_cp,game_id").unwrap();
        w
    });

    println!("Evaluating {} positions from {}", n, input);
    let file = std::fs::File::open(input).unwrap_or_else(|_| panic!("Failed to open {}", input));
    let reader = CompressedTrainingDataEntryReader::new(file)
        .unwrap_or_else(|_| panic!("Failed to parse binpack {}", input));

    let mut scores: Vec<i32> = Vec::with_capacity(if scan_all { 0 } else { n });
    let mut results: Vec<f64> = Vec::with_capacity(if scan_all { 0 } else { n }); // 1.0=white win, 0.5=draw, 0.0=black win
    let mut total = 0usize;
    let mut kept = 0usize;
    let mut reader = reader;

    // Globally-consistent game id, derived from ply-continuation breaks (the
    // binpack's own game-boundary signal: a new game resets ply, breaking
    // `prev_ply + 1 == ply`). Tracked for EVERY entry read — before the shard
    // skip — so all N shards compute identical ids and the union dedupes by game.
    let mut game_id: u64 = 0;
    let mut prev_ply: i64 = -2;

    while reader.has_next() && (scan_all || scores.len() < n) {
        let entry = reader.next();
        total += 1;
        if entry.ply as i64 != prev_ply + 1 { game_id += 1; }
        prev_ply = entry.ply as i64;

        // Modulo shard: copy k of N processes only entries where
        // (global_index) % N == k. Union of all N shards = the full file.
        // The reader is sequential (chain-compressed, no seek), so every
        // copy still decompresses everything, but the 9GB binpack caches
        // in RAM once and decompression is cheap vs the NNUE eval below.
        if let Some((k, nn)) = shard_kn {
            if (total - 1) % nn != k { continue; }
        }

        // Skip checks and extreme scores (same filter as training)
        if entry.pos.is_checked(entry.pos.side_to_move()) { continue; }
        if entry.score.unsigned_abs() > 10000 { continue; }

        // Band pre-filter (before the expensive NNUE eval): only harvest
        // positions the oracle rates as roughly balanced. Cheap reject.
        if max_abs_lc0 > 0 && (entry.score as i32).abs() > max_abs_lc0 { continue; }

        // Quiet filter (eval-quality export): skip positions whose played
        // move is a capture or promotion — matches how NNUE is consumed at
        // QS leaves. (The .min-v2.v6 binpacks are already quiet-filtered at
        // creation; this is belt-and-suspenders for raw binpacks.)
        if quiet_only {
            let is_capture = entry.pos.piece_at(entry.mv.to()) != sfbinpack::chess::piece::Piece::NONE
                || entry.mv.mtype() == sfbinpack::chess::r#move::MoveType::EnPassant;
            let is_promo = entry.mv.mtype() == sfbinpack::chess::r#move::MoveType::Promotion;
            if is_capture || is_promo { continue; }
        }

        // Convert to our Board and evaluate
        let fen = match entry.pos.fen() {
            Ok(f) => f,
            Err(_) => continue,
        };
        let board = board::Board::from_fen(&fen);

        let score = if let (Some(net), Some(acc)) = (&info.nnue_net, &mut info.nnue_acc) {
            // Build a real ThreatStack for v9 nets — without this,
            // the threat half of the eval is silently zeroed
            // (audit C2026-04-25-3).
            let mut ts = crate::threat_accum::ThreatStack::new(net.hidden_size);
            ts.active = net.has_threats;
            if ts.active {
                ts.ensure_computed(&net.threat_weights, net.num_threat_features, &board);
            }
            // CRITICAL: the accumulator is reused across positions; each FEN
            // is an unrelated board, so the cached acc is stale. Without a
            // full recompute, evaluate_nnue runs on garbage state (off by
            // 100cp+). The UCI path is refreshed by the `position` command;
            // this batch path must refresh explicitly.
            acc.force_recompute(net, &board);
            eval::evaluate_nnue(&board, net, acc, &ts)
        } else {
            continue;
        };

        // Min-error post-filter (after the NNUE eval): only harvest
        // positions where Coda disagrees with the LC0 oracle by at least
        // `min_error` cp (both STM-POV). This is the eval-blindspot gate.
        if min_error > 0 && (score - entry.score as i32).abs() < min_error { continue; }

        // Game result from STM perspective: 1.0=stm wins, 0.0=stm loses
        // entry.result: 1=white win, 0=draw, -1=black win
        let stm_is_white = entry.pos.side_to_move() == sfbinpack::chess::color::Color::White;
        let result_f = match entry.result {
            1 => if stm_is_white { 1.0 } else { 0.0 },
            -1 => if stm_is_white { 0.0 } else { 1.0 },
            _ => 0.5, // draw
        };

        // In scan/harvest mode we stream straight to CSV and never
        // accumulate (the file has billions of rows — Vecs would OOM).
        if !scan_all {
            scores.push(score);
            results.push(result_f);
        }
        kept += 1;

        if let Some(w) = csv_w.as_mut() {
            // White-POV result and eval, to match the white-POV `eval`
            // output of all engines in the benchmark driver.
            let white_result = match entry.result { 1 => 1.0, -1 => 0.0, _ => 0.5 };
            let white_cp = if stm_is_white { score } else { -score };
            let lc0_white = if stm_is_white { entry.score } else { -entry.score };
            writeln!(w, "{},{:.1},{},{},{}", fen, white_result, white_cp, lc0_white, game_id).unwrap();
        }

        if kept.is_multiple_of(100_000) {
            if scan_all {
                eprint!("\r  {} harvested ({} scanned)", kept, total);
            } else {
                eprint!("\r  {} / {} evaluated ({} scanned)", kept, n, total);
            }
        }
    }
    eprintln!();

    // Scan/harvest mode: CSV already streamed to disk; no in-memory stats.
    if scan_all {
        if let Some(mut w) = csv_w.take() { w.flush().ok(); }
        println!("Harvested {} positions from {} scanned → {}",
                 kept, total, csv.as_ref().unwrap());
        return;
    }

    let count = scores.len();
    if count == 0 {
        println!("No positions evaluated");
        return;
    }

    // Debug: result distribution
    let wins = results.iter().filter(|&&r| r > 0.7).count();
    let draws = results.iter().filter(|&&r| r > 0.3 && r < 0.7).count();
    let losses = results.iter().filter(|&&r| r < 0.3).count();
    println!("  Results: {} wins, {} draws, {} losses", wins, draws, losses);

    // Statistics
    let sum: f64 = scores.iter().map(|&s| s as f64).sum();
    let sum_sq: f64 = scores.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let abs_sum: f64 = scores.iter().map(|&s| (s as f64).abs()).sum();
    let mean = sum / count as f64;
    let rms = (sum_sq / count as f64).sqrt();
    let abs_mean = abs_sum / count as f64;

    let mut sorted = scores.clone();
    sorted.sort();

    println!("=== Eval Distribution ({} positions, {} scanned) ===", count, total);
    println!("  Mean:     {:+.1}", mean);
    println!("  Abs mean: {:.1}", abs_mean);
    println!("  RMS:      {:.1}", rms);
    println!("  Ratio to baseline (580): {:.2}x", rms / 580.0);
    println!();
    println!("  Percentiles:");
    println!("    p1:  {:+}", sorted[count / 100]);
    println!("    p5:  {:+}", sorted[count * 5 / 100]);
    println!("    p10: {:+}", sorted[count / 10]);
    println!("    p25: {:+}", sorted[count / 4]);
    println!("    p50: {:+}", sorted[count / 2]);
    println!("    p75: {:+}", sorted[count * 3 / 4]);
    println!("    p90: {:+}", sorted[count * 9 / 10]);
    println!("    p95: {:+}", sorted[count * 95 / 100]);
    println!("    p99: {:+}", sorted[count * 99 / 100]);
    println!();

    // === WDL Logistic Fit (SF approach) ===
    // Fit: win_rate(eval) = 1 / (1 + exp(-eval / K))
    // Find K (NormalizeToPawnValue) where EVAL_SCALE should be set so that
    // 100cp of NNUE output ≈ 50% win probability.
    //
    // Uses our static NNUE eval paired with game outcomes from the binpack.
    // NOTE: For best results, use data generated by the SAME net being evaluated,
    // or T80 data with a net trained on T80. Mismatched eval/results gives poor fit.
    println!("=== WDL Logistic Fit (NNUE eval vs game outcome) ===");

    let mut best_k = 200.0f64;
    let mut best_err = f64::MAX;

    // Binary search over K from 50 to 2000
    for k_int in 50..=2000 {
        let k = k_int as f64;
        let mut err_sum = 0.0f64;
        for i in 0..count {
            let eval = scores[i] as f64;
            let predicted = 1.0 / (1.0 + (-eval / k).exp());
            let actual = results[i];
            let diff = predicted - actual;
            err_sum += diff * diff;
        }
        if err_sum < best_err {
            best_err = err_sum;
            best_k = k;
        }
    }

    let mse = best_err / count as f64;
    println!("  Best K (NormalizeToPawnValue): {:.0}", best_k);
    println!("  MSE: {:.6}", mse);
    println!("  → EVAL_SCALE should be {:.0} so that 100cp ≈ 50% win", best_k);
    println!();

    // Show calibration: predicted win% at key eval values
    println!("  Calibration (eval → predicted win%):");
    for &ev in &[25, 50, 100, 150, 200, 300, 500] {
        let p = 1.0 / (1.0 + (-(ev as f64) / best_k).exp());
        println!("    {:+4}cp → {:.1}% win", ev, p * 100.0);
    }
    println!();

    // Show actual win rates in eval buckets for validation
    println!("  Actual win rates by eval bucket:");
    let buckets: Vec<(i32, i32)> = vec![
        (-500, -200), (-200, -100), (-100, -50), (-50, -25),
        (-25, 0), (0, 25), (25, 50), (50, 100),
        (100, 200), (200, 500),
    ];
    for (lo, hi) in &buckets {
        let mut wins = 0.0f64;
        let mut n_bucket = 0usize;
        for i in 0..count {
            if scores[i] >= *lo && scores[i] < *hi {
                wins += results[i];
                n_bucket += 1;
            }
        }
        if n_bucket > 100 {
            println!("    [{:+4}, {:+4}): {:.1}% win (n={})",
                lo, hi, wins / n_bucket as f64 * 100.0, n_bucket);
        }
    }
}

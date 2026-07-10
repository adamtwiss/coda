//! coda-binpack-rescale — multiply the non-sentinel scores of a binpack by a
//! factor, preserving chains. Chain-safe because we pass every entry through in
//! the exact same pos/mv/order — the writer re-derives identical continuation
//! chains from position continuity; only the score field changes.
//!
//! Use: put SF-datagen labels (~0.23x the LC0 T80 scale) onto the T80 scale
//! before mixing into a single Bullet run. Sentinel positions (|score|>10000,
//! the Coda-to-move half dropped by Bullet's filter) are left untouched. Scaled
//! scores clamp to ±9999 so a real SF label never crosses into the drop band.
//!
//!   coda-binpack-rescale -i in.binpack -o out.binpack --scale 4.0

use std::fs::File;
use std::io::BufWriter;

use clap::Parser;
use sfbinpack::{CompressedTrainingDataEntryReader, CompressedTrainingDataEntryWriter};

#[derive(Parser)]
#[command(about = "Rescale non-sentinel binpack scores, chain-preserving")]
struct Args {
    #[arg(short = 'i', long)]
    input: String,
    #[arg(short = 'o', long)]
    output: String,
    /// Multiply scores with |score| <= sentinel-above by this factor.
    #[arg(long)]
    scale: f64,
    /// Scores with |score| > this are sentinels (Coda-to-move) — left untouched.
    #[arg(long, default_value_t = 10000)]
    sentinel_above: i32,
}

fn main() {
    let a = Args::parse();
    let inf = File::open(&a.input).expect("open input");
    let mut reader = CompressedTrainingDataEntryReader::new(inf).expect("binpack reader");
    let outf = File::create(&a.output).expect("create output");
    let mut writer = CompressedTrainingDataEntryWriter::new(BufWriter::with_capacity(1 << 20, outf))
        .expect("binpack writer");

    let (mut n, mut scaled) = (0u64, 0u64);
    while reader.has_next() {
        let mut e = reader.next();
        if (e.score as i32).abs() <= a.sentinel_above {
            e.score = ((e.score as f64) * a.scale).round().clamp(-9999.0, 9999.0) as i16;
            scaled += 1;
        }
        writer.write_entry(&e).expect("write entry");
        n += 1;
        if n % 20_000_000 == 0 {
            eprintln!("  {} entries ({} scaled)", n, scaled);
        }
    }
    eprintln!("done: wrote {} entries ({} scaled, {} sentinel-untouched)", n, scaled, n - scaled);
}

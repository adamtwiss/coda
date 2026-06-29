#!/usr/bin/env python3
"""Per-shard stage-2 filter: join Coda CSV + SF static eval, calibrate both to
the LC0 scale, keep coda-specific misrates (calibrated |coda-lc0| >= THRESH and
coda worse than SF by >= MARGIN). Emit fen<TAB>lc0_stm_cp (the corrective label)."""
import sys

THRESH = 150   # calibrated |coda-lc0| floor (the "150")
MARGIN = 80    # how much worse than SF in lc0 scale (the "80")

ca, cb, sa, sb = (float(x) for x in open("/tmp/blindspot/calib.txt").read().split())
k = sys.argv[1]
csv_path = f"/tmp/blindspot/shard_{k}.csv"
sf_path  = f"/tmp/blindspot/sf_out_{k}.txt"
out_path = f"/tmp/blindspot/cands_{k}.tsv"

kept = tot = mism = 0
with open(csv_path) as cf, open(sf_path) as sf, open(out_path, "w") as out:
    next(cf)  # header: fen,white_result,coda_eval_white_cp,lc0_score_white_cp
    for cline, sline in zip(cf, sf):
        p = cline.rstrip("\n").split(",")
        if len(p) < 4:
            continue
        fen = p[0]
        try:
            coda = int(p[2]); lc0 = int(p[3])
        except ValueError:
            continue
        sid, scp = sline.rstrip("\n").split("\t")
        sf_cp = int(scp)
        # SF skips in-check (none here, stage-1 filtered) -> ids stay in lockstep,
        # but guard anyway: if ids drift, bail loudly.
        if int(sid) != tot:
            mism += 1
        tot += 1
        cerr = abs((ca * coda + cb) - lc0)
        serr = abs((sa * sf_cp + sb) - lc0)
        if cerr >= THRESH and (cerr - serr) >= MARGIN:
            stm_white = fen.split()[1] == "w"
            stm_cp = lc0 if stm_white else -lc0
            out.write(f"{fen}\t{stm_cp}\n")
            kept += 1
print(f"shard {k}: kept {kept}/{tot}" + (f"  ID-MISMATCH={mism}" if mism else ""))

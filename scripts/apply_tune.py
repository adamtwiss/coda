#!/usr/bin/env python3
"""Apply SPSA tune outputs to the tunables! macro in src/search.rs.

Canonical, self-validating replacement for ad-hoc /tmp apply scripts (which
drifted out of sync and silently dropped params — e.g. a hardcoded NMP-cluster
skip-list once detuned the trunk without any error). USE THIS, committed, tool.

Safety guarantees (any violation => non-zero exit, no partial write committed):
  * Every input parameter MUST exist in the live tunables! macro. An input
    param not found in the macro (typo, rename, removed tunable) is a HARD
    ERROR, never a silent skip.
  * Each parameter must match EXACTLY ONCE in src/search.rs (no ambiguous /
    duplicate matches).
  * The set of valid names is read from `./coda tune-spec` (the live macro),
    so the tool can never drift from the source of truth.
  * NO hardcoded include/exclude lists. A focused (subset) tune is fine — the
    only rule is input ⊆ macro.

Input format: the `/api/spsa/<id>/outputs/` API and `ob_tune_status.py <id>
--outputs` emit `NAME, value` per line. The full SPSA spec `NAME, int, value,
min, max, c_end, r_end` (>=7 fields) is also accepted (value = field 3).

Usage:
  python3 scripts/apply_tune.py <tune_outputs_file> [--src src/search.rs]
                                [--coda ./coda] [--dry-run]
"""
import argparse, re, subprocess, sys


def load_valid_names(coda):
    """Param names from the live macro via `coda tune-spec` (NAME, int, ...)."""
    try:
        out = subprocess.run([coda, "tune-spec"], capture_output=True, text=True,
                             timeout=60).stdout
    except Exception as e:
        sys.exit(f"ERROR: could not run `{coda} tune-spec`: {e}\n"
                 f"Build first (`make`) or pass --coda <path>.")
    names = set()
    for line in out.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]+),", line.strip())
        if m:
            names.add(m.group(1))
    if not names:
        sys.exit(f"ERROR: `{coda} tune-spec` produced no params — wrong binary?")
    return names


def parse_outputs(path):
    """Parse tune output file -> {name: value}. Handles 2-field and SPSA spec."""
    vals = {}
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("==="):
            continue
        parts = [p.strip() for p in line.split(",")]
        m = re.match(r"^[A-Z][A-Z0-9_]+$", parts[0])
        if not m:
            continue  # header / prose line
        name = parts[0]
        if len(parts) == 2:
            value = parts[1]
        elif len(parts) >= 7:          # NAME, int, value, min, max, c_end, r_end
            value = parts[2]
        else:
            sys.exit(f"ERROR: cannot parse line (unexpected field count): {line!r}")
        if not re.match(r"^-?\d+$", value):
            sys.exit(f"ERROR: non-integer value for {name}: {value!r}")
        if name in vals and vals[name] != value:
            sys.exit(f"ERROR: {name} appears twice with different values "
                     f"({vals[name]} vs {value})")
        vals[name] = value
    if not vals:
        sys.exit(f"ERROR: no parameters parsed from {path}")
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tune_file", help="tune outputs (NAME, value per line)")
    ap.add_argument("--src", default="src/search.rs")
    ap.add_argument("--coda", default="./coda", help="coda binary for tune-spec")
    ap.add_argument("--dry-run", action="store_true", help="report, don't write")
    args = ap.parse_args()

    valid = load_valid_names(args.coda)
    vals = parse_outputs(args.tune_file)

    # Gate 1: every input param must exist in the live macro.
    unknown = sorted(n for n in vals if n not in valid)
    if unknown:
        sys.exit("ERROR: these tune params are NOT in the live tunables! macro "
                 "(renamed/removed/typo?) — refusing to apply:\n  " +
                 "\n  ".join(unknown))

    src = open(args.src).read()
    changed, nochange, bad = [], [], []
    for name, val in vals.items():
        pat = re.compile(r"(\(\s*" + re.escape(name) + r"\s*,\s*)(-?\d+)(\s*,)")
        matches = pat.findall(src)
        if len(matches) != 1:
            bad.append(f"{name}: matched {len(matches)}x (expected 1)")
            continue
        old = matches[0][1]
        if old == val:
            nochange.append(name)
        else:
            src = pat.sub(lambda mm: mm.group(1) + val + mm.group(3), src, count=1)
            changed.append(f"{name}: {old} -> {val}")

    # Gate 2: every param must have matched exactly once.
    if bad:
        sys.exit("ERROR: ambiguous/missing matches in {} — refusing to write:\n  "
                 .format(args.src) + "\n  ".join(bad))

    print(f"OK: {len(vals)} params validated against live macro "
          f"({len(changed)} changed, {len(nochange)} already at value)")
    for c in changed:
        print("  ", c)
    if args.dry_run:
        print("(dry-run: not written)")
    else:
        open(args.src, "w").write(src)
        print(f"Wrote {args.src}")


if __name__ == "__main__":
    main()

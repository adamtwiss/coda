#!/usr/bin/env python3
"""Generate THIRD_PARTY_LICENSES.md from a cargo-bundle-licenses JSON dump.

Usage:
    cargo bundle-licenses --format json --output licenses.json
    python3 .github/gen-third-party-licenses.py licenses.json THIRD_PARTY_LICENSES.md

Run at release time (see .github/workflows/release.yml) and whenever the
dependency tree changes. Coda itself is GPL-3.0-or-later (see LICENSE); this
file reproduces the notices of the third-party crates Coda links.
"""
import json
import collections
import sys

def main(in_path: str, out_path: str) -> None:
    data = json.load(open(in_path))
    libs = sorted(data["third_party_libraries"], key=lambda x: x["package_name"].lower())

    out = []
    out.append("# Third-Party Licenses\n")
    out.append(
        "Coda is distributed under the **GNU General Public License v3.0 or later** "
        "(see [`LICENSE`](LICENSE)). It links a number of third-party Rust libraries; "
        "this file reproduces their license and copyright notices, as those licenses "
        "require.\n")
    out.append("## Summary\n")
    out.append(
        "- The **chess libraries** Coda links are **GPL-3.0-or-later** — `shakmaty`, "
        "`shakmaty-syzygy` and `pgn-reader` (Niklas Fiekas), and `sfbinpack`. Because "
        "Coda links these, the combined work is conveyed under the GPL, which matches "
        "Coda's own license.\n"
        "- The remaining crates are under **permissive** licenses (MIT, Apache-2.0, "
        "Unlicense, Unicode-3.0), all GPL-compatible. Their notices are reproduced below.\n"
        "- No dependency is under AGPL or any license incompatible with GPLv3 distribution.\n")
    out.append(
        "Corresponding source for Coda and for its GPL-licensed dependencies is available "
        "from the project repository. This file is regenerated from the dependency tree "
        "at release time via `cargo bundle-licenses`.\n")

    out.append("## Dependencies\n")
    out.append("| Crate | Version | License |")
    out.append("|---|---|---|")
    for l in libs:
        out.append(f"| `{l['package_name']}` | {l['package_version']} | {l['license']} |")
    out.append("")

    out.append("## License texts\n")
    by_text = collections.OrderedDict()
    for l in libs:
        for lic in l["licenses"]:
            t = (lic.get("text") or "").strip()
            name = lic.get("license", "")
            key = t if t else f"__NO_TEXT__:{name}"
            entry = by_text.setdefault(key, {"ids": set(), "crates": set(), "text": t})
            entry["ids"].add(name)
            entry["crates"].add(l["package_name"])

    for info in by_text.values():
        ids = ", ".join(sorted(info["ids"]))
        crates = ", ".join(f"`{c}`" for c in sorted(info["crates"]))
        out.append(f"### {ids}\n")
        out.append(f"*Used by: {crates}*\n")
        if info["text"]:
            out.append("```")
            out.append(info["text"])
            out.append("```\n")
        else:
            out.append("_(No standalone license file shipped in the crate; see the crate's repository.)_\n")

    open(out_path, "w").write("\n".join(out))
    print(f"wrote {out_path}: {len(libs)} crates, {len(by_text)} unique license texts")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: gen-third-party-licenses.py <licenses.json> <out.md>")
    main(sys.argv[1], sys.argv[2])

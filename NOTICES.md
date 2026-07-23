# Notices

Attribution notices for third-party material Coda draws on, beyond the *linked*
dependencies (whose notices are in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md),
generated from the dependency tree at release time). Coda's implementation is its own
code; the notices here are reproduced as attribution for ideas, techniques, and — in
one case — tuning constants studied from other open engines.

## Studied engines

**Viridithas** — MIT-licensed versions (through v20). Coda's time-management tuning
constants were initialised from its published values (since re-tuned on Coda's own
search via SPSA). MIT notice reproduced in full:

```
Copyright (c) 2022-2025 Cosmo Bobak

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the
Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

**Hobbes**, **Midnight** — MIT-licensed; techniques studied and credited, no code or
constants used. **Starzix** — WTFPL; likewise ideas only. GPL-3.0 engines (Stockfish
and the others listed in the README) are technique-level references, credited in-source.

## Training-data licenses

- **LC0 self-play data** — ODbL-1.0 (database) / DBCL-1.0 (contents), LCZero team.
  The ODbL attribution notice is reproduced in the README Credits section.
- **Some CC0 1.0 (public-domain) data** — Joost VandeVondele, published on Kaggle.

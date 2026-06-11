"""v18 Pine default-neutrality check (plan/v18-repair-spec.md Stage C, C3).

Text-level structural diff between pine/speculatores_v17_5_signalcard.pine
and pine/speculatores_v18_signalcard.pine. The v18 Pine cannot be TV-tested
until the user applies it on a chart; this is the local proof that the ONLY
source differences fall into the five spec-sanctioned categories:

  (a) title / banner (indicator() title string + the v18 comment banner)
  (b) the gjr clip removal line(s)         -> P2.2
  (c) the 4 new input lines (+ group/comment lines around them)
  (d) the momentum-vote comparison lines   -> P2.3
  (e) the max_votes lines                  -> P2.4

Anything else in the diff -> FAIL (exit 1). Prints the full unified diff for
controller review.
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "pine" / "speculatores_v17_5_signalcard.pine"
NEW = REPO / "pine" / "speculatores_v18_signalcard.pine"


def classify(tag: str, line: str) -> str | None:
    """Return the category for a changed line, or None if unsanctioned."""
    s = line.strip()
    # (a) title / banner -------------------------------------------------
    if tag == "-" and s.startswith('indicator("Speculatores V17.5'):
        return "a:title"
    if tag == "+" and s.startswith('indicator("Speculatores V18'):
        return "a:title"
    if tag == "+" and s.startswith("//") and (
        "SPECULATORES V18" in s
        or "requirable drift" in s
        or "Python was fixed to match Pine" in s
        or "bit-neutral" in s
        or "behaves IDENTICALLY to speculatores_v17_5_signalcard" in s
        or s == "// " + "=" * 76
    ):
        return "a:banner"
    # (b) gjr clip removal ------------------------------------------------
    if tag == "-" and s == "float gjr_asym_norm = clamp_unit((gjr_asym_ratio - 1.0) / 0.1)":
        return "b:gjr-clip-removed"
    if tag == "+" and s == "float gjr_asym_norm = (gjr_asym_ratio - 1.0) / 0.1":
        return "b:gjr-unclipped"
    if tag == "+" and s.startswith("// v18 P2.2"):
        return "b:gjr-comment"
    # (c) the 4 new inputs (+ their group/comment lines) -------------------
    if tag == "+" and (
        s == 'string GRP_V18 = "Votes (v18)"'
        or s.startswith("// v18 per-side vote inputs")
        or (
            s.startswith((
                "float momentum_diverge_thresh_high = input.float(0.0",
                "float momentum_diverge_thresh_low = input.float(0.0",
                "bool count_drift_vote_high = input.bool(false",
                "bool count_drift_vote_low = input.bool(false",
            ))
            and "group=GRP_V18" in s
        )
        or s == ""  # blank separator inside the inserted input block
    ):
        return "c:new-inputs"
    # (d) momentum-vote comparison ----------------------------------------
    if tag == "-" and s in (
        "bool _mom_div_neg_high = mom_diverge_high < 0",
        "bool _mom_div_neg_low = mom_diverge_low < 0",
    ):
        return "d:mom-vote-old"
    if tag == "+" and (
        s.startswith("bool _mom_div_neg_high = mom_diverge_high < -momentum_diverge_thresh_high")
        or s.startswith("bool _mom_div_neg_low = mom_diverge_low < -momentum_diverge_thresh_low")
    ):
        return "d:mom-vote-new"
    # (e) max_votes -------------------------------------------------------
    if s.startswith(("int max_votes_high =", "int max_votes_low =")):
        if tag == "-" and "count_drift_vote" not in s:
            return "e:max-votes-old"
        if tag == "+" and (
            s.endswith("(count_drift_vote_high ? 1 : 0)  // v18 P2.4")
            or s.endswith("(count_drift_vote_low ? 1 : 0)  // v18 P2.4")
        ):
            return "e:max-votes-new"
    return None


def main() -> int:
    old = OLD.read_text(encoding="utf-8").splitlines()
    new = NEW.read_text(encoding="utf-8").splitlines()

    diff = list(difflib.unified_diff(old, new, fromfile=OLD.name,
                                     tofile=NEW.name, lineterm="", n=2))
    print("=" * 78)
    print("UNIFIED DIFF (v17.5 -> v18)")
    print("=" * 78)
    for line in diff:
        print(line)
    print("=" * 78)

    counts: dict[str, int] = {}
    violations: list[str] = []
    for line in diff:
        if line.startswith(("---", "+++", "@@")):
            continue
        if not line.startswith(("+", "-")):
            continue
        cat = classify(line[0], line[1:])
        if cat is None:
            violations.append(line)
        else:
            counts[cat] = counts.get(cat, 0) + 1

    print("\nCATEGORY COUNTS:")
    for k in sorted(counts):
        print(f"  {k:24s} {counts[k]}")

    # Structural expectations: every sanctioned change must actually appear.
    expect = {
        "a:title": 2,            # -old +new
        "b:gjr-clip-removed": 1,
        "b:gjr-unclipped": 1,
        "d:mom-vote-old": 2,
        "d:mom-vote-new": 2,
        "e:max-votes-old": 2,
        "e:max-votes-new": 2,
    }
    ok = True
    for k, n in expect.items():
        if counts.get(k, 0) != n:
            print(f"MISSING/WRONG: expected {n}x {k}, got {counts.get(k, 0)}")
            ok = False
    if counts.get("c:new-inputs", 0) < 5:  # group + 4 inputs at minimum
        print(f"MISSING: expected >=5 c:new-inputs lines, got "
              f"{counts.get('c:new-inputs', 0)}")
        ok = False

    if violations:
        print("\nUNSANCTIONED DIFF LINES:")
        for v in violations:
            print(" ", v)
        ok = False

    print("\nRESULT:", "PASS — only sanctioned v18 changes present" if ok
          else "FAIL — diff contains unsanctioned changes")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

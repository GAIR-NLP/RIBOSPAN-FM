#!/usr/bin/env python3
"""Unit tests for the dot-bracket parser (Task 2)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_rna_contact_datasets import parse_dotbracket


def run():
    results = []

    def check(name, got, exp):
        ok = sorted(got) == sorted(exp)
        results.append((name, ok, sorted(got), sorted(exp)))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {sorted(got)} exp {sorted(exp)}")

    check("simple (((...)))", parse_dotbracket("(((...)))"), [(0, 8), (1, 7), (2, 6)])
    check("offset ..((..))..", parse_dotbracket("..((..)).."), [(2, 7), (3, 6)])
    # pseudoknot all bracket types: ([{<>}])
    check("pseudoknot ([{<>}])", parse_dotbracket("([{<>}])"),
          [(3, 4), (2, 5), (1, 6), (0, 7)])
    # crossing pseudoknot: ((..[[..))..]]
    check("crossing ((..[[..))..]]", parse_dotbracket("((..[[..))..]]"),
          [(0, 9), (1, 8), (4, 13), (5, 12)])

    # malformed: must warn, not crash
    print("--- malformed (expect warnings) ---")
    p = parse_dotbracket("(((..)")     # unmatched opens
    malformed_ok = isinstance(p, list)
    print(f"[{'PASS' if malformed_ok else 'FAIL'}] malformed handled (returned {p})")
    results.append(("malformed_no_crash", malformed_ok, p, "list"))

    n_pass = sum(1 for _, ok, *_ in results if ok)
    print(f"\n{n_pass}/{len(results)} passed")
    return n_pass == len(results)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)

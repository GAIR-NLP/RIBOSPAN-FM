#!/usr/bin/env python3
"""Unit tests for contact_metrics (Task 3). Writes
outputs/contact_eval/debug/contact_metric_tests.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contact_metrics import evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DBG = PROJECT_ROOT / "outputs" / "contact_eval" / "debug"


def main():
    DBG.mkdir(parents=True, exist_ok=True)
    out = {}
    L = 10
    gold = [(0, 9), (1, 8), (2, 7)]
    seq = "AAAAAAAAAA"   # all A; canonical pairing irrelevant here

    # Case A: perfect — gold pairs get the highest scores
    S = np.zeros((L, L))
    for (i, j) in gold:
        S[i, j] = S[j, i] = 10.0
    mA = evaluate(S, gold, seq)
    out["A_perfect"] = {"AUPRC": mA["AUPRC"], "P_at_L": mA["P_at_L"],
                        "best_F1": mA["best_F1"]}
    print(f"A perfect: AUPRC={mA['AUPRC']:.3f} P@L={mA['P_at_L']:.3f} F1={mA['best_F1']:.3f}")

    # Case B: random — averaged over many trials at larger L (random AUPRC -> pos_rate)
    Lb = 80
    goldB = [(i, Lb - 1 - i) for i in range(20)]            # 20 long-range pairs
    seqB = "A" * Lb
    aucs, prate = [], None
    for s in range(40):
        rng = np.random.RandomState(s)
        R = rng.rand(Lb, Lb); R = (R + R.T) / 2; np.fill_diagonal(R, 0)
        mb = evaluate(R, goldB, seqB)
        aucs.append(mb["AUPRC"]); prate = mb["num_gold_pairs"] / mb["num_candidates"]
    mB_auprc = float(np.mean(aucs)); pos_rate = prate
    out["B_random"] = {"mean_AUPRC_over_40": mB_auprc, "positive_rate": pos_rate}
    print(f"B random : mean AUPRC={mB_auprc:.3f} (pos_rate={pos_rate:.3f})")

    # Case C: transposed score == same after symmetrize (evaluate uses i<j of S)
    mC1 = evaluate(S, gold, seq)
    mC2 = evaluate(S.T, gold, seq)
    same = abs(mC1["AUPRC"] - mC2["AUPRC"]) < 1e-9
    out["C_transpose_invariant"] = {"equal": bool(same),
                                    "auprc_S": mC1["AUPRC"], "auprc_ST": mC2["AUPRC"]}
    print(f"C transpose-invariant: {same}")

    # Case D: |i-j|<4 filtering — a near-diagonal high score must be ignored
    S2 = np.zeros((L, L))
    S2[0, 1] = S2[1, 0] = 100.0          # |i-j|=1, should be filtered
    for (i, j) in gold:
        S2[i, j] = S2[j, i] = 1.0
    mD = evaluate(S2, gold, seq, min_sep=4)
    # the (0,1) near-diagonal must not be a candidate; gold still ranked
    cand_ok = mD["num_candidates"] == len(np.triu_indices(L, k=4)[0])
    out["D_minsep_filter"] = {"num_candidates": mD["num_candidates"],
                              "expected_candidates": int(len(np.triu_indices(L, k=4)[0])),
                              "near_diag_excluded": bool(cand_ok),
                              "AUPRC": mD["AUPRC"]}
    print(f"D min-sep filter: candidates={mD['num_candidates']} "
          f"expected={len(np.triu_indices(L,k=4)[0])} near_diag_excluded={cand_ok} "
          f"AUPRC={mD['AUPRC']:.3f}")

    checks = {
        "A_auprc_near_1": mA["AUPRC"] > 0.99,
        "A_f1_near_1": mA["best_F1"] > 0.99,
        "B_auprc_near_posrate": abs(mB_auprc - pos_rate) < 0.05,
        "C_transpose_invariant": same,
        "D_near_diag_excluded": cand_ok,
        "D_auprc_near_1_after_filter": mD["AUPRC"] > 0.99,
    }
    out["checks"] = checks
    out["all_passed"] = all(checks.values())
    json.dump(out, open(DBG / "contact_metric_tests.json", "w"), indent=2)
    print("\nchecks:", checks)
    print("ALL PASSED:", out["all_passed"])
    return out["all_passed"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

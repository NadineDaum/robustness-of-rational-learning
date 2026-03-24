from pathlib import Path
import pandas as pd


FILES = {
    "thompson_sampling": Path("results/data/fragility_map.csv"),
    "ucb1": Path("results/data/fragility_map_ucb1.csv"),
    "epsilon_greedy": Path("results/data/fragility_map_epsgreedy.csv"),
}

OUTPUT_DIR = Path("results/data")
SUMMARY_FILE = OUTPUT_DIR / "algorithm_comparison_summary.csv"
REGIME_COUNTS_FILE = OUTPUT_DIR / "algorithm_dominant_regime_counts.csv"


def load_with_label(label: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "algorithm" not in df.columns:
        df["algorithm"] = label
    else:
        df["algorithm"] = label  
    return df


def cell_regime(row: pd.Series) -> str:
    vals = {
        "optimal": row["p_optimal"],
        "lock_in": row["p_lock_in"],
        "ambiguous": row["p_ambiguous"],
    }
    return max(vals, key=vals.get)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algo, g in df.groupby("algorithm"):
        rows.append(
            {
                "algorithm": algo,
                "cells": len(g),
                "mean_p_optimal": g["p_optimal"].mean(),
                "mean_p_lock_in": g["p_lock_in"].mean(),
                "mean_p_ambiguous": g["p_ambiguous"].mean(),
                "share_optimal_ge_0_9": (g["p_optimal"] >= 0.9).mean(),
                "share_lockin_ge_0_5": (g["p_lock_in"] >= 0.5).mean(),
                "share_ambiguous_ge_0_5": (g["p_ambiguous"] >= 0.5).mean(),
                "max_p_lock_in": g["p_lock_in"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("algorithm").reset_index(drop=True)


def regime_counts(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["dominant_regime"] = tmp.apply(cell_regime, axis=1)
    out = (
        tmp.groupby(["algorithm", "dominant_regime"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["algorithm", "count"], ascending=[True, False])
    )
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = [load_with_label(label, path) for label, path in FILES.items()]
    df = pd.concat(frames, ignore_index=True)

    summary_df = summarize(df)
    regime_df = regime_counts(df)

    # Console output
    print("\n=== Algorithm summary ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Dominant regime counts by algorithm ===")
    print(regime_df.to_string(index=False))

    print("\nInterpretation hint:")
    print("- TS and epsilon-greedy typically show stronger lock-in regions.")
    print("- UCB1 often shifts toward ambiguous rather than lock-in in hard cells.")

    # Appendix-ready CSV exports
    summary_df.to_csv(SUMMARY_FILE, index=False)
    regime_df.to_csv(REGIME_COUNTS_FILE, index=False)

    print(f"\nSaved: {SUMMARY_FILE}")
    print(f"Saved: {REGIME_COUNTS_FILE}")


if __name__ == "__main__":
    main()
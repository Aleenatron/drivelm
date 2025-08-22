import json, argparse, pandas as pd, pathlib

def load_jsonl(path):
    p = pathlib.Path(path)
    if not p.exists(): return []
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="data/multi_frame/human_ratings.jsonl")
    ap.add_argument("--out", default="multi_frame_results/human_scores.csv")
    args = ap.parse_args()

    rows = load_jsonl(args.ratings)
    if not rows:
        print("No ratings yet."); return

    df = pd.DataFrame(rows)

    # Normalize columns that might be missing
    for col in ["category","reliable"]:
        if col not in df: df[col] = None

    # Category summary
    grp = (df
           .groupby("category", dropna=False, as_index=False)
           .agg(num_questions=("question","count"),
                reliability=("reliable", "mean"))
           .sort_values("category"))
    grp["reliability"] = (grp["reliability"]*100).round(1)

    # Example pass/fail per category
    examples = []
    for cat, sub in df.groupby("category"):
        ex_pass = sub[sub["reliable"]==True].head(1)
        ex_fail = sub[sub["reliable"]==False].head(1)
        examples.append({
            "category": cat,
            "#questions": len(sub),
            "accuracy(%)": round(sub["reliable"].mean()*100,1) if len(sub) else 0.0,
            "example_pass": (f'Q: {ex_pass.iloc[0]["question"]} → {ex_pass.iloc[0]["pred"]}'
                             if len(ex_pass) else "—"),
            "example_fail": (f'Q: {ex_fail.iloc[0]["question"]} → {ex_fail.iloc[0]["pred"]}'
                             if len(ex_fail) else "—"),
        })
    ex_df = pd.DataFrame(examples).sort_values("category")

    # Save both
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grp.to_csv(args.out, index=False)
    ex_df.to_csv(args.out.replace(".csv","_examples.csv"), index=False)

    print("Saved:", args.out, "and", args.out.replace(".csv","_examples.csv"))

if __name__ == "__main__":
    main()

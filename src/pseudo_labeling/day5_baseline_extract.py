from pathlib import Path
import pandas as pd

RUN_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\baseline_supervised")
MODEL = "YOLOv8n"
EPOCHS = 20

# choose criterion for "best"
BEST_BY = "metrics/mAP50(B)"  # or "metrics/mAP50-95(B)"

def main():
    csv_path = RUN_DIR / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    best_idx = df[BEST_BY].astype(float).idxmax()
    best = df.loc[best_idx]

    row = {
        "Model": MODEL,
        "Epochs": EPOCHS,
        "BestEpoch": int(best["epoch"]),
        "Precision": float(best["metrics/precision(B)"]),
        "Recall": float(best["metrics/recall(B)"]),
        "mAP@0.5": float(best["metrics/mAP50(B)"]),
        "mAP@0.5:0.95": float(best["metrics/mAP50-95(B)"]),
        "BestBy": BEST_BY,
        "RunDir": str(RUN_DIR),
    }

    out_csv = RUN_DIR / "baseline_row.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)

    print("✅ Baseline row saved:", out_csv)
    print("\n=== BASELINE ROW (copy into thesis table) ===")
    for k, v in row.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

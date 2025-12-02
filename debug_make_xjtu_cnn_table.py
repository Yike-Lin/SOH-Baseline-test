import os
import numpy as np
import csv
from openpyxl import Workbook

BASE_DIR = "results/XJTU-charge"
MODEL = "CNN"
N_EXP = 3

BATCH_TO_IDS = {
    1: range(1, 9),
    2: range(1, 16),
    3: range(1, 9),
    4: range(1, 9),
    5: range(1, 9),
    6: range(1, 9),
}

OUT_CSV = "XJTU_CNN_table.csv"
OUT_XLSX = "XJTU_CNN_table.xlsx"


def npz_path_of(batch, battery, exp_id):
    return os.path.join(
        BASE_DIR,
        MODEL,
        f"batch{batch}-testbattery{battery}",
        f"experiment{exp_id}",
        f"{MODEL}_charge_results.npz",
    )


def load_cnn_errors_for_cell(batch, battery):
    all_errs = []

    print(f"\n=== Batch {batch} Battery {battery} ===")
    for e in range(1, N_EXP + 1):
        path = npz_path_of(batch, battery, e)
        if not os.path.exists(path):
            print(f"[MISS] exp{e}: {path}")
            continue

        data = np.load(path, allow_pickle=True)
        if "test_errors" not in data:
            print(f"[BAD ] exp{e}: no 'test_errors' in {path}")
            continue

        err = data["test_errors"]
        print(f"[READ] exp{e}: test_errors shape = {err.shape}, value = {err}")

        # 兼容 (3,), (1,3), (3,1)
        err = np.array(err).reshape(-1)
        if err.shape[0] < 3:
            print(f"[BAD ] exp{e}: test_errors length < 3, got {err.shape}")
            continue

        mae, mape, mse = float(err[0]), float(err[1]), float(err[2])
        all_errs.append([mae, mape, mse])

    if len(all_errs) == 0:
        print("[WARN] no valid experiments for this cell.")
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    all_errs = np.array(all_errs, dtype=float)
    mean_err = all_errs.mean(axis=0)

    print(f"[MEAN] valid_exp={len(all_errs)} "
          f"MAE={mean_err[0]:.6f}, MAPE={mean_err[1]:.6f}, MSE={mean_err[2]:.6f} (raw)")
    print(f"[MEAN×1000] MAE={mean_err[0]*1000:.3f}, "
          f"MAPE={mean_err[1]*1000:.3f}, MSE={mean_err[2]*1000:.3f}")

    return mean_err * 1000.0


def save_csv(rows, header):
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if (isinstance(v, float) and np.isnan(v)) else v for v in row])


def save_xlsx(rows, header):
    wb = Workbook()
    ws = wb.active
    ws.title = "XJTU_CNN"
    ws.append(header)
    for row in rows:
        ws.append([None if (isinstance(v, float) and np.isnan(v)) else v for v in row])
    wb.save(OUT_XLSX)


def main():
    header = ["Batch", "Battery", "CNN_MAE", "CNN_MAPE", "CNN_MSE"]
    rows = []

    for batch, ids_list in BATCH_TO_IDS.items():
        for battery in ids_list:
            mae, mape, mse = load_cnn_errors_for_cell(batch, battery)

            mae = float(f"{mae:.3f}") if not np.isnan(mae) else np.nan
            mape = float(f"{mape:.3f}") if not np.isnan(mape) else np.nan
            mse = float(f"{mse:.3f}") if not np.isnan(mse) else np.nan

            rows.append([batch, battery, mae, mape, mse])

    save_csv(rows, header)
    save_xlsx(rows, header)

    print("\nDone.")
    print(f"Saved CSV:  {OUT_CSV}")
    print(f"Saved XLSX: {OUT_XLSX}")


if __name__ == "__main__":
    main()
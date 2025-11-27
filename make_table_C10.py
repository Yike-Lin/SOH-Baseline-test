import os
import numpy as np
import csv

# 基础路径：根据你训练脚本里的 save_folder 来改
# 你之前用的是：results/{args.data}-{args.input_type}/...
# 对应 XJTU + charge = "results/XJTU-charge"
BASE_DIR = "results/XJTU-charge"

# 5 个模型，对应 Table C.10 的列顺序
MODELS = ["CNN", "LSTM", "GRU", "MLP", "Attention"]

# XJTU 各个 batch 的电池数
BATCH_TO_IDS = {
    1: range(1, 9),
    2: range(1, 16),
    3: range(1, 9),
    4: range(1, 9),
    5: range(1, 9),
    6: range(1, 9),
    # 7: range(1, 9),
}

N_EXP = 3  # 每个 (batch, battery, model) 的实验次数，对应你训练里的 for e in range(3)

def load_errors_for_cell(model, batch, battery, n_exp=N_EXP):
    """
    读取某个 (model, batch, battery) 的多次实验结果，
    返回 (MAE, MAPE, MSE) 的平均值，并且 *1000（和表里展示一致）

    假设 npz 结构中:
      test_errors = [MAE, MAPE, MSE, R2] 或至少前三个是 MAE,MAPE,MSE
    """
    all_errs = []

    for e in range(1, n_exp + 1):
        # 对应你训练时的 save_folder
        npz_path = os.path.join(
            BASE_DIR,
            model,
            f"batch{batch}-testbattery{battery}",
            f"experiment{e}",
            f"{model}_charge_results.npz"
        )

        if not os.path.exists(npz_path):
            print(f"[WARN] Missing file: {npz_path}")
            continue

        data = np.load(npz_path)
        if "test_errors" not in data:
            print(f"[WARN] 'test_errors' not in {npz_path}")
            continue

        err = data["test_errors"]
        # 取前三个: MAE, MAPE, MSE
        err = err[:3]
        all_errs.append(err)

    if len(all_errs) == 0:
        # 没有任何有效实验，返回 NaN
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    all_errs = np.stack(all_errs, axis=0)  # (n_valid_exp, 3)
    mean_err = all_errs.mean(axis=0)       # (3,)

    # 展示用放大 1000 倍（和 Appendix Table Note 一致）
    return mean_err * 1000.0


def main():
    # 输出文件：你可以改成任意名字
    out_csv = "XJTU_table_C10_like.csv"

    # 构造表头：Batch, Battery, 然后每个模型 3 列
    header = ["Batch", "Battery"]
    for m in MODELS:
        header.extend([
            f"{m}_MAE",
            f"{m}_MAPE",
            f"{m}_MSE",
        ])

    rows = []

    for batch, ids_list in BATCH_TO_IDS.items():
        for battery in ids_list:
            row = [batch, battery]

            for model in MODELS:
                mae, mape, mse = load_errors_for_cell(model, batch, battery)
                # 保留 3 位小数，方便和论文里的数对比
                row.extend([
                    f"{mae:.3f}" if not np.isnan(mae) else "",
                    f"{mape:.3f}" if not np.isnan(mape) else "",
                    f"{mse:.3f}" if not np.isnan(mse) else "",
                ])

            rows.append(row)

    # 写 CSV 文件
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Done. Saved table-like results to: {out_csv}")
    print("可以用 Excel 打开这个 CSV，对照 Appendix 的 Table C.10 来看。")


if __name__ == "__main__":
    main()
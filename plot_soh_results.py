import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


# 命令
# python plot_soh_results.py \
#     --data XJTU \
#     --input_type charge \
#     --model CNN \
#     --batch 1 \
#     --test_battery_id 6 \
#     --experiment 1 \
#     --save_dir figs \
#     --no_show


def load_results(root, data, input_type, model, batch, test_battery_id, experiment):
    """
    根据路径规则加载 npz：
    results/{data}-{input_type}/{model}/batch{batch}-testbattery{test_battery_id}/experiment{experiment}/{model}_{input_type}_results.npz
    """
    folder = os.path.join(
        root,
        f"{data}-{input_type}",
        model,
        f"batch{batch}-testbattery{test_battery_id}",
        f"experiment{experiment}",
    )
    prefix = f"{model}_{input_type}"
    npz_path = os.path.join(folder, f"{prefix}_results.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"npz not found: {npz_path}")

    print(f"Loading: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    train_loss = data["train_loss"]
    valid_loss = data["valid_loss"]
    true_label = data["true_label"]
    pred_label = data["pred_label"]
    test_errors = data["test_errors"]  # 一般是 [MAE, MAPE, MSE] 之类

    return {
        "path": npz_path,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "true_label": true_label,
        "pred_label": pred_label,
        "test_errors": test_errors,
    }


def plot_loss_curve(train_loss, valid_loss, title=None, save_path=None):
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, valid_loss, label="valid loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved loss curve to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_true_vs_pred(true_label, pred_label, title=None, save_path=None):
    true_label = np.squeeze(true_label)
    pred_label = np.squeeze(pred_label)

    plt.figure()
    plt.scatter(true_label, pred_label, s=10, alpha=0.7)
    min_v = min(true_label.min(), pred_label.min())
    max_v = max(true_label.max(), pred_label.max())
    plt.plot([min_v, max_v], [min_v, max_v], "--")  # 对角线 y = x

    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    if title:
        plt.title(title)
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved true-vs-pred to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_error_curve(true_label, pred_label, title=None, save_path=None):
    true_label = np.squeeze(true_label)
    pred_label = np.squeeze(pred_label)
    errors = pred_label - true_label  # 残差

    idx = np.arange(len(errors))

    plt.figure()
    plt.plot(idx, errors, marker=".", linestyle="-", linewidth=1)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("sample index")
    plt.ylabel("pred - true")
    if title:
        plt.title(title)
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved error curve to: {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot SOH experiment results from npz.")
    parser.add_argument("--root", type=str, default="results",
                        help="根目录，比如 results")
    parser.add_argument("--data", type=str, default="XJTU",
                        choices=["XJTU", "MIT"])
    parser.add_argument("--input_type", type=str, default="charge",
                        choices=["charge", "partial_charge", "handcraft_features"])
    parser.add_argument("--model", type=str, default="CNN",
                        choices=["CNN", "LSTM", "GRU", "MLP", "Attention"])
    parser.add_argument("--batch", type=int, default=1,
                        help="batch id")
    parser.add_argument("--test_battery_id", type=int, default=1,
                        help="test battery id")
    parser.add_argument("--experiment", type=int, default=1,
                        help="experiment index (1,2,3...)")
    parser.add_argument("--no_show", action="store_true",
                        help="只保存图片，不弹出窗口")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="如果指定，就把图保存到这个目录下")

    args = parser.parse_args()

    res = load_results(
        root=args.root,
        data=args.data,
        input_type=args.input_type,
        model=args.model,
        batch=args.batch,
        test_battery_id=args.test_battery_id,
        experiment=args.experiment,
    )

    # 打印一下 test_errors，方便和论文表对比
    print("test_errors (from npz) :", res["test_errors"])

    # 准备保存路径
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        base_name = (
            f"{args.data}-{args.input_type}-"
            f"{args.model}-batch{args.batch}-bat{args.test_battery_id}-exp{args.experiment}"
        )
        loss_path = os.path.join(args.save_dir, base_name + "_loss.png")
        tv_path = os.path.join(args.save_dir, base_name + "_true_vs_pred.png")
        err_path = os.path.join(args.save_dir, base_name + "_error.png")
    else:
        loss_path = tv_path = err_path = None

    title_prefix = (
        f"{args.data}-{args.input_type}-{args.model}, "
        f"batch{args.batch}-battery{args.test_battery_id}-exp{args.experiment}"
    )

    # 1) train / valid loss 曲线
    plot_loss_curve(
        res["train_loss"],
        res["valid_loss"],
        title=title_prefix + " - loss",
        save_path=loss_path if args.save_dir else None,
    )

    # 2) true vs pred 散点
    plot_true_vs_pred(
        res["true_label"],
        res["pred_label"],
        title=title_prefix + " - true vs pred",
        save_path=tv_path if args.save_dir else None,
    )

    # 3) 残差曲线
    plot_error_curve(
        res["true_label"],
        res["pred_label"],
        title=title_prefix + " - error (pred-true)",
        save_path=err_path if args.save_dir else None,
    )

    if args.no_show:
        # 已经保存过图了，直接退出
        return
    else:
        # 如果没指定 no_show，前面函数默认调用 plt.show() 已经弹出窗口
        pass


if __name__ == "__main__":
    main()
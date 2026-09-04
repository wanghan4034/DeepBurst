import pandas as pd
import numpy as np


def add_mean_threshold_labels(df, bs_col="bs", bf_col="bf"):
    """
    使用 mean 作为 threshold，生成 bs/bf 的二分类标签。
    """
    df = df.copy()

    bs_mean = df[bs_col].mean()
    bf_mean = df[bf_col].mean()

    df["bs_label"] = (df[bs_col] > bs_mean).astype(int)
    df["bf_label"] = (df[bf_col] > bf_mean).astype(int)

    return df, bs_mean, bf_mean


def make_joint_exclude_mid20_dataset(
    df,
    bs_col="bs",
    bf_col="bf",
    bs_label_name="bs_label",
    bf_label_name="bf_label",
):
    """
    同时剔除 bs 和 bf 各自 median 上下 10% 的样本。

    对 bs:
        q40_bs < bs < q60_bs 的样本视为 bs 模糊区。
    对 bf:
        q40_bf < bf < q60_bf 的样本视为 bf 模糊区。

    最终保留：
        bs <= q40_bs 或 bs >= q60_bs
    且
        bf <= q40_bf 或 bf >= q60_bf

    标签：
        bs >= q60_bs -> 1
        bs <= q40_bs -> 0

        bf >= q60_bf -> 1
        bf <= q40_bf -> 0
    """
    df = df.copy()

    bs_q40 = df[bs_col].quantile(0.40)
    bs_q60 = df[bs_col].quantile(0.60)

    bf_q40 = df[bf_col].quantile(0.40)
    bf_q60 = df[bf_col].quantile(0.60)

    bs_keep = (df[bs_col] <= bs_q40) | (df[bs_col] >= bs_q60)
    bf_keep = (df[bf_col] <= bf_q40) | (df[bf_col] >= bf_q60)

    joint_keep = bs_keep & bf_keep

    sub_df = df.loc[joint_keep].copy()

    sub_df[bs_label_name] = (sub_df[bs_col] >= bs_q60).astype(int)
    sub_df[bf_label_name] = (sub_df[bf_col] >= bf_q60).astype(int)

    threshold_info = {
        "bs_q40": bs_q40,
        "bs_q60": bs_q60,
        "bf_q40": bf_q40,
        "bf_q60": bf_q60,
        "original_n": len(df),
        "kept_n": len(sub_df),
        "removed_n": len(df) - len(sub_df),
        "kept_ratio": len(sub_df) / len(df),
        "removed_ratio": 1 - len(sub_df) / len(df),
    }

    return sub_df, threshold_info


def process_one_eid_joint(
    eid,
    input_template="extra/datasets/processed/v2/meta_datasets/meta_data_{eid}_deeptx.csv",
    output_dir="benchmarks/revised_v2/reviewer3/meta_datasets",
):
    """
    对单个 eid 生成：
    1. mean threshold labels 文件
    2. 同时剔除 bs/bf median 上下 10% 的 joint exclude-mid20 文件
    """
    input_path = input_template.format(eid=eid)

    df = pd.read_csv(input_path)

    # 1. mean threshold
    df_mean, bs_mean, bf_mean = add_mean_threshold_labels(df)

    mean_output_path = f"{output_dir}/meta_data_{eid}_deeptx_mean_threshold.csv"
    df_mean.to_csv(mean_output_path, index=False)

    # 2. joint exclude mid20
    df_joint, threshold_info = make_joint_exclude_mid20_dataset(df)

    joint_output_path = f"{output_dir}/meta_data_{eid}_deeptx_exclude_mid20.csv"
    df_joint.to_csv(joint_output_path, index=False)

    summary = {
        "eid": eid,
        "original_n": len(df),

        "bs_mean": bs_mean,
        "bf_mean": bf_mean,

        "bs_q40": threshold_info["bs_q40"],
        "bs_q60": threshold_info["bs_q60"],
        "bf_q40": threshold_info["bf_q40"],
        "bf_q60": threshold_info["bf_q60"],

        "joint_exclude_mid20_kept_n": threshold_info["kept_n"],
        "joint_exclude_mid20_removed_n": threshold_info["removed_n"],
        "joint_exclude_mid20_kept_ratio": threshold_info["kept_ratio"],
        "joint_exclude_mid20_removed_ratio": threshold_info["removed_ratio"],

        "mean_output_path": mean_output_path,
        "joint_exclude_mid20_output_path": joint_output_path,
    }

    return df_mean, df_joint, summary

if __name__ == "__main__":
    eids = ["E003", "E118", "E116"]
    all_summaries = []
    for eid in eids:
        df_mean, df_joint, summary = process_one_eid_joint(eid)
        all_summaries.append(summary)
    summary_df = pd.DataFrame(all_summaries)
    print(summary_df)
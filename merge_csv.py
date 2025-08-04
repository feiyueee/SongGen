import pandas as pd
import os


def process_csv_files(results_path, testa_full_path, output_path):
    """
    处理CSV文件：
    1. 读取results_epoch2.csv（无表头，空格分隔）
    2. 读取aigc_speech_testA_full.csv（有表头）
    3. 合并两个文件
    4. 从wav_path中提取文件名
    5. 根据score预测label
    6. 保存为新的CSV文件（包含utt, wav_path, label三列）
    """
    # 读取results_epoch2.csv（无表头，空格分隔）
    results_df = pd.read_csv(
        results_path,
        sep=' ',
        header=None,
        names=['utt', 'score']
    )

    # 读取aigc_speech_testA_full.csv（有表头）
    testa_df = pd.read_csv(testa_full_path)

    # 合并两个DataFrame
    merged_df = pd.merge(testa_df, results_df, on='utt', how='left')

    # 从wav_path中提取文件名
    merged_df['wav_path'] = merged_df['wav_path'].apply(
        lambda x: os.path.basename(x) if isinstance(x, str) else x
    )

    # 根据score预测label
    merged_df['label'] = merged_df['score'].apply(
        lambda x: 'Bonafide' if x > 0.5 else 'Spoof'
    )

    # 选择需要的列（只保留utt, wav_path, label）
    final_df = merged_df[['utt', 'wav_path', 'label']]

    # 保存结果
    final_df.to_csv(output_path, index=False)
    print(f"结果已保存至: {output_path}")
    print(f"处理了 {len(final_df)} 条记录")


if __name__ == "__main__":
    # 文件路径配置
    results_path = "results_epoch2.csv"  # 模型输出结果文件
    testa_full_path = "aigc_speech_testA_full.csv"  # 原始测试集文件
    output_path = "final_results.csv"  # 输出文件路径

    # 执行处理
    process_csv_files(results_path, testa_full_path, output_path)
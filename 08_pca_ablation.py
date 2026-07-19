import pandas as pd
from config import OUTPUT_DIR
from data_utils import load_original_data
from training_utils import create_shared_split, evaluate_dataset

DATASETS = {
    'Clinical': OUTPUT_DIR / 'clinical_dataset.csv',
    'Lifestyle': OUTPUT_DIR / 'lifestyle_dataset.csv',
    'Merged': OUTPUT_DIR / 'merged_dataset.csv',
}


def main():
    full = load_original_data()
    train_ids, test_ids = create_shared_split(full)

    all_results = []
    for dataset_name, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(f'{path} not found. Run 02_split_dataset.py / 05_fusion.py first.')
        df = pd.read_csv(path)
        results, _ = evaluate_dataset(dataset_name, df, train_ids, test_ids, use_pca=False)
        all_results.extend(results)

    no_pca_df = pd.DataFrame(all_results)
    no_pca_df.to_csv(OUTPUT_DIR / 'all_results_no_pca_ablation.csv', index=False, encoding='utf-8-sig')

    with_pca_path = OUTPUT_DIR / 'all_results_pca.csv'
    if not with_pca_path.exists():
        print('\nWith-PCA results not found; run 07_compare_results_pca.py first to get a side-by-side comparison.')
        return

    with_pca_df = pd.read_csv(with_pca_path)
    merge_cols = ['Dataset', 'Model']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'Training_Time_Seconds']

    comparison = with_pca_df[merge_cols + metrics].merge(
        no_pca_df[merge_cols + metrics],
        on=merge_cols, suffixes=('_With_PCA', '_Without_PCA'),
    )
    for metric in metrics:
        comparison[f'{metric}_Diff'] = (
            comparison[f'{metric}_With_PCA'] - comparison[f'{metric}_Without_PCA']
        )
    comparison.to_csv(OUTPUT_DIR / 'pca_ablation_comparison.csv', index=False, encoding='utf-8-sig')

    print('\n=== PCA ABLATION: WITH PCA vs WITHOUT PCA ===')
    display_cols = merge_cols + [
        'F1_With_PCA', 'F1_Without_PCA', 'F1_Diff',
        'ROC_AUC_With_PCA', 'ROC_AUC_Without_PCA', 'ROC_AUC_Diff',
        'Training_Time_Seconds_With_PCA', 'Training_Time_Seconds_Without_PCA',
    ]
    print(comparison[display_cols].round(4).to_string(index=False))


if __name__ == '__main__':
    main()

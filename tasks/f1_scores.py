import argparse
import pandas as pd
from sklearn.metrics import f1_score
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--pre_path', type=str)

    args = parser.parse_args()

    return args

def calculate_f1(gt_df, pred_df):
    # Ensure the same structure
    assert list(gt_df.columns) == list(pred_df.columns), "Mismatched columns"

    # Extract label columns
    label_cols = [col for i, col in enumerate(gt_df.columns) if i != 0]

    # Store all true and predicted labels
    all_true = []
    all_pred = []

    for col in label_cols:
        gt_col = gt_df[col]
        pred_col = pred_df[col]

        # Only consider rows where both are -1 or 1 (ignore blanks)
        valid_mask = gt_col.isin([-1, 1, 0]) & pred_col.isin([-1, 1, 0])

        if valid_mask.sum() == 0:
            continue  # Skip label if no valid comparisons

        all_true.extend(gt_col[valid_mask])
        all_pred.extend(pred_col[valid_mask])

    # Calculate F1
    f1 = f1_score(all_true, all_pred, average='macro')  # or 'micro' depending on use case

    print("F1 Score:", f1)

def main():
    args = parse_agrs()
    gt_df = pd.read_csv(args.gt_path)
    pred_df = pd.read_csv(args.pre_path)
    calculate_f1(gt_df, pred_df)

if __name__ == "__main__":
    main()
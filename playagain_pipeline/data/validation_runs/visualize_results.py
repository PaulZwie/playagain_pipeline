import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

def visualize_results(run_dir):
    results_csv = os.path.join(run_dir, 'results.csv')
    per_class_f1_csv = os.path.join(run_dir, 'per_class_f1.csv')
    results_json = os.path.join(run_dir, 'results.json')

    # Load data
    df = pd.read_csv(results_csv)
    df_f1 = pd.read_csv(per_class_f1_csv)
    with open(results_json, 'r') as f:
        res_json = json.load(f)

    # Box and whiskers for accuracy and macro_f1
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='model_type', y='accuracy')
    plt.title('Accuracy by Model')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='model_type', y='macro_f1')
    plt.title('Macro F1 by Model')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'boxplots.png'))
    print(f"Saved boxplots to {os.path.join(run_dir, 'boxplots.png')}")

    # Get class names in order from per_class_f1.csv
    first_fold = df_f1['fold_id'].iloc[0]
    first_model = df_f1['model_type'].iloc[0]
    classes = df_f1[(df_f1['fold_id'] == first_fold) & (df_f1['model_type'] == first_model)]['class'].tolist()

    # Confusion Matrices
    agg_conf = res_json.get('aggregate_confusion', {})
    
    n_models = len(agg_conf)
    if n_models > 0:
        cols = 3
        rows = int(np.ceil(n_models / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()

        for idx, (model, cm) in enumerate(agg_conf.items()):
            ax = axes[idx]
            cm_np = np.array(cm)
            sns.heatmap(cm_np, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_title(f"Confusion Matrix: {model}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        for i in range(len(agg_conf), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'confusion_matrices.png'))
        print(f"Saved confusion matrices to {os.path.join(run_dir, 'confusion_matrices.png')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Validation Results')
    parser.add_argument('run_dir', type=str, help='Path to the validation run directory')
    args = parser.parse_args()
    
    if os.path.exists(args.run_dir):
        visualize_results(args.run_dir)
    else:
        print(f"Directory {args.run_dir} not found.")
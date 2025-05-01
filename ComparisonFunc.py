# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 09:48:00 2025

@author: DELL
"""

from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison(results_dict):
    """
    Plots a side-by-side bar chart of Accuracy, Precision, Recall, and F1-score
    for multiple models based on classification_report and accuracy_score.

    Parameters:
    - results_dict (dict): Format {
          'Model Name': {
              'y_true': true_labels,
              'y_pred': predicted_labels
          },
          ...
      }
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    scores = {metric: [] for metric in metrics}
    model_names = []

    for model_name, data in results_dict.items():
        y_true = data['y_true']
        y_pred = data['y_pred']

        report = classification_report(y_true, y_pred, output_dict=True)
        acc = accuracy_score(y_true, y_pred)
        scores['Accuracy'].append(acc)
        scores['Precision'].append(report['weighted avg']['precision'])
        scores['Recall'].append(report['weighted avg']['recall'])
        scores['F1-score'].append(report['weighted avg']['f1-score'])
        model_names.append(model_name)

    x = np.arange(len(metrics))
    total_models = len(model_names)
    width = 0.8 / total_models  # Adjust bar width to fit all models

    plt.figure(figsize=(10, 6))

    for idx, model_name in enumerate(model_names):
        offsets = x - 0.4 + width/2 + idx * width
        values = [scores[m][idx] for m in metrics]
        plt.bar(offsets, values, width=width, label=model_name)
        for i, v in enumerate(values):
            plt.text(offsets[i], v + 0.005, f'{v:.2f}', ha='center', fontsize=9)

    plt.xticks(x, metrics, fontsize=12)
    plt.ylim(0.75, 1.0)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

import torch
from train import train_model
from evaluate import plot_predictions

if __name__ == "__main__":
    train_model()
    # Example usage of plot_predictions
    # subject_id = 2  # Choose any subject between 1 to 11
    # y_true, y_pred = all_predictions[subject_id]
    # plot_predictions(y_true, y_pred, subject=subject_id)
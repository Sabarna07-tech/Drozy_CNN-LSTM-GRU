import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_predictions(y_true, y_pred, subject):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='True Labels', marker='o', linestyle='--', color='b')
    plt.plot(y_pred, label='Predicted Labels', marker='x', linestyle='-', color='r')
    plt.title(f'Subject {subject} - True vs. Predicted Labels')
    plt.xlabel('Sample Index')
    plt.ylabel('State (0: Alert, 1: Drowsy)')
    plt.legend()
    plt.grid(True)
    plt.show()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Alert", "Drowsy"])
    disp.plot(cmap='Blues')
    plt.title(f'Subject {subject} - Confusion Matrix')
    plt.show()
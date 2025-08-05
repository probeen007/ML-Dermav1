from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, test_generator):
    """
    Evaluate the model on the test set and generate a classification report and confusion matrix.

    Args:
        model: The trained TensorFlow model.
        test_generator: Test data generator.
    """
    # Evaluate test accuracy
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test Accuracy: {test_acc * 100:.2f}%')

    # Predict classes
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    # Classification report
    class_labels = list(test_generator.class_indices.keys())
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.show()
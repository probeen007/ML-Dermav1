import os
import matplotlib.pyplot as plt

def analyze_dataset(train_dir):
    """
    Analyze the dataset and plot the class distribution.

    Args:
        train_dir: Path to the training dataset directory.
    """
    class_counts = {}
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))

    # Plot class distribution
    plt.figure(figsize=(15, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=90)
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()

    return class_counts
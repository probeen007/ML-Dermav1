from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_data(train_dir, test_dir):
    """
    Preprocess the data and create data generators.

    Args:
        train_dir: Path to the training dataset directory.
        test_dir: Path to the test dataset directory.

    Returns:
        train_generator: Training data generator.
        validation_generator: Validation data generator.
        test_generator: Test data generator.
    """
    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # Split training data into 80% train, 20% validation
    )

    # Load training and validation data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Test data (no augmentation)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # Maintain order for evaluation
    )

    return train_generator, validation_generator, test_generator
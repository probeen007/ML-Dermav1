import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import numpy as np

def train_model(model, train_generator, validation_generator):
    """
    Train the model using the provided training and validation data generators.

    Args:
        model: The TensorFlow model to train.
        train_generator: Training data generator.
        validation_generator: Validation data generator.

    Returns:
        history: Training history object.
    """
    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights))

    # Callbacks
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('models/best_model.keras', save_best_only=True)
    ]

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20,
        class_weight=class_weights,
        callbacks=callbacks
    )

    return history

def fine_tune_model(model, train_generator, validation_generator):
    """
    Fine-tune the model by unfreezing some layers of the base model.

    Args:
        model: The TensorFlow model to fine-tune.
        train_generator: Training data generator.
        validation_generator: Validation data generator.

    Returns:
        history_fine: Fine-tuning history object.
    """
    # Unfreeze the top 20 layers while keeping BatchNorm layers frozen
    base_model = model.layers[1]
    base_model.trainable = True
    for layer in base_model.layers[:20]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    # Recompile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Continue training
    history_fine = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
    )

    return history_fine
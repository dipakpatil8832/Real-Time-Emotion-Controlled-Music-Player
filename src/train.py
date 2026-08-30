"""
FER-2013 Emotion Classification Training Pipeline.
Handles dataset ingestion, augmentation, class weighting, two-stage transfer learning,
LR scheduling, early stopping, evaluation metrics, and confusion matrix export.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
        CSVLogger
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except (ImportError, Exception):
    tf = None
    TF_AVAILABLE = False

from src.config import (
    ASSETS_DIR,
    DATA_DIR,
    EMOTION_TO_IDX,
    EMOTIONS,
    IDX_TO_EMOTION,
    MODELS_DIR,
    ModelConfig,
    model_config
)
from src.model import build_emotion_model, unfreeze_model_for_fine_tuning
from src.utils import logger, plot_confusion_matrix, save_json


def create_data_generators(
    train_dir: Path,
    val_dir: Path,
    test_dir: Optional[Path] = None,
    config: ModelConfig = model_config
) -> Tuple[ImageDataGenerator, ImageDataGenerator, Optional[ImageDataGenerator]]:
    """
    Builds data generators with robust data augmentation for training
    and deterministic rescaling for validation/testing.
    """
    target_size = (config.image_height, config.image_width)

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_test_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=target_size,
        batch_size=config.batch_size,
        classes=EMOTIONS,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = val_test_datagen.flow_from_directory(
        str(val_dir),
        target_size=target_size,
        batch_size=config.batch_size,
        classes=EMOTIONS,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = None
    if test_dir and test_dir.exists():
        test_gen = val_test_datagen.flow_from_directory(
            str(test_dir),
            target_size=target_size,
            batch_size=config.batch_size,
            classes=EMOTIONS,
            class_mode="categorical",
            shuffle=False
        )

    return train_gen, val_gen, test_gen


def load_fer2013_csv(
    csv_path: Path,
    config: ModelConfig = model_config,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses standard FER-2013 CSV format (emotion, pixels, Usage) and filters to 5 target emotions.
    
    FER2013 original mappings:
    0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
    We map: 0->angry, 3->happy, 4->sad, 5->surprise, 6->neutral
    """
    logger.info("Loading FER-2013 dataset from: %s", csv_path)
    df = pd.read_csv(csv_path)

    # Map original FER2013 indices to our 5 emotions
    fer_mapping = {0: "angry", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
    df = df[df["emotion"].isin(fer_mapping.keys())].copy()
    df["emotion_name"] = df["emotion"].map(fer_mapping)
    df["target"] = df["emotion_name"].map(EMOTION_TO_IDX)

    images = []
    for pixel_seq in df["pixels"]:
        arr = np.fromstring(pixel_seq, sep=" ", dtype=np.uint8).reshape((48, 48))
        # Resize to target model resolution (e.g. 224x224) and repeat 3 channels for MobileNetV2
        rgb_img = np.stack([arr] * 3, axis=-1)
        rgb_resized = tf.image.resize(rgb_img, (config.image_height, config.image_width)).numpy()
        images.append(rgb_resized)

    X = np.array(images, dtype=np.float32)
    y = tf.keras.utils.to_categorical(df["target"].values, num_classes=len(EMOTIONS))

    # Split: Train, Validation, Test with stratification
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_relative_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_ratio, random_state=random_state, stratify=y_train_val
    )

    logger.info(
        "Loaded dataset splits: Train=%d, Val=%d, Test=%d",
        len(X_train), len(X_val), len(X_test)
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_balanced_class_weights(y_train_categorical: np.ndarray) -> Dict[int, float]:
    """Computes balanced class weights to address dataset emotion imbalance."""
    y_integers = np.argmax(y_train_categorical, axis=1)
    classes = np.unique(y_integers)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_integers
    )
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    logger.info("Computed class weights: %s", class_weight_dict)
    return class_weight_dict


def train_emotion_model(
    data_source: str,
    epochs: int = 40,
    batch_size: int = 32,
    fine_tune_epochs: int = 15
) -> tf.keras.Model:
    """
    Executes full training pipeline with Stage 1 (Frozen Backbone) + Stage 2 (Fine-Tuning).
    """
    config = ModelConfig(epochs=epochs, batch_size=batch_size)
    data_path = Path(data_source)

    # 1. Prepare Model Architecture
    model = build_emotion_model(config, trainable_base=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    # 2. Callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath=str(config.model_save_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    csv_logger_cb = CSVLogger(
        filename=str(DATA_DIR / "training_history.csv"),
        separator=",",
        append=False
    )
    callbacks = [checkpoint_cb, early_stop_cb, reduce_lr_cb, csv_logger_cb]

    # 3. Training Execution
    if data_path.is_file() and data_path.suffix == ".csv":
        # CSV pipeline
        X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013_csv(data_path, config)
        class_weights = compute_balanced_class_weights(y_train)

        # Stage 1: Train classification head
        logger.info("=== Stage 1: Training Classification Head (%d Epochs) ===", epochs)
        history_stage1 = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        # Stage 2: Fine-tune backbone
        if fine_tune_epochs > 0:
            logger.info("=== Stage 2: Fine-Tuning Backbone (%d Epochs) ===", fine_tune_epochs)
            model = unfreeze_model_for_fine_tuning(model, fine_tune_from_layer=config.fine_tune_at)
            model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=fine_tune_epochs,
                batch_size=batch_size,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )

        # Evaluate on Test Split
        evaluate_and_save_metrics(model, X_test, y_test)

    elif data_path.is_dir():
        # Directory-based pipeline (train/val/test folders)
        train_gen, val_gen, test_gen = create_data_generators(
            train_dir=data_path / "train",
            val_dir=data_path / "val",
            test_dir=data_path / "test" if (data_path / "test").exists() else None,
            config=config
        )

        # Compute class weights from generator classes
        y_train_indices = train_gen.classes
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train_indices),
            y=y_train_indices
        )
        class_weights = {int(cls): float(weight) for cls, weight in zip(np.unique(y_train_indices), weights)}

        logger.info("=== Stage 1: Training Classification Head ===")
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        if fine_tune_epochs > 0:
            logger.info("=== Stage 2: Fine-Tuning Backbone ===")
            model = unfreeze_model_for_fine_tuning(model, fine_tune_from_layer=config.fine_tune_at)
            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=fine_tune_epochs,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )

        if test_gen is not None:
            evaluate_generator(model, test_gen)

    # Save final model weights and emotion labels JSON
    model.save(str(config.model_save_path))
    model.save_weights(str(config.weights_save_path))
    save_json(EMOTIONS, config.labels_save_path)
    logger.info("Successfully completed training and serialized model artifacts.")
    return model


def evaluate_and_save_metrics(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """Evaluates test set and generates confusion matrix + classification report."""
    logger.info("Running evaluation on test partition (%d samples)...", len(X_test))
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification Report
    report = classification_report(
        y_true,
        y_pred,
        target_names=EMOTIONS,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=EMOTIONS,
        zero_division=0
    )
    logger.info("\nClassification Report:\n%s", report_text)

    # Save reports
    save_json(report, ASSETS_DIR / "classification_report.json")
    with open(ASSETS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, EMOTIONS, save_path=ASSETS_DIR / "confusion_matrix.png")

    return report


def evaluate_generator(model: tf.keras.Model, test_gen: ImageDataGenerator) -> Dict:
    """Evaluates generator-based test partition."""
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report_text = classification_report(y_true, y_pred, target_names=EMOTIONS, zero_division=0)
    logger.info("\nClassification Report:\n%s", report_text)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, EMOTIONS, save_path=ASSETS_DIR / "confusion_matrix.png")
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emotion Recognition Model on FER-2013")
    parser.add_argument("--data", type=str, default="data/fer2013.csv", help="Path to fer2013.csv or dataset folder")
    parser.add_argument("--epochs", type=int, default=30, help="Initial training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--fine_tune_epochs", type=int, default=10, help="Fine-tuning epochs")
    args = parser.parse_args()

    train_emotion_model(
        data_source=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fine_tune_epochs=args.fine_tune_epochs
    )

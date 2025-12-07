# python/train_and_compress.py

import os
from pathlib import Path
import numpy as np
import tensorflow as tf

from utils.dataset_loader import load_image_folder_dataset
from early_exit_tuner import compute_conf_threshold
from convert_tflite_to_header import convert_tflite_to_header

# Try to import pruning library
try:
    import tensorflow_model_optimization as tfmot
    HAS_TFMOT = True
except ImportError:
    HAS_TFMOT = False
    print("[WARN] tensorflow-model-optimization not installed. Pruning disabled.")

# ------------------ PATHS ------------------ #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"          # you create this
EXPORT_DIR = PROJECT_ROOT / "models_export"
DOCS_DIR = PROJECT_ROOT / "docs"

IMG_SIZE = (64, 64)
GRAYSCALE = True

EPOCHS_FULL = 10
EPOCHS_EXIT1 = 5
EPOCHS_PRUNE_FINETUNE = 3
FINAL_SPARSITY = 0.5  # 50% pruning target


# -------------- UTIL / HELPERS ------------- #

def ensure_dirs():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def build_models(input_shape, num_classes):
    """
    Build two models sharing basic structure:
      - model_exit1: shallower early-exit head
      - model_full: deeper final classifier
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 2
    x2 = tf.keras.layers.SeparableConv2D(32, 3, padding="same", activation="relu")(x)
    x2 = tf.keras.layers.MaxPooling2D()(x2)

    # Early exit head after Block 2
    e1 = tf.keras.layers.GlobalAveragePooling2D()(x2)
    e1 = tf.keras.layers.Dense(32, activation="relu")(e1)
    e1_out = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="exit1_output"
    )(e1)

    # Deeper blocks
    x3 = tf.keras.layers.SeparableConv2D(64, 3, padding="same", activation="relu")(x2)
    x3 = tf.keras.layers.MaxPooling2D()(x3)
    x3 = tf.keras.layers.SeparableConv2D(64, 3, padding="same", activation="relu")(x3)
    x3 = tf.keras.layers.GlobalAveragePooling2D()(x3)
    x3 = tf.keras.layers.Dense(64, activation="relu")(x3)
    full_out = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="full_output"
    )(x3)

    model_exit1 = tf.keras.Model(inputs=inputs, outputs=e1_out, name="model_exit1")
    model_full = tf.keras.Model(inputs=inputs, outputs=full_out, name="model_full")

    return model_exit1, model_full


def compile_and_train(model, x_train, y_train, x_val, y_val,
                      epochs, lr, phase_name, csv_logger_path):
    """
    Compile and train a model, logging metrics to CSV.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = []
    # append to single CSV log file across phases
    csv_logger = tf.keras.callbacks.CSVLogger(
        csv_logger_path,
        append=True,
    )
    callbacks.append(csv_logger)

    print(f"[TRAIN] Phase={phase_name}, epochs={epochs}")

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    val_acc = history.history["val_accuracy"][-1]
    return float(val_acc)


def apply_pruning(model, final_sparsity=0.5):
    """
    Wrap a model with pruning wrappers (tfmot).
    """
    if not HAS_TFMOT:
        print("[WARN] Skipping pruning; TFMOT not available.")
        return model

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=final_sparsity,
        begin_step=0,
        end_step=1000,
    )
    pruned_model = prune_low_magnitude(
        model,
        pruning_schedule=pruning_schedule,
    )
    print("[INFO] Pruning wrappers applied.")
    return pruned_model


def strip_pruning(model):
    if not HAS_TFMOT:
        return model
    print("[INFO] Stripping pruning for export.")
    return tfmot.sparsity.keras.strip_pruning(model)


def representative_dataset_gen(x_train, num_samples=200):
    def gen():
        n = x_train.shape[0]
        for i in range(min(num_samples, n)):
            idx = np.random.randint(0, n)
            sample = x_train[idx : idx + 1].astype("float32")
            yield [sample]
    return gen


def convert_to_int8_tflite(model, x_train, out_path: Path):
    """
    Convert Keras model -> INT8 TFLite and save.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(x_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print(f"[OK] Saved INT8 TFLite model: {out_path}")
    return out_path


# ------------------ MAIN PIPELINE ------------------ #

def main():
    ensure_dirs()

    # ---- 1) Load dataset ---- #
    if not DATASET_DIR.exists():
        raise RuntimeError(
            f"Dataset directory does not exist: {DATASET_DIR}\n"
            "Create ../dataset/class_x/ folders with images first."
        )

    (x_train, y_train), (x_val, y_val), num_classes, class_names = \
        load_image_folder_dataset(
            str(DATASET_DIR),
            img_size=IMG_SIZE,
            grayscale=GRAYSCALE,
            test_split=0.2,
        )

    input_shape = x_train.shape[1:]
    print("[INFO] Dataset loaded.")
    print("  Train:", x_train.shape, " Val:", x_val.shape)
    print("  Classes:", class_names)

    # Training log CSV
    csv_log_path = DOCS_DIR / "training_log.csv"
    if csv_log_path.exists():
        csv_log_path.unlink()  # start fresh

    # ---- 2) Build models ---- #
    model_exit1, model_full = build_models(input_shape, num_classes)

    # ---- 3) Train full model ---- #
    full_acc = compile_and_train(
        model_full,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=EPOCHS_FULL,
        lr=1e-3,
        phase_name="full_baseline",
        csv_logger_path=str(csv_log_path),
    )

    # ---- 4) Train exit1 model ---- #
    exit1_acc = compile_and_train(
        model_exit1,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=EPOCHS_EXIT1,
        lr=5e-4,
        phase_name="exit1_baseline",
        csv_logger_path=str(csv_log_path),
    )

    print(f"[RESULT] Baseline ValAcc -> full={full_acc:.3f}, exit1={exit1_acc:.3f}")

    # ---- 5) Tune early-exit threshold ---- #
    th = compute_conf_threshold(model_exit1, x_val, y_val, target_accuracy=0.95)
    print(f"[INFO] Suggested EARLY_EXIT_THRESHOLD = {th:.3f}")

    (EXPORT_DIR / "early_exit_threshold.txt").write_text(f"{th:.5f}")

    # ---- 6) Apply pruning + fine-tune ---- #
    if HAS_TFMOT:
        print("[STEP] Apply pruning to exit1 model...")
        pruned_exit1 = apply_pruning(model_exit1, final_sparsity=FINAL_SPARSITY)
        pruned_exit1.compile(
            optimizer=tf.keras.optimizers.Adam(5e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        pruned_exit1.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS_PRUNE_FINETUNE,
            batch_size=64,
            verbose=1,
            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
        )
        pruned_exit1 = strip_pruning(pruned_exit1)
    else:
        pruned_exit1 = model_exit1

    if HAS_TFMOT:
        print("[STEP] Apply pruning to full model...")
        pruned_full = apply_pruning(model_full, final_sparsity=FINAL_SPARSITY)
        pruned_full.compile(
            optimizer=tf.keras.optimizers.Adam(5e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        pruned_full.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS_PRUNE_FINETUNE,
            batch_size=64,
            verbose=1,
            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
        )
        pruned_full = strip_pruning(pruned_full)
    else:
        pruned_full = model_full

    # ---- 7) Convert to INT8 TFLite ---- #
    exit1_tflite_path = EXPORT_DIR / "model_exit1_int8.tflite"
    full_tflite_path = EXPORT_DIR / "model_full_int8.tflite"

    convert_to_int8_tflite(pruned_exit1, x_train, exit1_tflite_path)
    convert_to_int8_tflite(pruned_full, x_train, full_tflite_path)

    # ---- 8) Convert TFLite -> .h for ESP32 ---- #
    exit1_h = EXPORT_DIR / "model_exit1_int8.h"
    full_h = EXPORT_DIR / "model_full_int8.h"

    convert_tflite_to_header(
        str(exit1_tflite_path),
        str(exit1_h),
        "g_model_exit1_int8",
    )
    convert_tflite_to_header(
        str(full_tflite_path),
        str(full_h),
        "g_model_full_int8",
    )

    # ---- 9) Log model stats ---- #
    info_path = EXPORT_DIR / "model_info.txt"
    with info_path.open("w") as f:
        f.write("Classes: %s\n" % ", ".join(class_names))
        f.write("Baseline ValAcc Full: %.4f\n" % full_acc)
        f.write("Baseline ValAcc Exit1: %.4f\n" % exit1_acc)
        f.write("Early Exit Threshold: %.5f\n" % th)
        if exit1_tflite_path.exists():
            f.write("Exit1 TFLite size (bytes): %d\n" % exit1_tflite_path.stat().st_size)
        if full_tflite_path.exists():
            f.write("Full TFLite size (bytes): %d\n" % full_tflite_path.stat().st_size)

    print("[DONE] Training + compression + export finished.")
    print("  - Models + headers in:", EXPORT_DIR)
    print("  - Training log CSV in:", DOCS_DIR / 'training_log.csv')
    print("  - Info file:", info_path)


if __name__ == "__main__":
    main()

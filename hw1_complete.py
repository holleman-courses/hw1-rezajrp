#!/usr/bin/env python
# hw1_complete.py  (TensorFlow 2.x + tf.keras / Keras 2.x compatible)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

keras = tf.keras
layers = tf.keras.layers
Input = tf.keras.Input
Sequential = tf.keras.Sequential

tf.config.optimizer.set_jit(False)

import numpy as np

print(f"TensorFlow Version: {tf.__version__}")
try:
    print(f"Keras Version: {tf.keras.version()}")
except Exception:
    print("Keras Version: (tf.keras bundled with TF)")




def build_model1():
    # ['Flatten', 'Dense', 'Dense', 'Dense', 'Dense']
    model = Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(128, activation="leaky_relu"),
        layers.Dense(128, activation="leaky_relu"),
        layers.Dense(128, activation="leaky_relu"),
        layers.Dense(10)  # logits
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_model2():

    model = Sequential([
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same",
                      activation="relu", input_shape=(32, 32, 3)),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same",
                      activation="relu"),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(10)  # logits
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_model3():

    model = Sequential([
        layers.SeparableConv2D(32, (3, 3), strides=(2, 2), padding="same",
                               activation="relu", input_shape=(32, 32, 3)),
        layers.BatchNormalization(),

        layers.SeparableConv2D(64, (3, 3), strides=(2, 2), padding="same",
                               activation="relu"),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(10)  # logits
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_model50k():
    # <= 50k params, should reach >= 0.60 test acc with decent training
    model = Sequential([
        layers.Input(shape=(32, 32, 3)),

        layers.SeparableConv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.SeparableConv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.SeparableConv2D(96, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.SeparableConv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.SeparableConv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Dropout(0.25),
        layers.GlobalAveragePooling2D(),

        layers.Dense(10)  # logits
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


# -------------------------------------------------------
# Main: training / saving (NOT executed during autograding)
# -------------------------------------------------------
if __name__ == "__main__":

    # -----------------------------
    # Load CIFAR-10 and split train/val
    # -----------------------------
    (train_images_full, train_labels_full), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    train_labels_full = train_labels_full.squeeze()
    test_labels = test_labels.squeeze()

    train_images_full = train_images_full.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    val_size = 5000
    val_images = train_images_full[-val_size:]
    val_labels = train_labels_full[-val_size:]
    train_images = train_images_full[:-val_size]
    train_labels = train_labels_full[:-val_size]

    print("Shapes:")
    print("  train_images:", train_images.shape, "train_labels:", train_labels.shape)
    print("  val_images:", val_images.shape, "val_labels:", val_labels.shape)
    print("  test_images:", test_images.shape, "test_labels:", test_labels.shape)

    # -----------------------------
    # Helper to train-or-load
    # -----------------------------
    def train_or_load(model, path, epochs=30, batch_size=64):
        if os.path.exists(path):
            print(f"Loading existing model from {path} ...")
            return keras.models.load_model(path)
        print(f"Training model and saving to {path} ...")
        model.fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2
        )
        model.save(path)  # .h5 caching to avoid format issues
        return model

    # -----------------------------
    # Model 1
    # -----------------------------
    model1 = build_model1()
    model1.summary()
    model1 = train_or_load(model1, "model1_trained.h5", epochs=30)

    tr_loss, tr_acc = model1.evaluate(train_images, train_labels, verbose=0)
    va_loss, va_acc = model1.evaluate(val_images, val_labels, verbose=0)
    te_loss, te_acc = model1.evaluate(test_images, test_labels, verbose=0)
    print("\nModel1 Final Accuracy:")
    print(f"  Train acc: {tr_acc:.4f}")
    print(f"  Val   acc: {va_acc:.4f}")
    print(f"  Test  acc: {te_acc:.4f}")

    # -----------------------------
    # Model 2
    # -----------------------------
    model2 = build_model2()
    model2.summary()
    model2 = train_or_load(model2, "model2_trained.h5", epochs=30)

    tr_loss2, tr_acc2 = model2.evaluate(train_images, train_labels, verbose=0)
    va_loss2, va_acc2 = model2.evaluate(val_images, val_labels, verbose=0)
    te_loss2, te_acc2 = model2.evaluate(test_images, test_labels, verbose=0)
    print("\nModel2 Final Accuracy:")
    print(f"  Train acc: {tr_acc2:.4f}")
    print(f"  Val   acc: {va_acc2:.4f}")
    print(f"  Test  acc: {te_acc2:.4f}")

    # Custom image prediction (optional)
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    img_path = "./cat.jpg"
    if os.path.exists(img_path):
        test_img = np.array(keras.utils.load_img(
            img_path,
            color_mode="rgb",
            target_size=(32, 32)
        )).astype("float32") / 255.0
        test_img = np.expand_dims(test_img, axis=0)
        logits = model2.predict(test_img, verbose=0)
        pred_class = int(np.argmax(logits, axis=1)[0])
        print(f"\nCustom image prediction for {img_path}:")
        print("  Predicted:", class_names[pred_class])
    else:
        print(f"\nCustom image not found: {img_path}")

    # -----------------------------
    # Model 3
    # -----------------------------
    model3 = build_model3()
    model3.summary()
    model3 = train_or_load(model3, "model3_trained.h5", epochs=30)

    tr_loss3, tr_acc3 = model3.evaluate(train_images, train_labels, verbose=0)
    va_loss3, va_acc3 = model3.evaluate(val_images, val_labels, verbose=0)
    te_loss3, te_acc3 = model3.evaluate(test_images, test_labels, verbose=0)
    print("\nModel3 Final Accuracy:")
    print(f"  Train acc: {tr_acc3:.4f}")
    print(f"  Val   acc: {va_acc3:.4f}")
    print(f"  Test  acc: {te_acc3:.4f}")

    # -----------------------------
    # Best <= 50k params model (must save as best_model.h5 for autograder)
    # -----------------------------
    best_path = "best_model.h5"
    if os.path.exists(best_path):
        print(f"\nLoading existing best model from {best_path} ...")
        best_model = keras.models.load_model(best_path)
    else:
        best_model = build_model50k()
        best_model.summary()
        print("Best model param count:", best_model.count_params())

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=6,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
                verbose=1
            )
        ]

        print("\nTraining best model (<=50k params) ...")
        best_model.fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            epochs=40,
            batch_size=64,
            verbose=2,
            callbacks=callbacks
        )

        print(f"\nSaving best model to {best_path} ...")
        best_model.save(best_path)

    loss_t, acc_t = best_model.evaluate(test_images, test_labels, verbose=0)
    print(f"\nBest model test accuracy: {acc_t:.4f}")
    print(f"Best model param count: {best_model.count_params()}")

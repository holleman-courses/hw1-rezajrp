#!/usr/bin/env python

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential
tf.config.optimizer.set_jit(False)

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


## 

def build_model1():
    # Fully-connected model:
    # Layers: Flatten, Dense, Dense, Dense, Dense
    model = keras.Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(128, activation=keras.layers.LeakyReLU(negative_slope=0.1)),
        layers.Dense(128, activation=keras.layers.LeakyReLU(negative_slope=0.1)),
        layers.Dense(128, activation=keras.layers.LeakyReLU(negative_slope=0.1)),
        layers.Dense(10)  # logits
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model



def build_model2():
    model = keras.Sequential([
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
        metrics=["accuracy"]
    )
    return model

def build_model3():
    model = keras.Sequential([
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
        metrics=["accuracy"]
    )
    return model



def build_model50k():
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),

        # Light augmentation (only active during training)
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.1, 0.1),

        # Block 1 (32x32 -> 32x32)
        layers.SeparableConv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        # Block 2 (32x32 -> 16x16)
        layers.SeparableConv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=2),

        # Block 3 (16x16 -> 8x8)
        layers.SeparableConv2D(96, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=2),

        # Block 4 (8x8 -> 8x8)
        layers.SeparableConv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        # Block 5 (8x8 -> 8x8)
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
        metrics=["accuracy"]
    )
    return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

    # ----------------------------------------
    # Load CIFAR-10 and create train/val/test
    # ----------------------------------------
    (train_images_full, train_labels_full), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    # Flatten labels from shape (N,1) -> (N,)
    train_labels_full = train_labels_full.squeeze()
    test_labels = test_labels.squeeze()

    # Normalize images to [0,1] float32
    train_images_full = train_images_full.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Split training into train + validation
    # (simple deterministic split: last 5000 for val)
    val_size = 5000
    val_images = train_images_full[-val_size:]
    val_labels = train_labels_full[-val_size:]
    train_images = train_images_full[:-val_size]
    train_labels = train_labels_full[:-val_size]

    print("Shapes:")
    print("  train_images:", train_images.shape, "train_labels:", train_labels.shape)
    print("  val_images:", val_images.shape, "val_labels:", val_labels.shape)
    print("  test_images:", test_images.shape, "test_labels:", test_labels.shape)

    # ----------------------------------------
    # Build + train model1
    # ----------------------------------------
    model1_path = "model1_trained.keras" 
    model1 = build_model1()
    model1.summary()

    # history1 = model1.fit(
    #     train_images, train_labels,
    #     validation_data=(val_images, val_labels),
    #     epochs=30,
    #     batch_size=64,
    #     verbose=2
    # )

    
    if os.path.exists(model1_path):
        print(f"Loading existing model1 from {model1_path} ...")
        model1 = keras.models.load_model(model1_path)
    else:
        print("Training model1 ...")
        history1 = model1.fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            epochs=30,
            batch_size=64,
            verbose=2
        )
        print(f"Saving model1 to {model1_path} ...")
        model1.save(model1_path)

    # Final accuracies
    train_loss, train_acc = model1.evaluate(train_images, train_labels, verbose=0)
    val_loss, val_acc = model1.evaluate(val_images, val_labels, verbose=0)
    test_loss, test_acc = model1.evaluate(test_images, test_labels, verbose=0)

    print(f"\nModel1 Final Accuracy:")
    print(f"  Train acc: {train_acc:.4f}")
    print(f"  Val   acc: {val_acc:.4f}")
    print(f"  Test  acc: {test_acc:.4f}")



    # ----------------------------------------
    # Build + train model2
    # ----------------------------------------
    model2_path = "model2_trained.keras"
    model2 = build_model2()
    model2.summary()

    if os.path.exists(model2_path):
        print(f"Loading existing model2 from {model2_path} ...")
        model2 = keras.models.load_model(model2_path)
    else:
        print("Training model2 ...")
        history2 = model2.fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            epochs=30,
            batch_size=64,
            verbose=2
        )
        print(f"Saving model2 to {model2_path} ...")
        model2.save(model2_path)

    train_loss2, train_acc2 = model2.evaluate(train_images, train_labels, verbose=0)
    val_loss2, val_acc2 = model2.evaluate(val_images, val_labels, verbose=0)
    test_loss2, test_acc2 = model2.evaluate(test_images, test_labels, verbose=0)

    print(f"\nModel2 Final Accuracy:")
    print(f"  Train acc: {train_acc2:.4f}")
    print(f"  Val   acc: {val_acc2:.4f}")
    print(f"  Test  acc: {test_acc2:.4f}")


    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    # Example: if your image is named test_image_cat.jpg
    img_path = "./cat.jpg"  # <-- change to your actual filename

    if os.path.exists(img_path):
        test_img = np.array(keras.utils.load_img(
            img_path,
            color_mode='rgb',
            target_size=(32, 32)
        )).astype("float32") / 255.0

        test_img = np.expand_dims(test_img, axis=0)  # shape (1,32,32,3)

        logits = model2.predict(test_img, verbose=0)
        pred_class = int(np.argmax(logits, axis=1)[0])

        print(f"\nCustom image prediction for {img_path}:")
        print("  Predicted:", class_names[pred_class])
    else:
        print(f"\nCustom image not found: {img_path}")
        print("Place a 32x32 (or any size; it will be resized) jpg/png in the repo with the correct name.")




    ### Repeat for model 3 and your best sub-50k params model


    # ----------------------------------------
    # Build + train model3 (depthwise-separable)
    # ----------------------------------------
    model3_path = "model3_trained.keras"
    model3 = build_model3()
    model3.summary()

    if os.path.exists(model3_path):
        print(f"Loading existing model3 from {model3_path} ...")
        model3 = keras.models.load_model(model3_path)
    else:
        print("Training model3 ...")
        history3 = model3.fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            epochs=30,
            batch_size=64,
            verbose=2
        )
        print(f"Saving model3 to {model3_path} ...")
        model3.save(model3_path)

    train_loss3, train_acc3 = model3.evaluate(train_images, train_labels, verbose=0)
    val_loss3, val_acc3 = model3.evaluate(val_images, val_labels, verbose=0)
    test_loss3, test_acc3 = model3.evaluate(test_images, test_labels, verbose=0)

    print(f"\nModel3 Final Accuracy:")
    print(f"  Train acc: {train_acc3:.4f}")
    print(f"  Val   acc: {val_acc3:.4f}")
    print(f"  Test  acc: {test_acc3:.4f}")



    # ----------------------------------------
    # Build + train best <=50k params model, save as best_model.h5
    # ----------------------------------------
    best_path = "best_model.h5"

    if os.path.exists(best_path):
        print(f"\nLoading existing best model from {best_path} ...")
        best_model = keras.models.load_model(best_path)
    else:
        best_model = build_model50k()
        best_model.summary()
        print("\nBest model params:", best_model.count_params())

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
            epochs=40,              # early stopping will usually stop earlier
            batch_size=64,
            verbose=2,
            callbacks=callbacks
        )

        print(f"\nSaving best model to {best_path} ...")
        best_model.save(best_path)

    # Quick report (this is what the autograder effectively checks)
    loss_t, acc_t = best_model.evaluate(test_images, test_labels, verbose=0)
    print(f"\nBest model test accuracy: {acc_t:.4f}")
    print(f"Best model param count: {best_model.count_params()}")
  
  

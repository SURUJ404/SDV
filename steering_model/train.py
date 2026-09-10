"""
Trains the CNN+LSTM steering model on driving_log.csv from the Udacity
simulator's Training Mode recordings.

Usage:
    python train.py --log_csv Data/driving_log.csv --epochs 30 --batch_size 32
"""

import argparse

import tensorflow as tf
from sklearn.model_selection import train_test_split

from cnn_lstm_model import build_cnn_lstm
from data_generator import load_log, SteeringSequenceGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_csv", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--camera_correction", type=float, default=0.2)
    parser.add_argument("--out_model", default="cnn_lstm_steering.h5")
    args = parser.parse_args()

    df = load_log(args.log_csv)
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, shuffle=False)
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    train_gen = SteeringSequenceGenerator(
        train_df, batch_size=args.batch_size, augment_data=True,
        camera_correction=args.camera_correction,
    )
    # No augmentation or multi-camera sampling on validation — we want an
    # honest read on true center-camera driving performance.
    val_gen = SteeringSequenceGenerator(
        val_df, batch_size=args.batch_size, augment_data=False,
    )

    model = build_cnn_lstm()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse",
        metrics=["mae"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.out_model, save_best_only=True, monitor="val_loss"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.out_model)
    print(f"Saved model to {args.out_model}")


if __name__ == "__main__":
    main()

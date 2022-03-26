import argparse
import os

import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

tf.random.set_seed(11)


def create_train_model(x_train: pd.DataFrame, y_train: pd.Series, num_of_epochs: int, batch_size: int, learning_rate: float):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(32, input_dim=x_train.shape[1], activation=tf.nn.relu),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(1),
        ]
    )
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    
    logger.info("Model compiled. Starting training...")

    model.fit(x_train, y_train, epochs=num_of_epochs, batch_size=batch_size, verbose=2)
    logger.info("Model training completed.")
    return model


def _load_training_data(base_dir):
    logger.info("Loading training data...")
    train_df = pd.read_parquet(os.path.join(base_dir, "train.parquet"))
    logger.info("Training data loaded")
    return train_df.drop(columns=['Load']), train_df.Load


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--training", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))

    # Mandatory hyperparameters
    parser.add_argument('--num_of_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--version_number', type=str)

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    train_data, train_labels = _load_training_data(args.training)
    
    ff_model = create_train_model(
        x_train=train_data,
        y_train=train_labels,
        num_of_epochs=args.num_of_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    version_number = args.version_number
    
    model_destination_filepath = os.path.join(args.model_dir, version_number)
    logger.info(f"Saving model to {model_destination_filepath}")
    ff_model.save(model_destination_filepath)

"""
Implementation of the model.

Written by Tanmay Patil
"""
from tensorflow.keras import models, layers, optimizers, callbacks

def create_model(loss: str = "mae", lr: float = 0.0001) -> models.Sequential:
    """Create a Tensorflow Keras model with LSTM and Conv1D layers with mae and lr of 0.0001 as default parameters."""
    model = models.Sequential()
    model.add(layers.Conv1D(filters=128, kernel_size=2, activation="relu", input_shape=(30,1)))
    model.add(layers.Conv1D(filters=128, kernel_size=2, activation="relu"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=256, kernel_size=2, activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.RepeatVector(30))
    model.add(layers.LSTM(units=100, return_sequences=True, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(units=100, return_sequences=True, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Bidirectional(layers.LSTM(units=128, activation="relu")))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(loss=loss, optimizer=optimizers.Adam(lr=lr))

    return model

def train_model(model, X_train, y_train, epochs: int = 200):
    """Training the model with early stopping and model checkpoint."""
    early_stopping = callbacks.EarlyStopping(monitor="loss", mode="min", patience=5, restore_best_weights=True)
    save_model_checkpoint = callbacks.ModelCheckpoint(filepath="saved_model/model_v1.h5", monitor="loss", save_best_only=True)
    return model.fit(X_train, y_train, epochs=epochs, callbacks=[early_stopping, save_model_checkpoint])

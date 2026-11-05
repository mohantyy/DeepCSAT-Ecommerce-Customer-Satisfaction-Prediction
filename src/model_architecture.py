import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class DeepCSAT_ANN:
    """
    3-Layer Regularized Artificial Neural Network for DeepCSAT Customer Satisfaction Prediction.
    Architecture:
    - Layer 1: Dense (128) -> BatchNormalization -> ReLU -> Dropout
    - Layer 2: Dense (64) -> BatchNormalization -> ReLU -> Dropout
    - Layer 3: Dense (32) -> BatchNormalization -> ReLU -> Dropout
    - Output Layer: Dense (Softmax activation)
    - Optimizer: Adam
    - Callback: EarlyStopping
    """
    def __init__(self, input_dim, num_classes=5, learning_rate=0.001, l2_reg=0.01, dropout_rate=0.2):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            # Layer 1: Dense 128
            Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg), input_shape=(self.input_dim,)),
            BatchNormalization(),
            ReLU(),
            Dropout(self.dropout_rate),

            # Layer 2: Dense 64
            Dense(64, kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            ReLU(),
            Dropout(self.dropout_rate),

            # Layer 3: Dense 32
            Dense(32, kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            ReLU(),
            Dropout(self.dropout_rate),

            # Output Layer: Softmax
            Dense(self.num_classes, activation='softmax')
        ])

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def get_early_stopping(self, monitor='val_loss', patience=10, restore_best_weights=True):
        return EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=restore_best_weights
        )

def build_deepcsat_ann(input_dim, num_classes=5, learning_rate=0.001, l2_reg=0.01, dropout_rate=0.2):
    ann = DeepCSAT_ANN(input_dim=input_dim, num_classes=num_classes, learning_rate=learning_rate, l2_reg=l2_reg, dropout_rate=dropout_rate)
    return ann.model

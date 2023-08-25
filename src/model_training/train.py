import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.legacy.Adam(), # Using legacy optimizer
                  metrics=['accuracy'])

    return model

def train_model(model, train_data, train_labels, validation_data, validation_labels, epochs, model_path):
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # Implement early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    callbacks_list = [checkpoint, early_stopping]

    model.fit(train_data, train_labels,
              batch_size=250,
              epochs=epochs,
              verbose=1,
              validation_data=(validation_data, validation_labels),
              callbacks=callbacks_list)

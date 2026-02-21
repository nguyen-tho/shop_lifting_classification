import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def build_model(input_shape):
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
    base_model.trainable = False  # freeze base model layers

    # Create Functional model
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)

    # Add L2 regularization to the Dense layer
    x = layers.Dense(2, kernel_regularizer=regularizers.l2(0.001))(x)

    # Separate activation of output layer so we can output float32 activations
    outputs = layers.Activation("sigmoid", dtype=tf.float32, name="output")(x)
    model = tf.keras.Model(inputs, outputs) 
    
    return model

def data_augmentation():
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomHeight(0.2),
        layers.RandomWidth(0.2),
    ], name="data_augmentation")
    return data_augmentation

def compile_model(model, loss, metrics, learning_rate=0.001, optimizer='adam'):
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
def summarize_model(model):
    model.summary()
    
def train_model(model, train_data, validation_data, epochs=10, callback=[]):
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callback
    )
    return history
    
def save_model(model, model_path):
    model.save(model_path)
    
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model
def model_predict(model, input_data):
    predictions = model.predict(input_data)
    return predictions

    
       
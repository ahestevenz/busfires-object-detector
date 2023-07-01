from typing import Tuple, Dict, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from loguru import logger as logging


class BushFiresDetector:
    def __init__(self, mobilenet: str, units_dense_layers: List):
        self.mobilenet = mobilenet
        self.units_dense_layers = units_dense_layers
        self.width = 224
        self.height = 224
        self.channels = 3

    def feature_extractor(self,
                          inputs: tf.keras.Input) -> tf.keras.layers:
        if self.mobilenet == 'V3':
            mobilenet_model = tf.keras.applications.MobileNetV3Large(
                input_shape=(self.width, self.height, self.channels), include_top=False, weights='imagenet')
        elif self.mobilenet == 'V2':
            mobilenet_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(self.width, self.height, self.channels), include_top=False, weights='imagenet')
        else:
            logging.error('Mobilenet model was not found!!')
            return
        feature_extractor = mobilenet_model(inputs)
        return feature_extractor

    def dense_layers(self, features: tf.keras.layers) -> tf.keras.layers:
        x = tf.keras.layers.GlobalAveragePooling2D()(features)
        x = tf.keras.layers.Flatten()(x)
        for units in self.units_dense_layers:
            x = tf.keras.layers.Dense(units, activation='relu')(x)
        return x

    def bounding_box_regression(self,
                                x: tf.keras.layers) -> tf.keras.layers:
        bounding_box_regression_output = tf.keras.layers.Dense(
            units='4', name='bounding_box')(x)
        return bounding_box_regression_output

    def final_model(self, inputs: tf.keras.Input) -> tf.keras.Input:
        feature_cnn = self.feature_extractor(inputs)
        last_dense_layer = self.dense_layers(feature_cnn)
        bounding_box_output = self.bounding_box_regression(last_dense_layer)
        model = tf.keras.Model(inputs, bounding_box_output)
        return model

    def define_and_compile_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(self.width, self.height, self.channels))
        model = self.final_model(inputs)
        model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9),
                      loss='mean_squared_error')
        return model

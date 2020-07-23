import keras
from keras.layers import Input, Dense, Conv1D, MaxPool1D, Flatten, Reshape, UpSampling1D
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np
import os
import math
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


class AutoEncoder:
    def __init__(self, encoding_dim):
        self.encoding_dim = encoding_dim
        self.laser_data = np.load('/home/zamzam/ar-tu-do/ros_ws/src/autonomous/reinforcement_learning/laser_scans.npy')
        self.laser_data = self.laser_data[0:20000]
        for i in range(len(self.laser_data)):
            for z in range(len(self.laser_data[0])):
                if math.isinf(self.laser_data[i, z]):
                    self.laser_data[i, z] = 0
        #self.laser_data /= np.max(np.abs(self.laser_data), axis=0)

    def _encoder(self):
        inputs = Input(shape=(180, 1), dtype='float64')  # instantiating a keras tensor with size of input
        first_layer = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs)
        max1 = MaxPool1D(pool_size=2, padding='same')(first_layer)
        second_layer = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(max1)
        max2 = MaxPool1D(pool_size=2, padding='same')(second_layer)
        third_layer = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(max2)
        max3 = MaxPool1D(pool_size=3, padding='same')(third_layer)
        flat = Flatten()(max3)
        fc1 = Dense(64, activation='relu')(flat)
        encoded = Dense(10, activation='relu')(fc1)

        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(10, ), dtype='float64')
        fc1 = Dense(64, activation='relu')(inputs)
        fc2 = Dense(1920, activation='relu')(fc1)
        reshaped = Reshape((15, 128))(fc2)
        up1 = UpSampling1D(3)(reshaped)
        conv1 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(up1)
        up2  = UpSampling1D(2)(conv1)
        conv2 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(up2)
        up3 = UpSampling1D(2)(conv2)
        decoded = Conv1D(filters=1, kernel_size=5, padding='same', activation='relu')(up3)
        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        print("Encoder model:")
        ec.summary()
        dc = self._decoder()
        print("Decoder model")
        dc.summary()

        inputs = Input(shape=(180,1))
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model
        return model

    def fit(self, batch_size, epochs):
        self.model.compile(optimizer='Adam', loss=root_mean_squared_error)
        log_dir = './log/'
        tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(self.laser_data, self.laser_data,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[tbCallBack])

    def save(self):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
        else:
            self.encoder.save(r'./weights/encoder_weights.h5')
            self.decoder.save(r'./weights/decoder_weights.h5')
            self.model.save(r'./weights/ae_weights.h5')


if __name__ == '__main__':
    ae = AutoEncoder(encoding_dim=10)
    ae.encoder_decoder()
    ae.fit(batch_size=32, epochs=100)
    ae.save()
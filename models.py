import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers.schedules import ExponentialDecay

class DiffusionModel:

    def __init__(self, args, dir_results):
        self.loss = MeanSquaredError()
        self.optimizer = Adam(learning_rate=ExponentialDecay(args["lr"], decay_steps=100000,decay_rate=0.96))
        self.tsteps = args["tsteps"]
        global batch_size 
        batch_size = args["batch_size"]
        self.kernel_size = args["kernel_size"]
        self.model = self.basic_mlp_version_bn(args["dataset_dim"])
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
    
    def basic_mlp_version_bn(self, input_dim):
        x = input_lay = Input(shape=(input_dim,), name="input_1")
        x_ts = x_ts_in = Input(shape=(1, ), name="input_2")
        for i in range(2):
            x = Concatenate(axis=1)([x, x_ts])

            x= Dense(x.shape[1]//2, activation='tanh')(x)
            x = BatchNormalization()(x)
        x= Dense(x.shape[1], activation='tanh')(x)
        x = BatchNormalization()(x)
        x= Dense(x.shape[1], activation='tanh')(x)

        for i in range(1):
            x= Dense(x.shape[1]*2, activation='tanh')(x)
        out = Dense(input_dim, activation='linear')(x)
        model = tf.keras.models.Model([input_lay, x_ts_in], out)
        return model
    

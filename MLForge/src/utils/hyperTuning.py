from hyperopt import hp
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Masking, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Dropout

def create_hyperopt_space(Algorithm):
    
    if Algorithm == "XGB":
        space = {
            "learning_rate": hp.quniform("learning_rate", 0.1, 0.2, 0.001), # hp.quniform("learning_rate", 0.1, 0.6, 0.025),
            "max_depth":  hp.choice("max_depth",np.arange(2, 3)), # A problem with max_depth casted to float instead of int with the hp.quniform method.
            "min_child_weight": hp.quniform("min_child_weight", 1, 30, 1), #hp.quniform("min_child_weight", 1, 20, 1),
            "subsample": hp.quniform("subsample", 0.6, 1, 0.05),
            "min_split_loss": hp.quniform("min_split_loss", 0, 2, 0.1),  # a.k.a. gamma
        }

    elif Algorithm == "DNN":
        space = {
            "learning_rate": hp.quniform("learning_rate", 0.001, 0.01, 0.001),
            "num_units_1": hp.choice("num_units_1", [16, 32, 64, 128, 256]),
            "num_units_2": hp.choice("num_units_2", [16, 32, 64, 128, 256]),
            "num_units_3": hp.choice("num_units_3", [16, 32, 64, 128, 256]),
            "dropout_rate": hp.quniform("dropout_rate", 0.05, 0.5, 0.01),
            "kernel_initializer": hp.choice("kernel_initializer", ["glorot_uniform", "he_uniform", "he_normal"]),
            # "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
            # "num_layers": hp.choice("num_layers", [1, 2, 3, 4]),
            # "activation": hp.choice("activation", ["relu", "tanh", "sigmoid"]),
            # "optimizer": hp.choice("optimizer", ["adam", "sgd", "rmsprop"]),
        }
    else :
        print("Algorithm not supported")
        space = {}    
    return space

def create_DNN_model(params):
    model = Sequential([
        Input(shape=params["input_dim"]),
        Masking(mask_value=-1),
        Dense(params["num_units_1"], kernel_initializer=params["kernel_initializer"], activation='relu'),
        Dropout(params["dropout_rate"]),
        Dense(params['num_units_2'], kernel_initializer=params["kernel_initializer"], activation="relu"),
        Dropout(params['dropout_rate']),
        Dense(params['num_units_3'], kernel_initializer=params["kernel_initializer"], activation="relu"),
        Dense(params["output_dim"], activation="softmax")
    ])
    
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


# def objective():
    
# def DNN_objective(params):
#     model = create_DNN_model(params)
  
#     history = model.fit(x_train, y_train, sample_weight=w_train, validation_data=(x_test, y_test, w_test),
#                         epochs=params["epochs"], batch_size=params['batchsize'], callbacks=[es], verbose=0)
#     val_loss = min(history.history['val_loss'])
#     return val_loss




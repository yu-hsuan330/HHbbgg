from hyperopt import hp
import numpy as np

def create_hyperopt_space(Algorithm):
    
    if Algorithm == "XGB":
        space = {
            "learning_rate": hp.quniform("learning_rate", 0.1, 0.15, 0.001), # hp.quniform("learning_rate", 0.1, 0.6, 0.025),
            "max_depth":  hp.choice("max_depth",np.arange(2, 7)), # A problem with max_depth casted to float instead of int with the hp.quniform method.
            "min_child_weight": hp.quniform("min_child_weight", 1, 30, 1), #hp.quniform("min_child_weight", 1, 20, 1),
            "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
            "min_split_loss": hp.quniform("min_split_loss", 0, 2, 0.1),  # a.k.a. gamma
        }
        
    if Algorithm == "DNN":
        space = {
            "learning_rate": hp.uniform("learning_rate", 0.0001, 0.1),
            "num_layers": hp.choice("num_layers", [1, 2, 3, 4]),
            "num_units": hp.choice("num_units", [32, 64, 128, 256, 512]),
            # "activation": hp.choice("activation", ["relu", "tanh", "sigmoid"]),
            "dropout_rate": hp.uniform("dropout_rate", 0.0, 0.5),
            "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
            # "optimizer": hp.choice("optimizer", ["adam", "sgd", "rmsprop"]),
        }
        
    return space

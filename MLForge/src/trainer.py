import sys, time, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pickle as pkl

import hyperopt as hpt
import xgboost as xgb

# tf.get_logger().setLevel('ERROR')  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Dropout

from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

from src.utils import * # the function details are in the __init__.py file

#// import tensorflow as tf
#// import pandas as pd 
#// from xgboost import XGBClassifier
#// from hyperopt.early_stop import no_progress_loss
#// import cudf
#// from sklearn.preprocessing import MinMaxScaler

#// from src.utils.data_loader import df_from_rootfiles, df_balance_rwt
#// from src.utils.hyperTuning import create_hyperopt_space
#// from src.utils.plotter import prGreen, plot_Features, plot_Features_Scaled, plot_ROC, plot_VarImportance, plot_MVA, plot_confusion_matrix, plot_correlation_matrix, loss_history

# XGB_space = create_hyperopt_space("XGB")
# DNN_space = create_hyperopt_space("DNN")

def XGB_objective(params): 
    
    model = xgb.train(
        params=params, 
        dtrain=dtrain, 
        evals=[(dtrain, 'train'), (dtest, 'eval')], 
        early_stopping_rounds=10, 
        num_boost_round=200, 
        verbose_eval=False
    )
    score = model.best_score
    return -score

def create_DNN_model(params):
    model = Sequential()
    model.add(Input(shape=params["input_dim"]))
    model.add(Dense(params['num_units'], activation='relu'))
    for _ in range(params['num_layers'] - 1):
        model.add(Dense(params['num_units'], activation='relu'))
        # model.add(Dense(params['num_units'], activation=params['activation']))

        model.add(Dropout(params['dropout_rate']))
    model.add(Dense(params["output_dim"], activation='softmax'))
    optimizer = Adam(learning_rate=params['learning_rate']) # if params['optimizer'] == 'adam' else (
    #     SGD(learning_rate=params['learning_rate']) if params['optimizer'] == 'sgd' else RMSprop(learning_rate=params['learning_rate'])
    # )
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def DNN_objective(params):
    model = create_DNN_model(params)
  
    history = model.fit(x_train, y_train, sample_weight=w_train, validation_data=(x_test, y_test, w_test),
                        epochs=params["epochs"], batch_size=params['batchsize'], callbacks=[es], verbose=0)
    val_loss = min(history.history['val_loss'])
    return val_loss

def PrepDataset(df, TrainIndices, TestIndices, features, weight):
    X_train = df.loc[TrainIndices, features]
    Y_train = df.loc[TrainIndices, "Category"]
    W_train = df.loc[TrainIndices, weight]

    X_test = df.loc[TestIndices, features]
    Y_test = df.loc[TestIndices, "Category"]
    W_test = df.loc[TestIndices, weight]

    return X_train, Y_train, W_train, X_test, Y_test, W_test

def trainer(Conf):

    #* Read ROOT file to pandas dataset
    df = df_from_rootfiles(Conf.Processes, Conf.Tree, Conf.Branches, Conf.Debug)
    
    # Record category(number) <-> class(name)
    df["Category"] = 0
    for i, k in enumerate(Conf.Classes):
        df.loc[df.Class == k, "Category"] = i

    # Index of dataset -> separate into train and test dataset
    TrainIndices, TestIndices = [], []
    index = df.index
    for myclass in Conf.Classes:
        Indices = index[ df["Class"] == myclass ].values.tolist()
        myclassTrainIndices, myclassTestIndices = train_test_split(Indices, test_size=Conf.TestSize, random_state=Conf.RandomState, shuffle=True) 
        TrainIndices = TrainIndices + myclassTrainIndices
        TestIndices  = TestIndices  + myclassTestIndices
    
    for MVA in Conf.MVAs:

        start_time = time.time()
        prGreen("Start "+MVA["MVAtype"]+":")
        
        # Make directory and copy the training code
        os.makedirs(Conf.OutputDirName + "/" + MVA["MVAtype"], exist_ok=True)
        # if os.path.exists(Conf.OutputDirName+"/"+MVA["MVAtype"]) == 0:
        #     prGreen("Making output directory")
        #     os.system("mkdir -p " + Conf.OutputDirName+"/"+)
        
        #* Add weight to dataset (additional weight & balanced weight)
        # Apply the mass resolution as additional weight (addwei) on the signal samples (Category == 0)
        # nowei: without additional weight; varName: (1 / specific variable) as additional weight
        df["add_weight"] = df["input_weight"].abs()
        if MVA["addwei"] != "nowei":
            df.loc[df["Category"] == 0, "add_weight"] /= df.loc[df["Category"] == 0, MVA["addwei"]]
        plot_Features(MVA, df, "add_weight", Conf)

        # Balance two classes
        weight = "balancedWt"
        if Conf.Debug == True: print( "Balanced reweighting for training sample", flush=True)
        df.loc[TrainIndices, weight] = df_balance_rwt(df.loc[TrainIndices], SumWeightCol="add_weight", NewWeightCol=weight, Classes=Conf.Classes, debug=Conf.Debug)
        if Conf.Debug == True: print( "Balanced reweighting for testing sample" , flush=True)
        df.loc[TestIndices, weight] = df_balance_rwt(df.loc[TestIndices], SumWeightCol="add_weight", NewWeightCol=weight, Classes=Conf.Classes, debug=Conf.Debug)

        #* Prepare the training/testing dataset
        global x_train, y_train, w_train, x_test, y_test, w_test
        x_train, y_train, w_train, x_test, y_test, w_test = PrepDataset(df, TrainIndices, TestIndices, MVA["features"], weight) 
        
        
        if MVA["Algorithm"] == "DNN":
            y_train = to_categorical(y_train, num_classes=len(Conf.Classes))
            y_test = to_categorical(y_test, num_classes=len(Conf.Classes))
            
            #* Scale the input dataset
            # x_train.loc[x_train["pair1_btagPNetB"] < 0, "pair1_btagPNetB"] = -99
            # x_train.loc[x_train["pair2_btagPNetB"] < 0, "pair2_btagPNetB"] = -99
            # x_train.loc[x_train["pair1_btagPNetQvG"] < 0, "pair1_btagPNetQvG"] = -999
            # x_train.loc[x_train["pair2_btagPNetQvG"] < 0, "pair2_btagPNetQvG"] = -999
            
            # x_test.loc[x_test["pair1_btagPNetB"] < 0, "pair1_btagPNetB"] = -99
            # x_test.loc[x_test["pair2_btagPNetB"] < 0, "pair2_btagPNetB"] = -99
            # x_test.loc[x_test["pair1_btagPNetQvG"] < 0, "pair1_btagPNetQvG"] = -999
            # x_test.loc[x_test["pair2_btagPNetQvG"] < 0, "pair2_btagPNetQvG"] = -999
            
            #* Apply the scaler
            try:
                exec("sc = "+MVA["Scaler"]+"()")
                # TODO optimize the scaling
                valid_mask = (x_train != -999) & (x_train != -99)
                x_train[valid_mask] = sc.fit_transform(x_train[valid_mask])
                # x_train[(x_train == -99)] = 0
                x_train[(x_train == -99)] = -1
                x_train[(x_train == -999)] = -1
                
                valid_mask = (x_test != -999) & (x_test != -99)
                x_test[valid_mask] = sc.transform(x_test[valid_mask])
                x_test[(x_test == -99)] = -1
                x_test[(x_test == -999)] = -1
                
                with open((Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+MVA["MVAtype"]+"_"+"scaler.pkl"), 'wb') as f:
                    pkl.dump(sc, f)
            except:
                print("Data is not being scaled! Either no scaling option provided or scaling not found")
            
            #* Set the parameters of early stopping
            global es
            try:
                es = MVA["DNNDict"]["earlyStopping"]
            except:
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                print("No early stopping specified, will use the default one")
            
            # TODO Search hyperparameter  
            if MVA["hyperopt"]:
                space = create_hyperopt_space("DNN")
                search_params = {**MVA["DNNDict"]["ModelParams"], **space}
                print(search_params)
                trials = hpt.Trials()
                best_params = hpt.fmin(fn=DNN_objective, space=search_params, algo=hpt.tpe.suggest, max_evals=4, trials=trials)
                #! best_params = hpt.space_eval(search_params, best)
                print(best_params)
                #! modelDNN = create_DNN_model(best_params); modelDNN.compile()
                modelDNN = create_DNN_model({**MVA["DNNDict"]["ModelParams"], **best_params})
            else:
                modelDNN = MVA["DNNDict"]["model"]
                modelDNN.compile(loss=MVA["DNNDict"]["compile"]["loss"], optimizer=MVA["DNNDict"]["compile"]["optimizer"], metrics=MVA["DNNDict"]["compile"]["metrics"])
            
            batchsize = MVA["DNNDict"]["ModelParams"]["batchsize"]
            train_history = modelDNN.fit(x_train, y_train, epochs=MVA["DNNDict"]["ModelParams"]["epochs"], batch_size=batchsize, validation_data=(x_test, y_test, w_test), callbacks=[es], sample_weight=w_train)
            # train_history = modelDNN.fit(x_train, y_train, epochs=MVA["DNNDict"]["ModelParams"]["epochs"], batch_size=batchsize, validation_data=(x_test, y_test, w_test), verbose=1, callbacks=[es], sample_weight=w_train)
            modelDNN.save(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+MVA["MVAtype"]+"_"+"modelDNN.keras")

            y_train_pred = modelDNN.predict(x_train, batch_size=batchsize)  
            y_test_pred  = modelDNN.predict(x_test, batch_size=batchsize)  

            #* Save loss history
            training_loss = train_history.history['loss']
            testing_loss = train_history.history['val_loss']
            loss_history(MVA, training_loss, testing_loss, Conf)
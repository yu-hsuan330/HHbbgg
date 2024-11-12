# from sklearn.datasets import fetch_openml
import sys, time, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd 
import numpy as np
import pickle as pkl
import hyperopt as hpt
from hyperopt import hp
from hyperopt.early_stop import no_progress_loss

# import cudf
import xgboost as xgb
from xgboost import XGBClassifier

import tensorflow as tf
# tf.get_logger().setLevel('ERROR')  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from Tools.dfUtils import df_from_rootfiles, df_balance_rwt
from Tools.PlotsUtils import prGreen, pltSty, plot_Features, plot_Features2, plot_ROC, plot_VarImportance, plot_MVA, plot_CM, plot_correlation_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

XGB_space = {
    "learning_rate": hp.quniform("learning_rate", 0.1, 0.15, 0.001), # hp.quniform("learning_rate", 0.1, 0.6, 0.025),
    "max_depth":  hp.choice("max_depth",np.arange(2, 7)), # A problem with max_depth casted to float instead of int with the hp.quniform method.
    "min_child_weight": hp.quniform("min_child_weight", 1, 30, 1), #hp.quniform("min_child_weight", 1, 20, 1),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
    "min_split_loss": hp.quniform("min_split_loss", 0, 2, 0.1),  # a.k.a. gamma
}
DNN_space = {
    "learning_rate": hp.uniform("learning_rate", 0.0001, 0.1),
    'num_layers': hp.choice('num_layers', [1, 2, 3, 4]),
    'num_units': hp.choice('num_units', [32, 64, 128, 256, 512]),
    'activation': hp.choice('activation', ['relu', 'tanh', 'sigmoid']),
    'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
    'optimizer': hp.choice('optimizer', ['adam', 'sgd', 'rmsprop']),
}
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
    model.add(Dense(params['num_units'], activation=params['activation'], input_shape=(input_dim)))
    for _ in range(params['num_layers'] - 1):
        model.add(Dense(params['num_units'], activation=params['activation']))
        model.add(Dropout(params['dropout_rate']))
    model.add(Dense(output_dim, activation='softmax'))
    optimizer = Adam(learning_rate=params['learning_rate']) if params['optimizer'] == 'adam' else (
        SGD(learning_rate=params['learning_rate']) if params['optimizer'] == 'sgd' else RMSprop(learning_rate=params['learning_rate'])
    )
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

if __name__ == "__main__":

    # get the information from config file
    TrainConfig = sys.argv[1]
    importConfig=TrainConfig.replace("/", ".")
    exec("import " + importConfig + " as Conf")

    if os.path.exists(Conf.OutputDirName) == 0:
        os.system("mkdir -p " + Conf.OutputDirName+"/{CodeANDConfig,Plots,Minitrees}")
    os.system("cp "+TrainConfig+".py ./"+ Conf.OutputDirName+"/CodeANDConfig/")
    os.system("cp Trainer.py ./"+ Conf.OutputDirName+"/CodeANDConfig/")

    # read ROOT file to pandas dataset
    df = df_from_rootfiles(Conf.Processes, Conf.Tree, Conf.Branches, Conf.Debug)
    # df = df_from_rootfiles_pairing(Conf.Processes, Conf.Tree, Conf.Branches, Conf.Branches_custom, Conf.Debug)
    
    # record category(number) <-> class(name)
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

    df.loc[TrainIndices, "Dataset"] = "Train"
    df.loc[TestIndices, "Dataset"] = "Test"
    df.loc[TrainIndices, "TrainDataset"] = 1
    df.loc[TestIndices, "TrainDataset"] = 0
    
    for MVA in Conf.MVAs:
                
        start_time = time.time()
        prGreen("Start "+MVA["MVAtype"]+":")
        
        # make directory and copy the training code
        if os.path.exists(Conf.OutputDirName+"/"+MVA["MVAtype"]) == 0:
            prGreen("Making output directory")
            os.system("mkdir -p " + Conf.OutputDirName+"/"+MVA["MVAtype"])
        
        # apply the mass resolution as additional weight (addwei) on the signal samples (Category == 0)
        # nowei: without additional weight; varName: (1 / specific variable) as additional weight
        df["add_weight"] = df["input_weight"].abs()
        if MVA["addwei"] != "nowei":
            df.loc[df["Category"] == 0, "add_weight"] /= df.loc[df["Category"] == 0, MVA["addwei"]]
       
        # balance two classes
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
            
            # Scale the input dataset
            try:
                exec("sc = "+MVA["Scaler"]+"()")
                x_train = sc.fit_transform(x_train)
                x_test = sc.transform(x_test)
                
                with open((Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+MVA["MVAtype"]+"_"+"scaler.pkl"), 'wb') as f:
                    pkl.dump(sc, f)
            except:
                print("Data is not being scaled! Either no scaling option provided or scaling not found")
                
            # set the parameters of early stopping
            global es
            try:
                es = MVA["DNNDict"]["earlyStopping"]
            except:
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                print("No early stopping specified, will use the default one")
            
            #TODO Search hyperparameter  
            
            if MVA["hyperopt"]:
                trials = hpt.Trials()
                best_params = hpt.fmin(fn=DNN_objective, space=DNN_space, algo=hpt.tpe.suggest, max_evals=50, trials=trials)
                #! best_params = hpt.space_eval(search_params, best)
                print(best_params)
                #! modelDNN = create_DNN_model(best_params); modelDNN.compile()
            else:
                modelDNN = MVA["DNNDict"]["model"]
                modelDNN.compile(loss=MVA["DNNDict"]["compile"]["loss"], optimizer=MVA["DNNDict"]["compile"]["optimizer"], metrics=MVA["DNNDict"]["compile"]["metrics"])
            
            train_history = modelDNN.fit(x_train, y_train, epochs=MVA["DNNDict"]["epochs"], batch_size=MVA["DNNDict"]["batchsize"], validation_data=(x_test, y_test, w_test), verbose=1, callbacks=[es], sample_weight=w_train)
            modelDNN.save(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+MVA["MVAtype"]+"_"+"modelDNN.keras")

            y_train_pred = modelDNN.predict(x_train, batch_size=MVA["DNNDict"]["batchsize"])  
            y_test_pred  = modelDNN.predict(x_test, batch_size=MVA["DNNDict"]["batchsize"])  

            # print(y_train_pred)
            training_loss = train_history.history['loss']
            test_loss = train_history.history['val_loss']

            # Create count of the number of epochs
            epoch_count = range(1, len(training_loss) + 1)

            # Visualize loss history
            fig, axes = plt.subplots(1, 1, figsize=(6, 6))
            axes.plot(epoch_count, training_loss)
            axes.plot(epoch_count, test_loss)
            axes.legend(['Training Loss', 'Test Loss'])
            # axes.set_xlabel('Epoch')
            # axes.set_ylabel(MVA["Label"]+': Loss')
            pltSty(axes, xName = "Epoch", yName = MVA["Label"]+': Loss')
            plt.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+MVA["MVAtype"]+"_"+"Loss.pdf")
            
        elif MVA["Algorithm"] == "XGB":
            # Move the data to GPU
            global dtrain, dtest
            dtrain = xgb.DMatrix(data=x_train, label=y_train, weight=w_train)
            dtest  = xgb.DMatrix(data=x_test , label=y_test , weight=w_test )
            
            #* Search hyperparameter  
            best_params = {**MVA["ModelParams"], **MVA["HyperParams"]}
            
            if MVA["hyperopt"]:
                search_params = {**MVA["ModelParams"], **XGB_space}
                trials = hpt.Trials()
                best = hpt.fmin(fn=XGB_objective, space=search_params, algo=hpt.tpe.suggest, max_evals=500,
                                early_stop_fn=no_progress_loss(iteration_stop_count=30, percent_increase=0.001), show_progressbar=True)
                best_params = hpt.space_eval(search_params, best)
                print(best_params)

            
            # num_boost_round=100, evals=evals, early_stopping_rounds=10
            evals_result = {}
            bst = xgb.train(
                params=best_params, 
                dtrain=dtrain, 
                evals=[(dtrain, 'train'), (dtest, 'eval')], 
                evals_result=evals_result, 
                early_stopping_rounds=10, 
                num_boost_round=200, 
                verbose_eval=False
            )
            y_train_pred = bst.predict(dtrain, iteration_range=(0, bst.best_iteration + 1))  
            y_test_pred  = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))  
            #TODO ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

            y_train = to_categorical(y_train, num_classes=len(Conf.Classes))
            y_test = to_categorical(y_test, num_classes=len(Conf.Classes))
            # print(bst.feature_names)
            # print("Feature Importance Scores:", bst.get_score(importance_type='weight'))

            # best_score = bst
            # print(evals_result)
            # print(max(evals_result["eval"]["merror"]))
            
            # print(y_train_pred)
            #* Train the Classifier
            # clf = XGBClassifier(**best_params)
            # clf.fit(x_train_gpu, y_train, sample_weight=w_train, verbose=0, eval_set=[(x_train_gpu, y_train), (x_test, y_test)], sample_weight_eval_set=[w_train, w_test])

            #* Save the model
            bst.save_model("{}/{}/{}_best_modelXGB.json".format(Conf.OutputDirName, MVA["MVAtype"], MVA["MVAtype"]))
            bst.dump_model('dump.raw.txt')     

        #* Plot the training results
        plot_Features(MVA, df, "add_weight", Conf)
        plot_Features2(MVA, x_train, y_train, "add_weight", Conf)
        plot_ROC(MVA, y_train, y_test, y_train_pred, y_test_pred, w_train, w_test, Conf)
        # plot_MVA(MVA, y_train[abs(x_train[:,10])>1.566], y_test[abs(x_test[:,10])>1.566], y_train_pred[abs(x_train[:,10])>1.566], y_test_pred[abs(x_test[:,10])>1.566], w_train[abs(x_train[:,10])>1.566], w_test[abs(x_test[:,10])>1.566], Conf, False)   
        plot_MVA(MVA, y_train, y_test, y_train_pred, y_test_pred, w_train, w_test, Conf, False)   
        #Confusion matrix with and without normalization
        plot_CM(MVA, y_test, y_test_pred, Conf)
        plot_CM(MVA, y_test, y_test_pred, Conf, True)
        
        plot_correlation_matrix(MVA, x_train, Conf)
        # plot_VarImportance(MVA, modelDNN, Conf, x_train, x_test) # plot_VarImportance(MVA, cv.best_estimator_, Conf)

        # print(classification_report(y_test, y_test_pred))

        seconds = time.time() - start_time
        print("[INFO] Time Taken: {}".format(time.strftime("%H:%M:%S",time.gmtime(seconds))))
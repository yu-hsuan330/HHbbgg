import sys, time, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pickle as pkl

import hyperopt as hpt
from hyperopt.early_stop import no_progress_loss

import xgboost as xgb

# tf.get_logger().setLevel('ERROR')  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import classification_report

from src.utils import * # the function details are in the __init__.py file

def trainer(Conf, show_progress):

    #* Read ROOT file to pandas dataset
    df = df_from_rootfiles(Conf.Processes, Conf.Tree, Conf.Branches, Conf.Debug)
    # print(df)
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

        prGreen("Start "+MVA["MVAtype"]+":")
        
        start_time = time.time()
        
        # Make directory and copy the training code
        os.makedirs(Conf.OutputDirName + "/" + MVA["MVAtype"], exist_ok=True)
        
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
        
        scaler_classes = {
            "MinMaxScaler": MinMaxScaler,
        }
        
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
                scaler_name = MVA["Scaler"]
                if scaler_name in scaler_classes:
                    sc = scaler_classes[scaler_name]()
                else:
                    raise ValueError(f"Invalid scaler: {scaler_name}")
                    
                #TODO optimize the scaling part
                valid_mask = (x_train != -999) & (x_train != -99)
                x_train[valid_mask] = sc.fit_transform(x_train[valid_mask])
                x_train[~valid_mask] = -1
                
                valid_mask = (x_test != -999) & (x_test != -99)
                x_test[valid_mask] = sc.transform(x_test[valid_mask])
                x_test[~valid_mask] = -1

                with open(Conf.OutputDirName + "/" + MVA["MVAtype"] + "/" + MVA["MVAtype"] + "_scaler.pkl", 'wb') as f:
                    pkl.dump(sc, f)
            
            except Exception as e:
                print("Data is not being scaled! Either no scaling option provided or scaling not found.")
                print("Error:", e)
                
            #* Set the parameters of early stopping
            global es
            try:
                es = MVA["DNNDict"]["earlyStopping"]
            except:
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
                print("No early stopping specified, will use the default one")
            
            #* Search hyperparameter
            if MVA["hyperopt"]:                
                def objective(params):
                    model = create_DNN_model(params)
                    history = model.fit(x_train, y_train, sample_weight=w_train, validation_data=(x_test, y_test, w_test),
                                        epochs=params["epochs"], batch_size=params['batchsize'], callbacks=[es], verbose=0)
                    val_loss = history.history['val_loss'][-1] #TODO Should use : min(history.history['val_loss'])
                    return val_loss

                space = create_hyperopt_space("DNN")
                search_params = {**MVA["DNNDict"]["ModelParams"], **space}
                # trials = hpt.Trials()
                best = hpt.fmin(fn=objective, space=search_params, algo=hpt.tpe.suggest, max_evals=500,
                                       early_stop_fn=no_progress_loss(iteration_stop_count=30, percent_increase=0.001), show_progressbar=show_progress)
                                    #    trials=trials
                best_params = hpt.space_eval(search_params, best)
                with open(Conf.OutputDirName + "/" + MVA["MVAtype"] + "/best_params.txt", 'w') as f:
                    f.write(str(best_params))       
                             
                modelDNN = create_DNN_model({**best_params})
            else:
                modelDNN = MVA["DNNDict"]["model"]
                modelDNN.compile(loss=MVA["DNNDict"]["compile"]["loss"], optimizer=MVA["DNNDict"]["compile"]["optimizer"], metrics=MVA["DNNDict"]["compile"]["metrics"])
            
            batchsize = MVA["DNNDict"]["ModelParams"]["batchsize"]
            train_history = modelDNN.fit(x_train, y_train, sample_weight=w_train, validation_data=(x_test, y_test, w_test), 
                                         epochs=MVA["DNNDict"]["ModelParams"]["epochs"], batch_size=batchsize, callbacks=[es], verbose=0)
            #// train_history = modelDNN.fit(x_train, y_train, epochs=MVA["DNNDict"]["ModelParams"]["epochs"], batch_size=batchsize, validation_data=(x_test, y_test, w_test), verbose=1, callbacks=[es], sample_weight=w_train)
            modelDNN.save(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+MVA["MVAtype"]+"_"+"modelDNN.keras")

            y_train_pred = modelDNN.predict(x_train, batch_size=batchsize)  
            y_test_pred  = modelDNN.predict(x_test, batch_size=batchsize)  

            #* Save loss history
            training_loss = train_history.history['loss']
            testing_loss = train_history.history['val_loss']
            save_loss_history(MVA, training_loss, testing_loss, Conf)
            
        elif MVA["Algorithm"] == "XGB":
            # Move the data to GPU
            global dtrain, dtest
            dtrain = xgb.DMatrix(data=x_train, label=y_train, weight=w_train)
            dtest = xgb.DMatrix(data=x_test, label=y_test, weight=w_test)
            
            #* Search hyperparameter  
            best_params = {**MVA["XGBDict"]["ModelParams"], **MVA["XGBDict"]["HyperParams"]}
            if MVA["hyperopt"]:
                def objective(params): 
                    model = xgb.train(
                        params=params, dtrain=dtrain, evals=[(dtrain, 'train'), (dtest, 'eval')], 
                        early_stopping_rounds=30, num_boost_round=500, verbose_eval=False
                    )
                    score = model.best_score
                    return -score
                
                space = create_hyperopt_space("XGB")
                search_params = {**MVA["XGBDict"]["ModelParams"], **space}
                trials = hpt.Trials()
                
                best = hpt.fmin(fn=objective, space=search_params, algo=hpt.tpe.suggest, max_evals=500,
                                early_stop_fn=no_progress_loss(iteration_stop_count=30, percent_increase=0.001), show_progressbar=show_progress)
                best_params = hpt.space_eval(search_params, best)
                print(best_params)
            
            evals_result = {}
            bst = xgb.train(
                params=best_params, dtrain=dtrain, evals=[(dtrain, 'train'), (dtest, 'eval')], 
                evals_result=evals_result, early_stopping_rounds=30, num_boost_round=500, verbose_eval=False
            )
            
            #* Save loss history
            training_loss = evals_result["train"]["merror"]
            testing_loss = evals_result["eval"]["merror"]
            save_loss_history(MVA, training_loss, testing_loss, Conf)
            
            y_train_pred = bst.predict(dtrain, iteration_range=(0, bst.best_iteration + 1))  
            y_test_pred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))  

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
        
        # Features' distribution
        plot_Features(MVA, df, "add_weight", Conf)
        plot_Features_Scaled(MVA, x_train, y_train, w_train, Conf)
        
        plot_correlation_matrix(MVA, x_train, Conf)

        plot_ROC(MVA, y_train, y_test, y_train_pred, y_test_pred, w_train, w_test, Conf)
        plot_MVA(MVA, y_train, y_test, y_train_pred, y_test_pred, w_train, w_test, Conf, False)   
        
        # Confusion matrix with and without normalization
        plot_confusion_matrix(MVA, y_test, y_test_pred, Conf)
        plot_confusion_matrix(MVA, y_test, y_test_pred, Conf, True)
        

        plot_VarImportance(MVA, modelDNN, Conf, x_train, x_test, y_test) # plot_VarImportance(MVA, cv.best_estimator_, Conf)


        seconds = time.time() - start_time
        print("[INFO] Time Taken: {}".format(time.strftime("%H:%M:%S",time.gmtime(seconds))))
            
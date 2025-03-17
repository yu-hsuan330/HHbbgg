import tensorflow as tf
import pandas as pd
import numpy as np
import awkward as ak
import argparse
import pickle
import uproot
from coffea.nanoevents.methods import candidate
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator

import warnings
warnings.filterwarnings("error", module="coffea.*") 
from Config import *

def get_parser():
    
    parser = argparse.ArgumentParser(prog="ML-prediction", description="Jet pairing - prediction", epilog="Good luck!")
    parser.add_argument("-e", "--era", type=str, required=True, help="Era of the sample")
    return parser

if __name__ == "__main__":
    #* Parse the arguments
    parser = get_parser()
    args = parser.parse_args()
    
    #* Load the saved DNN model and scaler
    model = tf.keras.models.load_model(model_param["path"])

    with open(model_param["scaler"], 'rb') as f:
        sc = pickle.load(f)

    for sample in samples:
        print("Start processing file: ", sample["name"])
        
        #* Load the ROOT files
        events = uproot.concatenate(sample[args.era], library="ak")
        
        #* Process the format for prediction and do the normalization
        DNN_Input = pd.DataFrame(events[model_param["features"]].to_numpy(), columns=events[model_param["features"]].fields)
        
        # DNN_Input.loc[DNN_Input["lead_btagPNetB"] < 0, "lead_btagPNetB"] = -99
        # DNN_Input.loc[DNN_Input["sublead_btagPNetB"] < 0, "sublead_btagPNetB"] = -99
        # DNN_Input.loc[DNN_Input["lead_btagPNetQvG"] < 0, "lead_btagPNetQvG"] = -999
        # DNN_Input.loc[DNN_Input["sublead_btagPNetQvG"] < 0, "sublead_btagPNetQvG"] = -999
        # DNN_Input.columns = feature_name

        valid_mask = (DNN_Input != -99) & (DNN_Input != -999) 
        
        DNN_Input[valid_mask] = sc.transform(DNN_Input[valid_mask])
        # DNN_Input[DNN_Input == -99] = 0
        # DNN_Input[DNN_Input == -999] = -1
        DNN_Input[~valid_mask] = -1
        
        #* DNN prediction
        DNN_Score = model.predict(DNN_Input)

        # #* pair-level info -> store to ROOT file
        events["DNN_class"] = ak.argmax(DNN_Score, axis=1, mask_identity=False) 
        events["DNN0"] = DNN_Score[:,0]
        events["DNN1"] = DNN_Score[:,1]
        events["DNN2"] = DNN_Score[:,2]
        events["DNN3"] = DNN_Score[:,3]
        events["DNN4"] = DNN_Score[:,4]

        
        #* Store the selected pair to ROOT file
        with uproot.recreate(f"../samples/{model_param['output']}/{args.era}_{sample['name']}.root") as new_file:
            if sample["name"] == "data":
                new_file[TreeName_data] = events
            else:
               new_file[TreeName_mc] = events


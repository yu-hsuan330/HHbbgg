import tensorflow as tf
import pandas as pd
import numpy as np
import awkward as ak
import pickle
import uproot
from coffea.nanoevents.methods import candidate
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator

import warnings
warnings.filterwarnings("error", module="coffea.*") 

TreeName = "DiphotonTree/data_125_13TeV_NOTAG"
# TreeName = "DiphotonTree/Data_13TeV_NOTAG"
train_model = "MultiClass_DNN_EventCate_v3" #"MultiClass_DNN_STXS_0107" 
version = "ver0121"
minitree = [
            "22preEE_VBFToHH", "22postEE_VBFToHH", 
            "22preEE_GluGluToHH", "22postEE_GluGluToHH",
            "22preEE_GJet_Pt20to40_MGG80", "22postEE_GJet_Pt20to40_MGG80", "22preEE_GJet_Pt40_MGG80", "22postEE_GJet_Pt40_MGG80",
            "22preEE_GGJets", "22postEE_GGJets",
            "22preEE_QCD_Pt30to40_MGG80", "22postEE_QCD_Pt30to40_MGG80", 
            "22preEE_QCD_Pt40_MGG80", "22postEE_QCD_Pt40_MGG80",
            "22preEE_QCD_Pt30_MGG40to80", "22postEE_QCD_Pt30_MGG40to80",
            # "data/22postEE_EGammaE", "data/22postEE_EGammaF", "data/22postEE_EGammaG",
            # "data/22preEE_EGammaC", "data/22preEE_EGammaD"
            ]
lumi_xs = [
    7980.4*0.00192, 26671.7*0.00192, 
    7980.4*0.03443, 26671.7*0.03443, 
    7980.4*242.5, 26671.7*242.5, 7980.4*919.1, 26671.7*919.1,
    7980.4*88.75, 26671.7*88.75,
    7980.4*25950, 26671.7*25950, 
    7980.4*124700, 26671.7*124700,
    7980.4*252200, 26671.7*252200,
    # 1,1,1,
    # 1,1
]
DNN_features = [
    # photon info
    "pho_lead_ptOverM",         "pho_sublead_ptOverM",          "pho_lead_eta",             "pho_sublead_eta",
    "diphoton_ptOverM",         "diphoton_eta",                 "diphoton_phi",             "n_jets",
    # b-jet info
    "bjet_lead_ptOverM",        "bjet_sublead_ptOverM",         "bjet_lead_eta",            "bjet_sublead_eta",
    "bjet_lead_btagPNetB",      "bjet_sublead_btagPNetB",       "bjet_lead_btagPNetQvG",    "bjet_sublead_btagPNetQvG",
    "bjet_pair_ptOverM",        "bjet_pair_eta",                "bjet_pair_phi",            "bjet_pair_Cgg",
    "bjet_pair_eta_prod",       "bjet_pair_eta_diff",           "bjet_pair_DeltaR",         "bjet_pair_DeltaPhi", 
    # vbf-jet info
    "vbfjet_lead_ptOverM",      "vbfjet_sublead_ptOverM",       "vbfjet_lead_eta",          "vbfjet_sublead_eta", 
    "vbfjet_lead_btagPNetB",    "vbfjet_sublead_btagPNetB",     "vbfjet_lead_btagPNetQvG",  "vbfjet_sublead_btagPNetQvG",
    "vbfjet_pair_ptOverM",      "vbfjet_pair_eta",              "vbfjet_pair_phi",          "vbfjet_pair_mass", 
    "vbfjet_pair_eta_prod",     "vbfjet_pair_eta_diff",         "vbfjet_pair_DeltaR",       "vbfjet_pair_DeltaPhi",         
    "vbfjet_pair_Cgg",          "vbfjet_pair_Cbb" 
]
if __name__ == "__main__":

    #* Load the saved DNN model
    model = tf.keras.models.load_model("../{}/DNN/DNN_modelDNN.keras".format(train_model))

    with open('../{}/DNN/DNN_scaler.pkl'.format(train_model), 'rb') as f:
        sc = pickle.load(f)

    for i, j in zip(minitree, lumi_xs):
        print("Start processing file: ", i)
        #* Load the ROOT files
        events = uproot.concatenate(["../update_minitree/{}_/{}.root:{}".format(version, i, TreeName)], library="ak")
                
        #* Process the format for prediction and do the normalization
        DNN_Input = pd.DataFrame(events[DNN_features].to_numpy(), columns=events[DNN_features].fields)
        
        # DNN_Input.loc[DNN_Input["lead_btagPNetB"] < 0, "lead_btagPNetB"] = -99
        # DNN_Input.loc[DNN_Input["sublead_btagPNetB"] < 0, "sublead_btagPNetB"] = -99
        # DNN_Input.loc[DNN_Input["lead_btagPNetQvG"] < 0, "lead_btagPNetQvG"] = -999
        # DNN_Input.loc[DNN_Input["sublead_btagPNetQvG"] < 0, "sublead_btagPNetQvG"] = -999
        # DNN_Input.columns = feature_name

        valid_mask = (DNN_Input != -99) & (DNN_Input != -999) 
        
        DNN_Input[valid_mask] = sc.transform(DNN_Input[valid_mask])
        DNN_Input[DNN_Input == -99] = 0
        DNN_Input[DNN_Input == -999] = -1
        # DNN_Input[~valid_mask] = -1
        
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
        with uproot.recreate("../update_minitree/{}_cate/{}.root".format(version, i)) as new_file:
            new_file[TreeName] = events


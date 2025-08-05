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
from Config_raw import *

def Cxx(higgs_eta, VBFjet_eta_diff, VBFjet_eta_sum):
    # Centrality variable
    return np.exp(-4 / (VBFjet_eta_diff) ** 2 * (higgs_eta - (VBFjet_eta_sum) / 2) ** 2)

def get_parser():
    
    parser = argparse.ArgumentParser(prog="ML-prediction", description="Jet pairing - prediction", epilog="Good luck!")
    parser.add_argument("-e", "--era", type=str, required=True, help="Era of the sample")
    return parser

if __name__ == "__main__":
    #* Parse the arguments
    parser = get_parser()
    args = parser.parse_args()
    orig_features = [
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
    #* Load the saved DNN model and scaler
    model = tf.keras.models.load_model(model_param["path"])

    with open(model_param["scaler"], 'rb') as f:
        sc = pickle.load(f)

    for sample in samples:
        print("Start processing file: ", sample["name"])
        #* Load the ROOT files
        events = uproot.concatenate(sample[args.era], library="ak")
        events = events[(events["is_nonRes"] == 1) & (events["nonRes_has_two_btagged_jets"] == 1) & (events["nonRes_lead_bjet_eta"] - events["nonRes_sublead_bjet_eta"] != 0)]
                
        lead_bjet = ak.zip({
            "pt": events["nonRes_lead_bjet_pt"],
            "eta": events["nonRes_lead_bjet_eta"],
            "phi": events["nonRes_lead_bjet_phi"],
            "mass": events["nonRes_lead_bjet_mass"],
            "charge": events["nonRes_lead_bjet_charge"]
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)
        
        sublead_bjet = ak.zip({
            "pt": events["nonRes_sublead_bjet_pt"],
            "eta": events["nonRes_sublead_bjet_eta"],
            "phi": events["nonRes_sublead_bjet_phi"],
            "mass": events["nonRes_sublead_bjet_mass"],
            "charge": events["nonRes_sublead_bjet_charge"]
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)
        
        lead_vbfjet = ak.zip({
            "pt": events["VBF_first_jet_pt"],
            "eta": events["VBF_first_jet_eta"],
            "phi": events["VBF_first_jet_phi"],
            "mass": events["VBF_first_jet_mass"],
            "charge": events["VBF_first_jet_charge"]
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)
        
        sublead_vbfjet = ak.zip({
            "pt": events["VBF_second_jet_pt"],
            "eta": events["VBF_second_jet_eta"],
            "phi": events["VBF_second_jet_phi"],
            "mass": events["VBF_second_jet_mass"],
            "charge": events["VBF_second_jet_charge"]
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)
        
        events["diphoton_ptOverM"] = events["pt"]/events["CMS_hgg_mass"]
        events["bjet_pair_ptOverM"] = events["nonRes_dijet_pt"]/events["nonRes_dijet_mass"]
        events["vbfjet_pair_ptOverM"] = ak.where(events["VBF_dijet_pt"] != -999, events["VBF_dijet_pt"]/events["VBF_dijet_mass"], -999)
        
        events["nonRes_dijet_deltaR"] = lead_bjet.delta_r(sublead_bjet)
        events["nonRes_dijet_deltaPhi"] = lead_bjet.delta_phi(sublead_bjet)
        events["nonRes_dijet_prodEta"] = lead_bjet.eta * sublead_bjet.eta
        events["nonRes_dijet_deltaEta"] = lead_bjet.eta - sublead_bjet.eta
        events["nonRes_dijet_eta_sum"] = lead_bjet.eta + sublead_bjet.eta
        events["nonRes_dijet_Cgg"] = Cxx(events["eta"], events["nonRes_dijet_deltaEta"], events["nonRes_dijet_eta_sum"])

        events["VBF_dijet_deltaR"] = ak.where(events["VBF_dijet_pt"] != -999, lead_vbfjet.delta_r(sublead_vbfjet), -999)
        events["VBF_dijet_deltaPhi"] = ak.where(events["VBF_dijet_pt"] != -999, lead_vbfjet.delta_phi(sublead_vbfjet), -999)

        #* Process the format for prediction and do the normalization
        # DNN_Input = pd.DataFrame(events[model_param["features"]].to_numpy(), columns=orig_features)
        DNN_Input = pd.DataFrame(events[model_param["features"]].to_numpy())
        DNN_Input.columns = orig_features

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


import tensorflow as tf
import pandas as pd
import numpy as np
import awkward as ak
import pickle
import uproot
import argparse
from coffea.nanoevents.methods import candidate
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator

import warnings
warnings.filterwarnings("error", module="coffea.*")
    
from Config_Yu import *

def Cxx(jet_eta_diff, jet_eta_sum, xx_eta):
    jet_eta_diff = np.where(jet_eta_diff == 0, 1e-10, jet_eta_diff)
    return np.exp(-4 / (jet_eta_diff) ** 2 * (xx_eta - (jet_eta_sum) / 2) ** 2)
    
def pltSty(ax, xName = "x-axis", yName = "y-axis", TitleSize = 17, LabelSize = 16, TickSize = 13, MajTickLength = 7, MinTickLength = 4, yAuto = True):
    ax.set_xlabel(xName, fontsize = LabelSize, loc = "right")
    ax.set_ylabel(yName, fontsize = LabelSize, loc = "top")
    # ax.text(1, 1, "(13 TeV)", horizontalalignment = "right", verticalalignment = "bottom", transform = ax.transAxes, fontsize = TitleSize)
    ax.text(0, 1.01, "CMS", horizontalalignment = "left", verticalalignment = "bottom", transform = ax.transAxes, fontsize = TitleSize * 1.3, fontweight = "bold")
    ax.text(TitleSize * 0.01, 1.015, "work-in-progress", horizontalalignment = "left", verticalalignment = "bottom", transform = ax.transAxes, fontsize = TitleSize)

    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if yAuto :
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction = "in", length = MajTickLength, labelsize = TickSize, top = True, right = True)
    ax.tick_params(direction = "in", length = MinTickLength, which = "minor", labelsize = TickSize, top = True, right = True)

feature_name = [
    "pair1_ptOverM",        "pair2_ptOverM",        "pair1_eta",            "pair2_eta", 
    "pair1_btagPNetB",      "pair2_btagPNetB",      "pair1_btagPNetQvG",    "pair2_btagPNetQvG",
    "n_jets",               "pair_pt",              "pair_eta",             "pair_mass", 
    "pair_DeltaR",          "pair_DeltaPhi",        "pair_eta_prod",        "pair_eta_diff", 
    "pair_Cgg",             "pair1_phi",            "pair2_phi"
] 

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
        # events = events[:1000]
        #* Check the existence of the LHE info
        check_list = ["jet1_lheMatched", "VBF_lead_jet_lheMatched", "VBF_sublead_jet_lheMatched", "jet1_genFlav", "nonRes_lead_bjet_genFlav", "nonRes_sublead_bjet_genFlav"]
        
        for check_var in check_list:
            if check_var not in events.fields:
                if check_var == "jet1_lheMatched":
                    for k in range(1, 7): events["jet{}_lheMatched".format(k)] = -999
                elif check_var == "jet1_genFlav":
                    for k in range(1, 7): events["jet{}_genFlav".format(k)] = -999
                else:
                    events[check_var] = -999

        #* Define jet collection
        jet1 = ak.zip({"pt": events["jet1_pt"], "eta": events["jet1_eta"], "phi": events["jet1_phi"], "mass": events["jet1_mass"], "charge": events["jet1_charge"], "btagPNetB": events["jet1_btagPNetB"], "btagPNetQvG": events["jet1_btagPNetQvG"], "genFlav": events["jet1_genFlav"], "lheMatched": events["jet1_lheMatched"]})
        jet2 = ak.zip({"pt": events["jet2_pt"], "eta": events["jet2_eta"], "phi": events["jet2_phi"], "mass": events["jet2_mass"], "charge": events["jet2_charge"], "btagPNetB": events["jet2_btagPNetB"], "btagPNetQvG": events["jet2_btagPNetQvG"], "genFlav": events["jet2_genFlav"], "lheMatched": events["jet2_lheMatched"]})
        jet3 = ak.zip({"pt": events["jet3_pt"], "eta": events["jet3_eta"], "phi": events["jet3_phi"], "mass": events["jet3_mass"], "charge": events["jet3_charge"], "btagPNetB": events["jet3_btagPNetB"], "btagPNetQvG": events["jet3_btagPNetQvG"], "genFlav": events["jet3_genFlav"], "lheMatched": events["jet3_lheMatched"]})
        jet4 = ak.zip({"pt": events["jet4_pt"], "eta": events["jet4_eta"], "phi": events["jet4_phi"], "mass": events["jet4_mass"], "charge": events["jet4_charge"], "btagPNetB": events["jet4_btagPNetB"], "btagPNetQvG": events["jet4_btagPNetQvG"], "genFlav": events["jet4_genFlav"], "lheMatched": events["jet4_lheMatched"]})
        jet5 = ak.zip({"pt": events["jet5_pt"], "eta": events["jet5_eta"], "phi": events["jet5_phi"], "mass": events["jet5_mass"], "charge": events["jet5_charge"], "btagPNetB": events["jet5_btagPNetB"], "btagPNetQvG": events["jet5_btagPNetQvG"], "genFlav": events["jet5_genFlav"], "lheMatched": events["jet5_lheMatched"]})
        jet6 = ak.zip({"pt": events["jet6_pt"], "eta": events["jet6_eta"], "phi": events["jet6_phi"], "mass": events["jet6_mass"], "charge": events["jet6_charge"], "btagPNetB": events["jet6_btagPNetB"], "btagPNetQvG": events["jet6_btagPNetQvG"], "genFlav": events["jet6_genFlav"], "lheMatched": events["jet6_lheMatched"]})

        jets = ak.concatenate([jet1[:,None], jet2[:,None], jet3[:,None], jet4[:,None], jet5[:,None], jet6[:,None]], axis=1)
        jets = ak.with_name(jets, "PtEtaPhiMCandidate", behavior=candidate.behavior)
        
        jets = ak.drop_none(jets.mask[jets.pt > 0]) 
        jets["idx"] = ak.local_index(jets, axis=1)
        
        #* Define the true b-jet and vbf-jet
        jets["true_bjet"] = abs(jets.genFlav) == 5
        jets["true_vbfjet"] = (abs(jets.genFlav) != 5) & (jets.lheMatched == 1)
        
        # # Change the nan value to -999
        # # jets["btagPNetB"] = ak.where(jets["btagPNetB"] < 0, -999, jets["btagPNetB"])
        # # jets["btagPNetQvG"] = ak.where(jets["btagPNetQvG"] < 0, -999, jets["btagPNetQvG"])
        events["lumi"] = lumi[args.era]
        events["xs"] = sample["xs"]
        
        events["true_bbjj"] = (ak.sum(jets["true_bjet"], axis=-1) == 2) & (ak.sum(jets["true_vbfjet"], axis=-1) == 2)
        events["true_bb"] = (ak.sum(jets["true_bjet"], axis=-1) == 2)
        events["true_jj"] = (ak.sum(jets["true_vbfjet"], axis=-1) == 2)
  
        dijets = ak.combinations(jets, 2, fields=["lead", "sublead"])
        pair = dijets["lead"] + dijets["sublead"]
        
        dijets["eta_diff"] = dijets.lead.eta - dijets.sublead.eta
        dijets["eta_sum"] = dijets.lead.eta + dijets.sublead.eta
        
        dijets["diphoton_eta"] = events["eta"]
  
        branches = ak.zip({
            "event"             : events["event"],
            "lumi_xs"           : events["lumi"]*events["xs"],
            "lumi"              : events["lumi"],
            "xs"                : events["xs"],
            "weight"            : events["weight"],
            "n_jets"            : events["n_jets"], 
            
            "N_DNN_class"       : events["DNN_class"],
            "N_DNN0"            : events["DNN0"],
            "N_DNN1"            : events["DNN1"],
            "N_DNN2"            : events["DNN2"],
            "N_DNN3"            : events["DNN3"],
            "N_DNN4"            : events["DNN4"],
            
            "VBF_selected"      : events["VBF_dijet_pt"] > 0,
            "true_bbjj"         : events["true_bbjj"],
            "true_bb"           : events["true_bb"],
            "true_jj"           : events["true_jj"],
            
            "diphoton_pt"       : events["pt"],
            "diphoton_eta"      : events["eta"], 
            "diphoton_phi"      : events["phi"], 
            "diphoton_mass"     : events["CMS_hgg_mass"],
            
            "pho_lead_pt"       : events["lead_pt"] ,
            "pho_sublead_pt"    : events["sublead_pt"],
            "pho_lead_eta"      : events["lead_eta"],
            "pho_sublead_eta"   : events["sublead_eta"],
            "pho_lead_phi"      : events["lead_phi"],
            "pho_sublead_phi"   : events["sublead_phi"],

            "lead_idx"          : dijets.lead.idx,
            "sublead_idx"       : dijets.sublead.idx,
            "lead_pt"           : dijets.lead.pt, 
            "sublead_pt"        : dijets.sublead.pt, 
            "lead_eta"          : dijets.lead.eta, 
            "sublead_eta"       : dijets.sublead.eta, 
            "lead_phi"          : dijets.lead.phi,  
            "sublead_phi"       : dijets.sublead.phi,
            "lead_ptOverM"      : dijets.lead.pt / pair.mass, 
            "sublead_ptOverM"   : dijets.sublead.pt / pair.mass, 
            "lead_btagPNetB"    : dijets.lead.btagPNetB, 
            "sublead_btagPNetB" : dijets.sublead.btagPNetB, 
            "lead_btagPNetQvG"      : dijets.lead.btagPNetQvG, 
            "sublead_btagPNetQvG"   : dijets.sublead.btagPNetQvG,
            
            "pair_pt"           : pair.pt,
            "pair_ptOverM"      : pair.pt / pair.mass,
            "pair_eta"          : pair.eta, 
            "pair_phi"          : pair.phi, 
            "pair_mass"         : pair.mass,
            "pair_charge"       : pair.charge,
            "pair_DeltaR"       : dijets["lead"].delta_r(dijets["sublead"]), 
            "pair_DeltaPhi"     : dijets["lead"].delta_phi(dijets["sublead"]),
            "pair_eta_prod"     : dijets.lead.eta * dijets.sublead.eta, 
            "pair_eta_diff"     : dijets["eta_diff"],
            "pair_eta_sum"      : dijets["eta_sum"],
            "pair_Cgg"          : Cxx(dijets["eta_diff"], dijets["eta_sum"], dijets["diphoton_eta"]),
            "true_bjet_pair"    : dijets["lead"].true_bjet & dijets["sublead"].true_bjet,
            "true_vbfjet_pair"  : dijets["lead"].true_vbfjet & dijets["sublead"].true_vbfjet,
            # "select_bjet_pair"  : dijets["lead"].selected_bjet & dijets["sublead"].selected_bjet,
            # "select_vbfjet_pair": dijets["lead"].selected_vbfjet & dijets["sublead"].selected_vbfjet,
            }
        )
        branches["lead"] = dijets["lead"]
        branches["sublead"] = dijets["sublead"]
        
        #* Check purity of the sample
        if sample["name"] == "VBFToHH":
            
            print("===== Total events =====")
            have_vbf_event = (events["VBF_dijet_pt"] > 0) & (events["nonRes_lead_bjet_pt"] > 0)
            print("# of all events in minitree: {}, {}".format(ak.num(events, axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"], axis=0)))
            print("# of VBF-events in minitree: {}, {}".format(ak.num(events[have_vbf_event], axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*have_vbf_event, axis=0)))
            
            print("=====  Cut-based  =====")
            criteria_bb = have_vbf_event & (abs(events["nonRes_lead_bjet_genFlav"]) == 5) & (abs(events["nonRes_sublead_bjet_genFlav"]) == 5)
            criteria_jj = have_vbf_event & (events["VBF_lead_jet_lheMatched"] == 1) & (events["VBF_sublead_jet_lheMatched"] == 1)
            print("true b-jet   in VBF-events: {}, {}".format(ak.num(events[criteria_bb],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(criteria_bb), axis=0)))
            print("true vbf-jet in VBF-events: {}, {}".format(ak.num(events[criteria_jj],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(criteria_jj), axis=0)))
            print("true bbjj    in VBF-events: {}, {}".format(ak.num(events[criteria_bb & criteria_jj],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(criteria_bb & criteria_jj), axis=0)))

            print("===== True events =====")
            is_true_bb = (abs(branches["lead_eta"]) < 2.5) & (abs(branches["sublead_eta"]) < 2.5) & (branches["true_bjet_pair"] == 1) & (branches["pair_mass"] > 70) & (branches["pair_mass"] < 190)
            is_true_jj = (branches["lead_pt"] > 40) & (branches["sublead_pt"] > 30) & (branches["true_vbfjet_pair"] == 1)
            
            true_pair_bbjj = ak.cartesian({"bb": branches[is_true_bb], "jj": branches[is_true_jj]})
            
            deltaRCut = (true_pair_bbjj["bb"].lead.delta_r(true_pair_bbjj["jj"].lead) > 0.4) & (true_pair_bbjj["bb"].lead.delta_r(true_pair_bbjj["jj"].sublead) > 0.4) & (true_pair_bbjj["bb"].sublead.delta_r(true_pair_bbjj["jj"].lead) > 0.4) & (true_pair_bbjj["bb"].sublead.delta_r(true_pair_bbjj["jj"].sublead) > 0.4)
            idxCut = (true_pair_bbjj["bb"].lead.idx != true_pair_bbjj["jj"].lead.idx) & (true_pair_bbjj["bb"].sublead.idx != true_pair_bbjj["jj"].sublead.idx) # supposely useless

            check_results = ak.firsts(true_pair_bbjj[deltaRCut & idxCut])
            
            # print("true bbjj wo cut in all events: {}, {} (deprecated)".format(ak.num(events[(events["true_bbjj"]==1)],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(events["true_bbjj"]==1), axis=0)))
            # criteria = (events["true_bbjj"]==1) & (~ak.is_none(check_results.jj.event))
            # print("true bbjj comb   in all events: {}, {} (deprecated)".format(ak.num(events[criteria],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(criteria), axis=0)))

            criteria = (~ak.is_none(check_results.jj.event))
            print("true bbjj    in all events: {}, {}".format(ak.num(events[criteria],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(criteria), axis=0)))
            criteria = (~ak.is_none(check_results.jj.event)) & criteria_bb
            print("true b-jet   in cut-based: {}, {}".format(ak.num(events[criteria],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(criteria), axis=0)))
            criteria = (~ak.is_none(check_results.jj.event)) & criteria_jj
            print("true vbf-jet in cut-based: {}, {}".format(ak.num(events[criteria],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(criteria), axis=0)))
            criteria = (~ak.is_none(check_results.jj.event)) & criteria_bb & criteria_jj
            print("true bbjj    in cut-based: {}, {}".format(ak.num(events[criteria],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(criteria), axis=0)))
       
        #* Process the format for prediction
        counts = ak.num(branches)
        flat_features = ak.flatten(branches[model_param["features"]])
        #* Convert structured array to a flat array and do the normalization
        DNN_Input = pd.DataFrame(flat_features.to_numpy(), columns=flat_features.fields)
        
        DNN_Input.loc[DNN_Input["lead_btagPNetB"] < 0, "lead_btagPNetB"] = -999
        DNN_Input.loc[DNN_Input["sublead_btagPNetB"] < 0, "sublead_btagPNetB"] = -999
        DNN_Input.loc[DNN_Input["lead_btagPNetQvG"] < 0, "lead_btagPNetQvG"] = -999
        DNN_Input.loc[DNN_Input["sublead_btagPNetQvG"] < 0, "sublead_btagPNetQvG"] = -999
        DNN_Input.columns = feature_name

        valid_mask = (DNN_Input != -999)
        DNN_Input[valid_mask] = sc.transform(DNN_Input[valid_mask])
        DNN_Input[~valid_mask] = -1

        #* DNN prediction
        DNN_Score = model.predict(DNN_Input)
        DNN_VBF = ak.unflatten(DNN_Score, counts)

        branches["DNN_class"] = ak.argmax(DNN_VBF, axis=2, mask_identity=False)
        branches["DNN_bb_pair"] = DNN_VBF[:,:,0]
        branches["DNN_jj_pair"] = DNN_VBF[:,:,1]
        branches["DNN_wrong_pair"] = DNN_VBF[:,:,2]
        
        #* b-jet pair selection
        is_bb = (branches["DNN_class"] == 0) & (abs(branches["lead_eta"]) < 2.5) & (abs(branches["sublead_eta"]) < 2.5) & (branches["pair_mass"] > 70) & (branches["pair_mass"] < 190)
        is_jj = (branches["DNN_class"] == 1) & (branches["lead_pt"] > 40) & (branches["sublead_pt"] > 30)

        candidate_bb = branches[is_bb]
        candidate_bb = candidate_bb[ak.argsort(candidate_bb.lead_btagPNetB+candidate_bb.sublead_btagPNetB, ascending=False)]     

        candidate_jj = branches[is_jj]
        candidate_jj = candidate_jj[ak.argsort(candidate_jj.pair_mass, ascending=False)] 

        pair_bb = ak.firsts(candidate_bb)

        deltaRCut = (pair_bb.lead.delta_r(candidate_jj.lead) > 0.4) & (pair_bb.lead.delta_r(candidate_jj.sublead) > 0.4) & (pair_bb.sublead.delta_r(candidate_jj.lead) > 0.4) & (pair_bb.sublead.delta_r(candidate_jj.sublead) > 0.4)
        idxCut = (pair_bb.lead.idx != candidate_jj.lead.idx) & (pair_bb.sublead.idx != candidate_jj.sublead.idx)
        
        pair_jj = ak.firsts(candidate_jj[deltaRCut & idxCut])
        
        mask = ak.is_none(pair_bb)
        # print(pair_bb.event, pair_jj.event, (~ak.is_none(check_results.jj.event)))

        #* Check purity of the sample
        if sample["name"] == "VBFToHH":
            print("===== DNN =====")
            criteria = (~ak.is_none(pair_bb.event)) & (~ak.is_none(pair_jj.event))
            print("# of VBF-events by DNN: {}, {}".format(ak.num(events[criteria],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(criteria), axis=0)))

            criteria = (~ak.is_none(check_results.jj.event))
            print("true bbjj    in all events: {}, {}".format(ak.num(events[criteria],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(criteria), axis=0)))
            criteria = (~ak.is_none(check_results.jj.event)) & (~ak.is_none(pair_bb.event))
            print("true b-jet   in DNN: {}, {}".format(ak.num(events[criteria],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(criteria), axis=0)))
            criteria = (~ak.is_none(check_results.jj.event)) & (~ak.is_none(pair_jj.event))
            print("true vbf-jet in DNN: {}, {}".format(ak.num(events[criteria],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(criteria), axis=0)))
            criteria = (~ak.is_none(check_results.jj.event)) & (~ak.is_none(pair_bb.event)) & (~ak.is_none(pair_jj.event))
            print("true bbjj    in DNN: {}, {}".format(ak.num(events[criteria],axis=0), ak.sum(events["weight"]*lumi[args.era]*sample["xs"]*(criteria), axis=0)))
       
        pair_bb = ak.drop_none(pair_bb)
        pair_jj = pair_jj[~mask]

        output = ak.zip({
            "event"             : pair_bb.event,
            "lumi_xs"           : pair_bb.lumi_xs,
            "weight"            : pair_bb.weight,
            "n_jets"            : pair_bb.n_jets,    
            
            "diphoton_pt"       : pair_bb.diphoton_pt,
            "diphoton_eta"      : pair_bb.diphoton_eta,
            "diphoton_phi"      : pair_bb.diphoton_phi,
            "diphoton_mass"     : pair_bb.diphoton_mass,
            "diphoton_ptOverM"  : pair_bb.diphoton_pt / pair_bb.diphoton_mass,
            
            "pho_lead_pt"       : pair_bb.pho_lead_pt, 
            "pho_sublead_pt"    : pair_bb.pho_sublead_pt, 
            "pho_lead_eta"      : pair_bb.pho_lead_eta, 
            "pho_sublead_eta"   : pair_bb.pho_sublead_eta, 
            "pho_lead_phi"      : pair_bb.pho_lead_phi, 
            "pho_sublead_phi"   : pair_bb.pho_sublead_phi, 
            "pho_lead_ptOverM"      : pair_bb.pho_lead_pt / pair_bb.diphoton_mass,
            "pho_sublead_ptOverM"   : pair_bb.pho_sublead_pt / pair_bb.diphoton_mass,
            
            "N_DNN_class"       : pair_bb.N_DNN_class,
            "N_DNN0"            : pair_bb.N_DNN0,
            "N_DNN1"            : pair_bb.N_DNN1,
            "N_DNN2"            : pair_bb.N_DNN2,
            "N_DNN3"            : pair_bb.N_DNN3,
            "N_DNN4"            : pair_bb.N_DNN4,
        })

        store_variables = [
            "lead_pt", "sublead_pt", "lead_ptOverM", "sublead_ptOverM", "lead_eta", "sublead_eta", "lead_phi", "sublead_phi", 
            "lead_btagPNetB", "sublead_btagPNetB", "lead_btagPNetQvG", "sublead_btagPNetQvG",
            "pair_pt", "pair_ptOverM", "pair_eta", "pair_phi", "pair_mass", "pair_DeltaR", "pair_DeltaPhi", 
            "pair_eta_prod", "pair_eta_diff", "pair_eta_sum", "pair_Cgg", 
            "DNN_class", "DNN_bb_pair", "DNN_jj_pair", "DNN_wrong_pair", 
            "true_bjet_pair", "true_vbfjet_pair"
        ]
        for field in store_variables:
            output["bjet_"+field] = pair_bb[field]
            
        for field in store_variables:
            value = ak.fill_none(getattr(pair_jj, field), -999)
            output["vbfjet_"+field] = value
            
        output["vbfjet_pair_Cbb"] = ak.where(output["vbfjet_pair_pt"] > 0, Cxx(output["vbfjet_pair_eta_diff"], output["vbfjet_pair_eta_sum"], output["bjet_pair_eta"]), -999)

        # print(output["vbfjet_pair_Cbb"])
        #* Store the selected pair to ROOT file
        with uproot.recreate(f"../samples/{model_param['output']}/{args.era}_{sample['name']}.root") as new_file:
            if sample["name"] == "data":
                new_file[TreeName_data] = output
            else:
               new_file[TreeName_mc] = output

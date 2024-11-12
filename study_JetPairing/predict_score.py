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

def Cgg(jet_eta_diff, jet_eta_sum, gg_eta):
    jet_eta_diff = np.where(jet_eta_diff == 0, 1e-10, jet_eta_diff)
    return np.exp(-4 / (jet_eta_diff) ** 2 * (gg_eta - (jet_eta_sum) / 2) ** 2)
    
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

TreeName = "DiphotonTree/data_125_13TeV_NOTAG"
version = "ver1108"
minitree = [
            "22preEE_VBFToHH", "22postEE_VBFToHH", 
            # "22preEE_GluGluToHH", "22postEE_GluGluToHH",
            # "22preEE_GJet_Pt20to40_MGG80", "22postEE_GJet_Pt20to40_MGG80", "22preEE_GJet_Pt40_MGG80", "22postEE_GJet_Pt40_MGG80",
            # "22preEE_GGJets", "22postEE_GGJets",
            # "22preEE_QCD_Pt30to40_MGG80", "22postEE_QCD_Pt30to40_MGG80", "22preEE_QCD_Pt40_MGG80", "22postEE_QCD_Pt40_MGG80",
            # "22preEE_QCD_Pt30_MGG40to80", "22postEE_QCD_Pt30_MGG40to80",
            ]
lumi_xs = [
    7980.4*0.00192, 26671.7*0.00192, 
    # 7980.4*0.03443, 26671.7*0.03443, 
    # 7980.4*242.5, 26671.7*242.5, 7980.4*919.1, 26671.7*919.1,
    # 7980.4*88.75, 26671.7*88.75,
    # 7980.4*25950, 26671.7*25950, 7980.4*124700, 26671.7*124700,
    # 7980.4*252200, 26671.7*252200
]
branch_features = [
    "pt", "eta", "phi", "n_jets", 
    "pair1_ptOverM", "pair2_ptOverM", "pair1_eta", "pair2_eta", 
    "pair1_btagPNetB", "pair2_btagPNetB", "pair1_btagPNetQvG", "pair2_btagPNetQvG",
    "pair_pt","pair_eta", "pair_phi", "pair_mass",
    "pair_DeltaR", #"pair_DeltaPhi", 
    "pair_eta_prod", "pair_eta_diff", 
    "pair_Cgg", #"pair1_phi", "pair2_phi"
]

if __name__ == "__main__":

    #* Load the saved DNN model
    model = tf.keras.models.load_model("./MultiClass_DNN_VBF_1110/DNN/DNN_modelDNN.keras")

    with open('./MultiClass_DNN_VBF_1110/DNN/DNN_scaler.pkl', 'rb') as f:
        sc = pickle.load(f)

    #* Load the ROOT files
    for i, j in zip(minitree, lumi_xs):
        print("Processing: ", i)
        events = uproot.concatenate(["../minitree/{}/{}.root:{}".format(version, i, TreeName)], library="ak")
        # events = events[:50]
        jet1 = ak.zip({"pt": events["jet1_pt"], "eta": events["jet1_eta"], "phi": events["jet1_phi"], "mass": events["jet1_mass"], "charge": events["jet1_charge"], "btagPNetB": events["jet1_btagPNetB"], "btagPNetQvG": events["jet1_btagPNetQvG"]})
        jet2 = ak.zip({"pt": events["jet2_pt"], "eta": events["jet2_eta"], "phi": events["jet2_phi"], "mass": events["jet2_mass"], "charge": events["jet2_charge"], "btagPNetB": events["jet2_btagPNetB"], "btagPNetQvG": events["jet2_btagPNetQvG"]})
        jet3 = ak.zip({"pt": events["jet3_pt"], "eta": events["jet3_eta"], "phi": events["jet3_phi"], "mass": events["jet3_mass"], "charge": events["jet3_charge"], "btagPNetB": events["jet3_btagPNetB"], "btagPNetQvG": events["jet3_btagPNetQvG"]})
        jet4 = ak.zip({"pt": events["jet4_pt"], "eta": events["jet4_eta"], "phi": events["jet4_phi"], "mass": events["jet4_mass"], "charge": events["jet4_charge"], "btagPNetB": events["jet4_btagPNetB"], "btagPNetQvG": events["jet4_btagPNetQvG"]})
        jet5 = ak.zip({"pt": events["jet5_pt"], "eta": events["jet5_eta"], "phi": events["jet5_phi"], "mass": events["jet5_mass"], "charge": events["jet5_charge"], "btagPNetB": events["jet5_btagPNetB"], "btagPNetQvG": events["jet5_btagPNetQvG"]})
        jet6 = ak.zip({"pt": events["jet6_pt"], "eta": events["jet6_eta"], "phi": events["jet6_phi"], "mass": events["jet6_mass"], "charge": events["jet6_charge"], "btagPNetB": events["jet6_btagPNetB"], "btagPNetQvG": events["jet6_btagPNetQvG"]})

        jets = ak.concatenate([jet1[:,None], jet2[:,None], jet3[:,None], jet4[:,None], jet5[:,None], jet6[:,None]], axis=1)
        jets = ak.with_name(jets, "PtEtaPhiMCandidate", behavior=candidate.behavior)
        
        jets = ak.drop_none(jets.mask[jets.pt > 0]) 
        jets["idx"] = ak.local_index(jets, axis=1)

        dijets = ak.combinations(jets, 2, fields=["lead", "sublead"])
        pair = dijets["lead"] + dijets["sublead"]
        
        dijets["pt"] = pair.pt
        dijets["eta"] = pair.eta
        dijets["phi"] = pair.phi
        dijets["mass"] = pair.mass
        dijets["charge"] = pair.charge
        
        dijets["event"] = events["event"]
        dijets["weight"] = events["weight"]
        dijets["selected"] = events["VBF_dijet_pt"] > 0        
        dijets["gg_pt"] = events["pt"]
        dijets["gg_eta"] = events["eta"]
        dijets["gg_phi"] = events["phi"]
        dijets["n_jets"] = events["n_jets"]

        dijets["eta_diff"] = dijets.lead.eta - dijets.sublead.eta
        dijets["eta_sum"] = dijets.lead.eta + dijets.sublead.eta

  
        branches = ak.zip({
            "event": dijets["event"],
            "weight": dijets["weight"],
            "selected": dijets["selected"],
            "lead_idx": dijets["lead"].idx,
            "sublead_idx": dijets["sublead"].idx,
            "pt": dijets["gg_pt"],
            "eta": dijets["gg_eta"], 
            "phi": dijets["gg_phi"], 
            "n_jets": dijets["n_jets"], 
            "pair1_pt": dijets["lead"].pt, 
            "pair2_pt": dijets["sublead"].pt, 
            "pair1_ptOverM": dijets["lead"].pt / dijets["mass"], 
            "pair2_ptOverM": dijets["sublead"].pt / dijets["mass"], 
            "pair1_eta": dijets.lead.eta, 
            "pair2_eta": dijets.sublead.eta, 
            "pair1_phi": dijets.lead.eta,  
            "pair2_phi": dijets.sublead.eta,
            "pair1_btagPNetB": dijets.lead.btagPNetB, 
            "pair2_btagPNetB": dijets.sublead.btagPNetB, 
            "pair1_btagPNetQvG": dijets.lead.btagPNetQvG, 
            "pair2_btagPNetQvG": dijets.sublead.btagPNetQvG,
            "pair_pt": dijets.pt,
            "pair_eta": dijets.eta, 
            "pair_phi": dijets.phi, 
            "pair_mass": dijets.mass,
            "pair_DeltaR": dijets["lead"].delta_r(dijets["sublead"]), 
            "pair_DeltaPhi": dijets["lead"].delta_phi(dijets["sublead"]),
            "pair_eta_prod": dijets.lead.eta * dijets.sublead.eta, 
            "pair_eta_diff": dijets["eta_diff"],
            "pair_Cgg": Cgg(dijets["eta_diff"], dijets["eta_sum"], dijets["gg_eta"]),
            }
        )  

        #* Process the format for prediction
        counts = ak.num(branches)
        flat_features = ak.flatten(branches[branch_features])

        #* Convert structured array to a flat array and do the normalization 
        DNN_Input = pd.DataFrame(flat_features.to_numpy(), columns=flat_features.fields)
        DNN_Input = sc.transform(DNN_Input)
        
        #* DNN prediction
        DNN_Score = model.predict(DNN_Input)

        #* pair-level info -> store to ROOT file
        flat_features["DNN_class"] = ak.argmax(DNN_Score, axis=1, mask_identity=False) 
        flat_features["DNN_bb_pair"] = DNN_Score[:,0]
        flat_features["DNN_jj_pair"] = DNN_Score[:,1]
        flat_features["DNN_wrong_pair"] = DNN_Score[:,2]
        
        with uproot.recreate("./update_minitree/{}/{}_PairLevel.root".format(version, i)) as new_file:
            new_file[TreeName] = flat_features

        #* event-level info
        DNN_VBF = ak.unflatten(DNN_Score, counts)

        branches["DNN_class"] = ak.argmax(DNN_VBF, axis=2, mask_identity=False)
        branches["DNN_bb_pair"] = DNN_VBF[:,:,0]
        branches["DNN_jj_pair"] = DNN_VBF[:,:,1]
        branches["DNN_wrong_pair"] = DNN_VBF[:,:,2]
        branches["lead"] = dijets["lead"]
        branches["sublead"] = dijets["sublead"]
        
        print("number of events: ", ak.num(branches, axis=0))
        print("total yeild: ", j*ak.sum(branches["weight"], axis=0)[0])
        print("total VBF yeild: ", j*ak.sum(branches["weight"][branches["selected"]==1], axis=0)[0])
        #* b-jet pair selection
        is_bb = (branches["DNN_class"] == 0) & (abs(branches["pair1_eta"]) < 2.5) & (abs(branches["pair2_eta"]) < 2.5)
        is_jj = (branches["DNN_class"] == 1) & (branches["pair1_pt"] > 40) & (branches["pair2_pt"] > 30)
        
        is_bbjj = (ak.sum(is_bb, axis=1) > 0) & (ak.sum(is_jj, axis=1) > 0)
        # print(is_bb, is_jj, is_bbjj)
        # print(ak.sum(ak.sum(is_bb, axis=1) > 0), ak.sum(ak.sum(is_jj, axis=1) > 0), ak.sum(is_bbjj))
        candidate_bbjj = branches[is_bbjj]
        candidate_bb = candidate_bbjj[candidate_bbjj["DNN_class"] == 0]
        candidate_bb = candidate_bb[ak.argsort(candidate_bb.pair1_btagPNetB+candidate_bb.pair2_btagPNetB, ascending=False)]
        pair_bb = ak.drop_none(ak.firsts(candidate_bb))
        print("number of bb pair: ", ak.num(pair_bb, axis=0))
        print("total yeild after bb: ", j*ak.sum(pair_bb["weight"], axis=0))

        candidate_jj = candidate_bbjj[candidate_bbjj["DNN_class"] == 1]
        candidate_jj = candidate_jj[ak.argsort(candidate_jj.pair_mass, ascending=False)]

        pair_bbjj = ak.cartesian({"bb": pair_bb, "jj":candidate_jj})
        
        deltaRCut = (pair_bbjj["bb"].lead.delta_r(pair_bbjj["jj"].lead) > 0.4) & (pair_bbjj["bb"].lead.delta_r(pair_bbjj["jj"].sublead) > 0.4) & (pair_bbjj["bb"].sublead.delta_r(pair_bbjj["jj"].lead) > 0.4) & (pair_bbjj["bb"].sublead.delta_r(pair_bbjj["jj"].sublead) > 0.4)
        idxCut = (pair_bbjj["bb"].lead.idx != pair_bbjj["jj"].lead.idx) & (pair_bbjj["bb"].sublead.idx != pair_bbjj["jj"].sublead.idx)
        
        results = ak.firsts(pair_bbjj[deltaRCut & idxCut])
        results = ak.drop_none(results)
        print(results["bb"].fields)
        print("number of bb+jj pair: ", ak.num(results, axis=0))
        print("total yeild after bb+jj: ", j*ak.sum(results["bb"]["weight"], axis=0))

        output = ak.zip({
            "event": results["bb"].event,
            "weight": results["bb"].weight,
            "pt": results["bb"].pt,
            "eta": results["bb"].eta, 
            "phi": results["bb"].phi, 
            "n_jets": results["bb"].n_jets
        })

        store_variables = [
            "pair1_pt", "pair2_pt", "pair1_ptOverM", "pair2_ptOverM", "pair1_eta", "pair2_eta", "pair1_phi", "pair2_phi", 
            "pair1_btagPNetB", "pair2_btagPNetB", "pair1_btagPNetQvG", "pair2_btagPNetQvG",
            "pair_pt", "pair_eta", "pair_phi", "pair_mass", "pair_DeltaR", "pair_DeltaPhi", 
            "pair_eta_prod", "pair_eta_diff", "pair_Cgg", "DNN_class", "DNN_bb_pair", "DNN_jj_pair", "DNN_wrong_pair",
        ]
    
        for jet in ["bb", "jj"]:
            for field in store_variables:
                output[jet+"_"+field] = results[jet][field]
            
        
        #* Store the selected pair to ROOT file
        with uproot.recreate("./update_minitree/{}/{}_EventLevel.root".format(version, i)) as new_file:
            new_file[TreeName] = output

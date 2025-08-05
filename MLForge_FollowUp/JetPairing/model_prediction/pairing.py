import vector
import pandas as pd
import awkward as ak
import numpy as np
import pickle
import tensorflow as tf
import uproot
import os

vector.register_awkward()

def Cxx(jet_eta_diff, jet_eta_sum, xx_eta):
    jet_eta_diff = np.where(jet_eta_diff == 0, 1e-10, jet_eta_diff)
    return np.exp(-4 / (jet_eta_diff) ** 2 * (xx_eta - (jet_eta_sum) / 2) ** 2)
    
def apply_VBFHH_pairing(events, model, sc):
    
    if "jet1_btagPNetQvG" not in events.columns:
        print("Warning: btagPNetQvG is missing in v2 parquet files. Setting it to -999.")
        for idx in range(1, 7): events.loc[:, f"jet{idx}_btagPNetQvG"] = -999

    #* Define jet collection
    jet1 = ak.zip({"pt": events["jet1_pt"], "eta": events["jet1_eta"], "phi": events["jet1_phi"], "mass": events["jet1_mass"], "charge": events["jet1_charge"], "btagPNetB": events["jet1_btagPNetB"], "btagPNetQvG": events["jet1_btagPNetQvG"]})
    jet2 = ak.zip({"pt": events["jet2_pt"], "eta": events["jet2_eta"], "phi": events["jet2_phi"], "mass": events["jet2_mass"], "charge": events["jet2_charge"], "btagPNetB": events["jet2_btagPNetB"], "btagPNetQvG": events["jet2_btagPNetQvG"]})
    jet3 = ak.zip({"pt": events["jet3_pt"], "eta": events["jet3_eta"], "phi": events["jet3_phi"], "mass": events["jet3_mass"], "charge": events["jet3_charge"], "btagPNetB": events["jet3_btagPNetB"], "btagPNetQvG": events["jet3_btagPNetQvG"]})
    jet4 = ak.zip({"pt": events["jet4_pt"], "eta": events["jet4_eta"], "phi": events["jet4_phi"], "mass": events["jet4_mass"], "charge": events["jet4_charge"], "btagPNetB": events["jet4_btagPNetB"], "btagPNetQvG": events["jet4_btagPNetQvG"]})
    jet5 = ak.zip({"pt": events["jet5_pt"], "eta": events["jet5_eta"], "phi": events["jet5_phi"], "mass": events["jet5_mass"], "charge": events["jet5_charge"], "btagPNetB": events["jet5_btagPNetB"], "btagPNetQvG": events["jet5_btagPNetQvG"]})
    jet6 = ak.zip({"pt": events["jet6_pt"], "eta": events["jet6_eta"], "phi": events["jet6_phi"], "mass": events["jet6_mass"], "charge": events["jet6_charge"], "btagPNetB": events["jet6_btagPNetB"], "btagPNetQvG": events["jet6_btagPNetQvG"]})

    jets = ak.concatenate([jet1[:,None], jet2[:,None], jet3[:,None], jet4[:,None], jet5[:,None], jet6[:,None]], axis=1)
    jets = ak.with_name(jets, "Momentum4D")
    jets = ak.drop_none(jets.mask[jets.pt > 0]) 
    jets["idx"] = ak.local_index(jets, axis=1)

    #* Define the jet pairs
    dijets = ak.combinations(jets, 2, fields=["lead", "sublead"])
    pair = dijets["lead"] + dijets["sublead"]
    
    dijets["mass"] = pair.mass
    dijets["pt"] = pair.pt
    dijets["eta"] = pair.eta
    dijets["phi"] = pair.phi
    dijets["charge"] = dijets["lead"].charge + dijets["sublead"].charge
    dijets["btagPNetB"] = dijets["lead"].btagPNetB + dijets["sublead"].btagPNetB
    dijets["btagPNetQvG"] = dijets["lead"].btagPNetQvG + dijets["sublead"].btagPNetQvG
    
    dijets["n_jets"] = events["n_jets"] 
    dijets["Hgg_eta"] = events["eta"]
    dijets["eta_diff"] = dijets["lead"].eta - dijets["sublead"].eta
    dijets["eta_sum"] = dijets["lead"].eta + dijets["sublead"].eta
    dijets["eta_prod"] = dijets["lead"].eta * dijets["sublead"].eta
    dijets["Cgg"] = Cxx(dijets["eta_diff"], dijets["eta_sum"], dijets["Hgg_eta"])
    
    input_features = ak.zip({
        "pair1_ptOverM": dijets["lead"].pt / dijets.mass,
        "pair2_ptOverM": dijets["sublead"].pt / dijets.mass,
        "pair1_eta": dijets["lead"].eta,
        "pair2_eta": dijets["sublead"].eta,
        "pair1_btagPNetB": dijets["lead"].btagPNetB,
        "pair2_btagPNetB": dijets["sublead"].btagPNetB,
        "pair1_btagPNetQvG": dijets["lead"].btagPNetQvG,
        "pair2_btagPNetQvG": dijets["sublead"].btagPNetQvG,
        "n_jets": dijets["n_jets"],
        "pair_ptOverM": dijets.pt / dijets.mass,
        "pair_eta": dijets.eta,
        "pair_DeltaR": vector.Spatial.deltaR(dijets["lead"], dijets["sublead"]),
        "pair_DeltaPhi": vector.Spatial.deltaphi(dijets["lead"], dijets["sublead"]),
        "pair_eta_prod": dijets["eta_prod"],
        "pair_eta_diff": dijets["eta_diff"],
        "pair_Cgg": dijets["Cgg"],
        "pair1_phi": dijets["lead"].phi,
        "pair2_phi": dijets["sublead"].phi
    })
    
    #* Process the format for prediction
    counts = ak.num(input_features)
    flat_features = ak.flatten(input_features)
    #* Convert structured array to a flat array and do the normalization
    DNN_Input = pd.DataFrame(flat_features.to_numpy(), columns=flat_features.fields)
    
    DNN_Input.loc[DNN_Input["pair1_btagPNetB"] < 0, "pair1_btagPNetB"] = -999
    DNN_Input.loc[DNN_Input["pair2_btagPNetB"] < 0, "pair2_btagPNetB"] = -999
    DNN_Input.loc[DNN_Input["pair1_btagPNetQvG"] < 0, "pair1_btagPNetQvG"] = -999
    DNN_Input.loc[DNN_Input["pair2_btagPNetQvG"] < 0, "pair2_btagPNetQvG"] = -999

    valid_mask = (DNN_Input != -999)
    DNN_Input = DNN_Input.astype(float)
    DNN_Input[valid_mask] = sc.transform(DNN_Input[valid_mask])
    DNN_Input[~valid_mask] = -1

    #* DNN prediction
    DNN_Score = model.predict(DNN_Input)
    DNN_VBF = ak.unflatten(DNN_Score, counts)

    dijets["DNN_class"] = ak.argmax(DNN_VBF, axis=2, mask_identity=False)
    dijets["DNN_bb_pair"] = DNN_VBF[:,:,0]
    dijets["DNN_jj_pair"] = DNN_VBF[:,:,1]
    dijets["DNN_wrong_pair"] = DNN_VBF[:,:,2]
    
    is_bb = (dijets["DNN_class"] == 0) & (abs(dijets.lead.eta) < 2.5) & (abs(dijets.sublead.eta) < 2.5) & (dijets.mass > 70) & (dijets.mass < 190)
    is_jj = (dijets["DNN_class"] == 1) & (dijets.lead.pt > 40) & (dijets.sublead.pt > 30)

    candidate_bb = dijets[is_bb]
    candidate_bb = candidate_bb[ak.argsort(candidate_bb.lead.btagPNetB+candidate_bb.sublead.btagPNetB, ascending=False)]     
    pair_bb = ak.firsts(candidate_bb)

    candidate_jj = dijets[is_jj]
    candidate_jj = candidate_jj[ak.argsort(candidate_jj.mass, ascending=False)] 

    deltaRCut = (
        (vector.Spatial.deltaR(pair_bb.lead, candidate_jj.lead) > 0.4) & (vector.Spatial.deltaR(pair_bb.lead, candidate_jj.sublead) > 0.4) &
        (vector.Spatial.deltaR(pair_bb.sublead, candidate_jj.lead) > 0.4) & (vector.Spatial.deltaR(pair_bb.sublead, candidate_jj.sublead) > 0.4)
    )
    idxCut = (pair_bb.lead.idx != candidate_jj.lead.idx) & (pair_bb.sublead.idx != candidate_jj.sublead.idx)
    
    pair_jj = ak.firsts(candidate_jj[deltaRCut & idxCut])
    
    mask = ~ak.is_none(pair_bb) & ~ak.is_none(pair_jj)
    
    jet_properties = ["pt", "eta", "phi", "mass", "charge", "btagPNetB", "btagPNetQvG"]

    for prop in jet_properties:
        events.loc[:, f"nonRes_dijet_{prop}_vbfpairing"] = ak.where(mask, pair_bb[prop], -999)
        events.loc[:, f"VBF_dijet_{prop}_vbfpairing"] = ak.where(mask, pair_jj[prop], -999)
        events.loc[:, f"nonRes_lead_bjet_{prop}_vbfpairing"] = ak.where(mask, pair_bb.lead[prop], -999)
        events.loc[:, f"nonRes_sublead_bjet_{prop}_vbfpairing"] = ak.where(mask, pair_bb.sublead[prop], -999)
        events.loc[:, f"VBF_first_jet_{prop}_vbfpairing"] = ak.where(mask, pair_jj.lead[prop], -999)
        events.loc[:, f"VBF_second_jet_{prop}_vbfpairing"] = ak.where(mask, pair_jj.sublead[prop], -999)

    events.loc[:, "VBF_jet_eta_prod_vbfpairing"] = ak.where(mask, pair_jj["eta_prod"], -999)
    events.loc[:, "VBF_Cgg_vbfpairing"] = ak.where(mask, pair_jj["Cgg"], -999)
    events.loc[:, "VBF_Cbb_vbfpairing"] = ak.where(mask, Cxx(pair_jj["eta_diff"], pair_jj["eta_sum"], pair_bb["eta"]), -999)
    
    lead_photon = ak.zip({
        "pt": events["lead_pt"],
        "eta": events["lead_eta"],
        "phi": events["lead_phi"],
        "mass": ak.zeros_like(events["lead_pt"]),
        "charge": ak.zeros_like(events["lead_pt"]),
    }, with_name="Momentum4D")
    
    sublead_photon = ak.zip({
        "pt": events["sublead_pt"],
        "eta": events["sublead_eta"],
        "phi": events["sublead_phi"],
        "mass": ak.zeros_like(events["sublead_pt"]),
        "charge": ak.zeros_like(events["sublead_pt"]),
    }, with_name="Momentum4D")
    
    DeltaR_j1g1 = ak.where(mask, vector.Spatial.deltaR(pair_jj.lead, lead_photon), -999)
    DeltaR_j1g2 = ak.where(mask, vector.Spatial.deltaR(pair_jj.lead, sublead_photon), -999)
    DeltaR_j2g1 = ak.where(mask, vector.Spatial.deltaR(pair_jj.sublead, lead_photon), -999)
    DeltaR_j2g2 = ak.where(mask, vector.Spatial.deltaR(pair_jj.sublead, sublead_photon), -999)
    DeltaR_j1b1 = ak.where(mask, vector.Spatial.deltaR(pair_jj.lead, pair_bb.lead), -999)
    DeltaR_j1b2 = ak.where(mask, vector.Spatial.deltaR(pair_jj.lead, pair_bb.sublead), -999)
    DeltaR_j2b1 = ak.where(mask, vector.Spatial.deltaR(pair_jj.sublead, pair_bb.lead), -999)
    DeltaR_j2b2 = ak.where(mask, vector.Spatial.deltaR(pair_jj.sublead, pair_bb.sublead), -999)

    DeltaR_jg = ak.Array([DeltaR_j1g1, DeltaR_j1g2, DeltaR_j2g1, DeltaR_j2g2])
    DeltaR_jb = ak.Array([DeltaR_j1b1, DeltaR_j1b2, DeltaR_j2b1, DeltaR_j2b2])

    events.loc[:, "VBF_DeltaR_jb_min"] = ak.min(DeltaR_jb, axis=0)
    events.loc[:, "VBF_DeltaR_jg_min"] = ak.min(DeltaR_jg, axis=0)

    return events

if __name__ == "__main__":
    model = tf.keras.models.load_model("/home/cosine/HHbbgg/MLForge/results/Pairing_vbf_v3_6/DNN/DNN_modelDNN.keras")
    with open("/home/cosine/HHbbgg/MLForge/results/Pairing_vbf_v3_6/DNN/DNN_scaler.pkl", 'rb') as f:
        sc = pickle.load(f)
    
    events = pd.read_parquet(
        "/home/cosine/HHbbgg/minitree_parquet/ver0518/mc/merged/22preEE_VBFToHH/NOTAG_merged.parquet", # columns=new_columns, filters=filters
    ) 
    events = apply_VBFHH_pairing(events, model, sc)
    
    directory = "/home/cosine/HHbbgg/minitree_parquet/ver0518_jetpairing/mc/merged/22preEE_VBFToHH"
    if not os.path.exists(directory):
        os.makedirs(directory)

    events.to_parquet("/home/cosine/HHbbgg/minitree_parquet/ver0518_jetpairing/mc/merged/22preEE_VBFToHH/NOTAG_merged.parquet")
    # with uproot.recreate(f"./samples/22preEE_VBFToHH.root") as new_file:

    #     new_file["DiphotonTree/data_125_13TeV_NOTAG"] = events

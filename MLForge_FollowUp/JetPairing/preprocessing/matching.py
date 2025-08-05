import pandas as pd
import numpy as np
import awkward as ak
import uproot
from coffea.nanoevents.methods import candidate

def get_match_jet(reco_jets, gen_jets, n, fill_value, btype="isMatch"):
    """
    this helper function is used to identify if a reco jet (or lepton) has a matching gen jet (lepton) for MC,
    -> Returns an array with 3 possible values:
        0 if reco not genMatched,
        1 if reco genMatched,
        -999 if reco doesn't exist
    parameters:
    * reco_jets: (ak array) reco_jet from the jets collection.
    * gen_jets: (ak array) gen_jet from the events.GenJet (or equivalent, e.g. events.Electron) collection.
    * n: (int) nth jet to be selected.
    * fill_value: (float) value with wich to fill the padded none if nth jet doesnt exist in the event.
    """
    if n is not None:
        reco_jets_i = reco_jets[ak.local_index(reco_jets, axis=1) == n]
    else:
        # This is for arrays already split into separate event slices, used for Higgs an bjet matching.
        reco_jets_i = ak.singletons(reco_jets)
    reco_jets_i = ak.pad_none(reco_jets_i, 1, clip=True)

    candidate_jet_matches = ak.cartesian({"reco": reco_jets_i, "gen": gen_jets}, axis=1)

    is_matched_idx = candidate_jet_matches["reco"].idx == candidate_jet_matches["gen"].reco_idx

    matched_jets = ak.firsts(
        candidate_jet_matches[is_matched_idx], axis=1
    )

    matched_jets_bool = ~ak.is_none(matched_jets["gen"].reco_idx)
    
    if btype == "isMatch":
        matched_jets_status = ak.where(
            ~ak.is_none(ak.firsts(reco_jets_i)), matched_jets_bool, fill_value
        )
        return matched_jets_status
    else: 
        matched_flav = ak.where(
            matched_jets_bool, getattr(matched_jets["gen"], btype), fill_value
        )
        matched_flav = ak.where(
            ~ak.is_none(ak.firsts(reco_jets_i)), matched_flav, fill_value
        )
        return matched_flav
    
def match_jet_idx(reco_jets, gen_jets, jet_size=0.4):

    candidate = ak.cartesian({"gen":gen_jets, "reco":reco_jets}, axis=1, nested=True)
    candidate = ak.with_name(candidate, "PtEtaPhiMCandidate", behavior=candidate.behavior)

    candidate["deltaR_jj"] = candidate["gen"].delta_r(candidate["reco"])
    candidate["delPtRel_jj"] = abs(candidate["reco"].pt-candidate["gen"].pt)/candidate["gen"].pt

    candidate = candidate[ak.argsort(candidate["delPtRel_jj"], ascending=True)]
    candidate = candidate[candidate["deltaR_jj"] < jet_size]

    matched_jets = ak.firsts(candidate, axis=2)

    match_idx = ak.where(
        ~ak.is_none(matched_jets["reco"].idx, axis=1), matched_jets["reco"].idx, -999
    )
    return match_idx

if __name__ == "__main__":
    #* Load the ROOT files
    events = uproot.concatenate("../../../Shared_file/parquet/22/22postEE_VBFHHto2B2G_CV_1_C2V_1_C3_1.root:DiphotonTree/data_125_13TeV_NOTAG", library="ak")
    # events = events[:10]
    jet1 = ak.zip({"pt": events["jet1_pt"], "eta": events["jet1_eta"], "phi": events["jet1_phi"], "mass": events["jet1_mass"], "charge": events["jet1_charge"]})
    jet2 = ak.zip({"pt": events["jet2_pt"], "eta": events["jet2_eta"], "phi": events["jet2_phi"], "mass": events["jet2_mass"], "charge": events["jet2_charge"]})
    jet3 = ak.zip({"pt": events["jet3_pt"], "eta": events["jet3_eta"], "phi": events["jet3_phi"], "mass": events["jet3_mass"], "charge": events["jet3_charge"]})
    jet4 = ak.zip({"pt": events["jet4_pt"], "eta": events["jet4_eta"], "phi": events["jet4_phi"], "mass": events["jet4_mass"], "charge": events["jet4_charge"]})
    jet5 = ak.zip({"pt": events["jet5_pt"], "eta": events["jet5_eta"], "phi": events["jet5_phi"], "mass": events["jet5_mass"], "charge": events["jet5_charge"]})
    jet6 = ak.zip({"pt": events["jet6_pt"], "eta": events["jet6_eta"], "phi": events["jet6_phi"], "mass": events["jet6_mass"], "charge": events["jet6_charge"]})
    jet7 = ak.zip({"pt": events["jet7_pt"], "eta": events["jet7_eta"], "phi": events["jet7_phi"], "mass": events["jet7_mass"], "charge": events["jet7_charge"]})
    jet8 = ak.zip({"pt": events["jet8_pt"], "eta": events["jet8_eta"], "phi": events["jet8_phi"], "mass": events["jet8_mass"], "charge": events["jet8_charge"]})
    jet9 = ak.zip({"pt": events["jet9_pt"], "eta": events["jet9_eta"], "phi": events["jet9_phi"], "mass": events["jet9_mass"], "charge": events["jet9_charge"]})
    jet10 = ak.zip({"pt": events["jet10_pt"], "eta": events["jet10_eta"], "phi": events["jet10_phi"], "mass": events["jet10_mass"], "charge": events["jet10_charge"]})
    
    jets = ak.concatenate([jet1[:,None], jet2[:,None], jet3[:,None], jet4[:,None], jet5[:,None], jet6[:,None], jet7[:,None], jet8[:,None], jet9[:,None], jet10[:,None]], axis=1)
    jets = ak.with_name(jets, "PtEtaPhiMCandidate", behavior=candidate.behavior)

    jets = ak.drop_none(jets.mask[jets.pt > 0]) 
    jets["idx"] = ak.local_index(jets, axis=1)

    lhe_parton1 = ak.zip({"pt": events["lhe_vbf_parton1_pt"], "eta": events["lhe_vbf_parton1_eta"], "phi": events["lhe_vbf_parton1_phi"], "mass": events["lhe_vbf_parton1_mass"]})    
    lhe_parton2 = ak.zip({"pt": events["lhe_vbf_parton2_pt"], "eta": events["lhe_vbf_parton2_eta"], "phi": events["lhe_vbf_parton2_phi"], "mass": events["lhe_vbf_parton2_mass"]})
    lhe_parton1 = ak.with_name(lhe_parton1, "PtEtaPhiMCandidate", behavior=candidate.behavior)
    lhe_parton2 = ak.with_name(lhe_parton2, "PtEtaPhiMCandidate", behavior=candidate.behavior)
    lhe_parton = ak.concatenate([lhe_parton1[:,None], lhe_parton2[:,None]], axis=1)
    lhe_parton = ak.with_name(lhe_parton, "PtEtaPhiMCandidate", behavior=candidate.behavior)
    lhe_parton["reco_idx"] = match_jet_idx(jets, lhe_parton)
    
    for i in range(10):
        # LHE parton matching
        events[f"jet{i+1}_lheMatched"] = get_match_jet(jets, lhe_parton, i, -999.0, "isMatch") 
        # print(events[f"jet{i+1}_lheMatched"])
    events["lheMatched"] = ak.sum(lhe_parton["reco_idx"] > 0, axis=1)   
    # print(ak.sum(lhe_parton["reco_idx"] > 0, axis=1))

    with uproot.recreate("../samples/22postEE_VBFHHto2B2G_CV_1_C2V_1_C3_1.root") as new_file:
        new_file["DiphotonTree/data_125_13TeV_NOTAG"] = events

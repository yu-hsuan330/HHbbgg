lumi = {
    "22preEE": 7980.4, 
    "22postEE": 26671.7
}
model_param = {
    "output": "EventCate_Nitish",
    "path": "/home/cosine/HHbbgg/MLForge/results/EventCate_0301/DNN/DNN_modelDNN.keras",
    "scaler": "/home/cosine/HHbbgg/MLForge/results/EventCate_0301/DNN/DNN_scaler.pkl",
    "features": [
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
    ], 
}

TreeName_mc = "DiphotonTree/data_125_13TeV_NOTAG"
TreeName_data = "DiphotonTree/Data_13TeV_NOTAG"

sample_set = "/home/cosine/HHbbgg/MLForge_FollowUp/samples/Pairing_Nitishi"
samples = [
    {
        "name": "GluGluToHH",
        "xs": 0.03443*(0.00227*0.582*2),
        "22preEE": [f"{sample_set}/22preEE_GluGluToHH.root:{TreeName_mc}"], 
        "22postEE": [f"{sample_set}/22postEE_GluGluToHH.root:{TreeName_mc}"]
    },
    {
        "name": "VBFToHH",
        "xs": 0.00192*(0.00227*0.582*2), 
        "22preEE": [f"{sample_set}/22preEE_VBFToHH.root:{TreeName_mc}"], 
        "22postEE": [f"{sample_set}/22postEE_VBFToHH.root:{TreeName_mc}"]
    },
    # {
    #     "name": "GJet_Pt20to40_MGG80",
    #     "xs": 242.5,
    #     "22preEE": [f"{sample_set}/22preEE_GJet_Pt20to40_MGG80.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set}/22postEE_GJet_Pt20to40_MGG80.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "GJet_Pt40_MGG80",
    #     "xs": 919.1,   
    #     "22preEE": [f"{sample_set}/22preEE_GJet_Pt40_MGG80.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set}/22postEE_GJet_Pt40_MGG80.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "GGJets",
    #     "xs": 88.75,
    #     "22preEE": [f"{sample_set}/22preEE_GGJets.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set}/22postEE_GGJets.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "QCD_Pt30to40_MGG80",
    #     "xs": 25950,
    #     "22preEE": [f"{sample_set}/22preEE_QCD_Pt30to40_MGG80.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set}/22postEE_QCD_Pt30to40_MGG80.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "QCD_Pt40_MGG80",
    #     "xs": 124700,
    #     "22preEE": [f"{sample_set}/22preEE_QCD_Pt40_MGG80.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set}/22postEE_QCD_Pt40_MGG80.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "QCD_Pt30_MGG40to80",
    #     "xs": 252200,
    #     "22preEE": [f"{sample_set}/22preEE_QCD_Pt30_MGG40to80.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set}/22postEE_QCD_Pt30_MGG40to80.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "data",
    #     "xs": 1,
    #     "22preEE": [f"{sample_set}/22preEE_data.root:{TreeName_data}"],
    #     "22postEE": [f"{sample_set}/22postEE_data.root:{TreeName_data}"]
    # } 
]
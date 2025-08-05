lumi = {
    "22preEE": 7980.4, 
    "22postEE": 26671.7
}
model_param = {
    "output": "EventCate_0402",
    "path": "/home/cosine/HHbbgg/MLForge/results/EventCate_0301/DNN/DNN_modelDNN.keras",
    "scaler": "/home/cosine/HHbbgg/MLForge/results/EventCate_0301/DNN/DNN_scaler.pkl",
    "features": [
        # photon info
        "nonRes_pholead_PtOverM",       "nonRes_phosublead_PtOverM",       "lead_eta",                         "sublead_eta",
        "diphoton_ptOverM",             "eta",                              "phi",                              "n_jets",
        # b-jet info
        "nonRes_FirstJet_PtOverM",      "nonRes_SecondJet_PtOverM",         "nonRes_lead_bjet_eta",             "nonRes_sublead_bjet_eta",
        "nonRes_lead_bjet_btagPNetB",   "nonRes_sublead_bjet_btagPNetB",    "nonRes_lead_bjet_btagPNetQvG",     "nonRes_sublead_bjet_btagPNetQvG",
        "bjet_pair_ptOverM",            "nonRes_dijet_eta",                 "nonRes_dijet_phi",                 "nonRes_dijet_Cgg",
        "nonRes_dijet_prodEta",         "nonRes_dijet_deltaEta",            "nonRes_dijet_deltaR",              "nonRes_dijet_deltaPhi", 
        # vbf-jet info
        "VBF_first_jet_PtOverM",         "VBF_second_jet_PtOverM",          "VBF_first_jet_eta",                 "VBF_second_jet_eta", 
        "VBF_first_jet_btagPNetB",       "VBF_second_jet_btagPNetB",        "VBF_first_jet_btagPNetQvG",         "VBF_second_jet_btagPNetQvG",
        "vbfjet_pair_ptOverM",          "VBF_dijet_eta",                    "VBF_dijet_phi",                    "VBF_dijet_mass", 
        "VBF_jet_eta_prod",             "VBF_jet_eta_diff",                 "VBF_dijet_deltaR",                 "VBF_dijet_deltaPhi",         
        "VBF_Cgg",                      "VBF_Cbb" 
    ], 
}

TreeName_mc = "DiphotonTree/data_125_13TeV_NOTAG"
TreeName_data = "DiphotonTree/Data_13TeV_NOTAG"

# sample_set = "/home/cosine/HHbbgg/MLForge_FollowUp/samples/sample_from_Nitishi_add_v2"
sample_set = "/home/cosine/HHbbgg/minitree/ver0402_v3"
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
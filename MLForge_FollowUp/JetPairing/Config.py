lumi = {
    "22preEE": 7980.4, 
    "22postEE": 26671.7
}

model_param = {
    "output": "Pairing_vbf_0301_",
    "path": "/home/cosine/HHbbgg/MLForge/results/Pairing_vbf_0301/DNN/DNN_modelDNN.keras",
    "scaler": "/home/cosine/HHbbgg/MLForge/results/Pairing_vbf_0301/DNN/DNN_scaler.pkl",
    "features": [
        "lead_ptOverM", "sublead_ptOverM", "lead_eta", "sublead_eta", 
        "lead_btagPNetB", "sublead_btagPNetB", "lead_btagPNetQvG", "sublead_btagPNetQvG",
        "n_jets","pair_pt","pair_eta", "pair_mass",
        "pair_DeltaR", "pair_DeltaPhi", "pair_eta_prod", "pair_eta_diff", 
        "pair_Cgg", "lead_phi", "sublead_phi"
    ], 
}
TreeName_mc = "DiphotonTree/data_125_13TeV_NOTAG"
TreeName_data = "DiphotonTree/Data_13TeV_NOTAG"

sample_set = "/home/cosine/HHbbgg/minitree/ver0121"
samples = [
    # {
    #     "name": "GluGluToHH",
    #     "xs": 0.03443*(0.00227*0.582*2),
    #     "22preEE": [f"{sample_set}/22preEE_GluGluToHH.root:{TreeName_mc}"], 
    #     "22postEE": [f"{sample_set}/22postEE_GluGluToHH.root:{TreeName_mc}"]
    # },
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
    #     "22preEE": [f"{sample_set}/data/22preEE_EGammaC.root:{TreeName_data}", f"{sample_set}/data/22preEE_EGammaD.root:{TreeName_data}"],
    #     "22postEE": [f"{sample_set}/data/22postEE_EGammaE.root:{TreeName_data}", f"{sample_set}/data/22postEE_EGammaF.root:{TreeName_data}", f"{sample_set}/data/22postEE_EGammaG.root:{TreeName_data}"]
    # } 
]
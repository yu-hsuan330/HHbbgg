TreeName_mc = "DiphotonTree/data_125_13TeV_NOTAG"
TreeName_data = "DiphotonTree/Data_13TeV_NOTAG"

sample_set_1 = "/home/cosine/HHbbgg/MLForge_FollowUp/samples/sample_from_Nitishi"
sample_set_2 = "/home/cosine/HHbbgg/minitree/ver0121"

samples_1 = [
    {
        "name": "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00",
        "xs": 0.03443*(0.00227*0.582*2),
        "22preEE": [f"{sample_set_1}/22preEE_GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00.root:{TreeName_mc}"], 
        "22postEE": [f"{sample_set_1}/22postEE_GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00.root:{TreeName_mc}"]
    },
    {
        "name": "VBFHHto2B2G_CV_1_C2V_1_C3_1",
        "xs": 0.00192*(0.00227*0.582*2), 
        "22preEE": [f"{sample_set_1}/22preEE_VBFHHto2B2G_CV_1_C2V_1_C3_1.root:{TreeName_mc}"], 
        "22postEE": [f"{sample_set_1}/22postEE_VBFHHto2B2G_CV_1_C2V_1_C3_1.root:{TreeName_mc}"]
    },
    # {
    #     "name": "GJet_Pt20to40",
    #     "xs": 242.5,
    #     "22preEE": [f"{sample_set_1}/22preEE_GJet_Pt20to40.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set_1}/22postEE_GJet_Pt20to40.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "GJet_Pt40",
    #     "xs": 919.1,   
    #     "22preEE": [f"{sample_set_1}/22preEE_GJet_Pt40.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set_1}/22postEE_GJet_Pt40.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "GGJets",
    #     "xs": 88.75,
    #     "22preEE": [f"{sample_set_1}/22preEE_GGJets.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set_1}/22postEE_GGJets.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "QCD_Pt30to40_MGG80",
    #     "xs": 25950,
    #     "22preEE": [f"{sample_set_1}/22preEE_QCD_Pt30to40_MGG80.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set_1}/22postEE_QCD_Pt30to40_MGG80.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "QCD_Pt40_MGG80",
    #     "xs": 124700,
    #     "22preEE": [f"{sample_set_1}/22preEE_QCD_Pt40_MGG80.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set_1}/22postEE_QCD_Pt40_MGG80.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "QCD_Pt30_MGG40to80",
    #     "xs": 252200,
    #     "22preEE": [f"{sample_set_1}/22preEE_QCD_Pt30_MGG40to80.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set_1}/22postEE_QCD_Pt30_MGG40to80.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "data",
    #     "xs": 1,
    #     "22preEE": [f"{sample_set_1}/data/22preEE_EGammaC.root:{TreeName_data}", f"{sample_set_1}/data/22preEE_EGammaD.root:{TreeName_data}"],
    #     "22postEE": [f"{sample_set_1}/data/22postEE_EGammaE.root:{TreeName_data}", f"{sample_set_1}/data/22postEE_EGammaF.root:{TreeName_data}", f"{sample_set_1}/data/22postEE_EGammaG.root:{TreeName_data}"]
    # } 
]

samples_2 = [
    {
        "name": "GluGluToHH",
        "xs": 0.03443*(0.00227*0.582*2),
        "22preEE": [f"{sample_set_2}/22preEE_GluGluToHH.root:{TreeName_mc}"], 
        "22postEE": [f"{sample_set_2}/22postEE_GluGluToHH.root:{TreeName_mc}"]
    },
    {
        "name": "VBFToHH",
        "xs": 0.00192*(0.00227*0.582*2), 
        "22preEE": [f"{sample_set_2}/22preEE_VBFToHH.root:{TreeName_mc}"], 
        "22postEE": [f"{sample_set_2}/22postEE_VBFToHH.root:{TreeName_mc}"]
    },
    # {
    #     "name": "GJet_Pt20to40_MGG80",
    #     "xs": 242.5,
    #     "22preEE": [f"{sample_set_2}/22preEE_GJet_Pt20to40_MGG80.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set_2}/22postEE_GJet_Pt20to40_MGG80.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "GJet_Pt40_MGG80",
    #     "xs": 919.1,   
    #     "22preEE": [f"{sample_set_2}/22preEE_GJet_Pt40_MGG80.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set_2}/22postEE_GJet_Pt40_MGG80.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "GGJets",
    #     "xs": 88.75,
    #     "22preEE": [f"{sample_set_2}/22preEE_GGJets.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set_2}/22postEE_GGJets.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "QCD_Pt30to40_MGG80",
    #     "xs": 25950,
    #     "22preEE": [f"{sample_set_2}/22preEE_QCD_Pt30to40_MGG80.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set_2}/22postEE_QCD_Pt30to40_MGG80.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "QCD_Pt40_MGG80",
    #     "xs": 124700,
    #     "22preEE": [f"{sample_set_2}/22preEE_QCD_Pt40_MGG80.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set_2}/22postEE_QCD_Pt40_MGG80.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "QCD_Pt30_MGG40to80",
    #     "xs": 252200,
    #     "22preEE": [f"{sample_set_2}/22preEE_QCD_Pt30_MGG40to80.root:{TreeName_mc}"],
    #     "22postEE": [f"{sample_set_2}/22postEE_QCD_Pt30_MGG40to80.root:{TreeName_mc}"]
    # },
    # {
    #     "name": "data",
    #     "xs": 1,
    #     "22preEE": [f"{sample_set_2}/data/22preEE_EGammaC.root:{TreeName_data}", f"{sample_set_2}/data/22preEE_EGammaD.root:{TreeName_data}"],
    #     "22postEE": [f"{sample_set_2}/data/22postEE_EGammaE.root:{TreeName_data}", f"{sample_set_2}/data/22postEE_EGammaF.root:{TreeName_data}", f"{sample_set_2}/data/22postEE_EGammaG.root:{TreeName_data}"]
    # } 
]
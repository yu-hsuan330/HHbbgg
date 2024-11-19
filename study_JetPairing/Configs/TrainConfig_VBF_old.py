#=============================================#
#           training configurations           #
#              ggF pairing study              #
#=============================================# 
import numpy as np

OutputDirName = "MultiClass_XGB_VBF" # All plots, models, config file will be stored here
Debug, MVAlogplot = False, False

# ROOT files
Tree = "DiphotonTree/data_125_13TeV_NOTAG"
Branches = [
    "pair1_pt", "pair2_pt", "pair1_ptOverM", "pair2_ptOverM", "pair1_eta", "pair2_eta", "pair1_phi", "pair2_phi", 
    "pair1_btagPNetB", "pair2_btagPNetB", "pair1_btagPNetQvG", "pair2_btagPNetQvG",
    "pair_pt", "pair_eta", "pair_phi", "pair_mass", "pair_DeltaR", "pair_eta_prod", "pair_eta_diff", "pair_Cgg",
    "weight", "pt", "eta", "phi", "n_jets",
    "lead_pt", "lead_phi", "lead_eta", "sublead_eta", "lead_mvaID", "sublead_mvaID",
    "pholead_PtOverM", "phosublead_PtOverM", "FirstJet_PtOverM", "SecondJet_PtOverM",
    "lead_bjet_pt", "lead_bjet_eta", "lead_bjet_phi", "lead_bjet_btagPNetB", 
    "sublead_bjet_pt", "sublead_bjet_eta", "sublead_bjet_phi", "sublead_bjet_btagPNetB",
    "dijet_mass", "dijet_eta", "DeltaR_jg_min",
    "absCosThetaStar_CS", "absCosThetaStar_gg", "absCosThetaStar_jj",
]

# Training parameters D7263D
Classes, ClassColors = ["b-jet pair", "VBF-jet pair", "wrong pair"], ["#22577E", "#EF5B5B", "#F97D10"]
# Classes, ClassColors = ["ggF", "bkg"], ["#EF5B5B", "#22577E"]
TestSize, RandomState = 0.5, 123

Processes = [
    {
    "Class":"b-jet pair",
    "path":["../minitree/ver0830_VBFpair/22*_VBFToHH*.root"],
    "selection": "(b_jetPair == 1)",
    "input_weight": 1,#("weight"),
    },
    {
    "Class":"VBF-jet pair",
    "path":["../minitree/ver0830_VBFpair/22*_VBFToHH*.root"],
    "selection": "(vbf_jetPair == 1)",
    "input_weight": 1,#("weight"),
    },
    {
    "Class":"wrong pair",
    "path":["../minitree/ver0830_VBFpair/22*_VBFToHH*.root"],
    "selection": "(WrongPair == 1)",
    "input_weight": 1,#("weight"),
    },
]

MVAs = [
    {
    "MVAtype":"test", 
    "Label":"no-weight",
    "addwei":"nowei",
    "features":[
        "pt", "eta", "phi", "n_jets", 
        "pair1_ptOverM", "pair2_ptOverM", "pair1_eta", "pair2_eta", 
        "pair1_btagPNetB", "pair2_btagPNetB", "pair1_btagPNetQvG", "pair2_btagPNetQvG",
        "pair_pt","pair_eta", "pair_phi", "pair_mass",
        "pair_DeltaR", "pair_eta_prod", "pair_eta_diff", "pair_Cgg",
        "pair1_phi", "pair2_phi"
    ], # "FirstJet_PtOverM", "SecondJet_PtOverM", "absCosThetaStar_CS", "absCosThetaStar_gg", "absCosThetaStar_jj"
    "features_unit":[
        "$p_{T}^{\gamma\gamma}$", "$\eta_{\gamma\gamma}$", "$\\Phi_{\gamma\gamma}$", "$N_{jets}$", 
        "leading $p_{T}^{j}/M_{jj}$", "subleading $p_{T}^{j}/M_{jj}$", "leading $\eta_{j}$", "subleading $\eta_{j}$", 
        "leading jet PNetB", "subleading jet PNetB", "leading jet PNetQvG", "subleading jet PNetQvG", 
        "$p_{T}^{jj}$", "$\eta_{jj}$", "$\Phi_{jj}$", "$M_{jj} (GeV)$",
        "$\Delta R(j_1, j_2)$", "$\eta_{j_1} X \eta_{j_2}$", "$|\Delta\eta(j_1, j_2)|$", "$C_{\gamma\gamma}$",
        "leading $\Phi_{j}$", "subleading $\Phi_{j}$"
    ],
    "feature_bins":[
        np.linspace(0, 500, 31), np.linspace(-5, 5, 31), 15, np.linspace(2, 10, 9), 
        np.linspace(0, 2.5, 31), np.linspace(0, 1.5, 31), 15, 15, 
        np.linspace(0, 1, 21), np.linspace(0, 1, 21), np.linspace(0, 1, 21), np.linspace(0, 1, 21),
        np.linspace(0, 500, 31), np.linspace(-5, 5, 31), 15, np.linspace(0, 3000, 31),
        31, 31, 31, 31,
        15, 15      
    ],
    "hyperopt":False,
    "ModelParams":{"tree_method": "hist", "device": "cuda", "objective": "multi:softprob", "num_class": 3,  "eval_metric": "merror", "random_state": RandomState},
    "HyperParams":{"max_depth": 2, "learning_rate": 0.1, "min_child_weight": 12.0, "min_split_loss": 0.0, "subsample": 0.7000000000000001},
    }
]

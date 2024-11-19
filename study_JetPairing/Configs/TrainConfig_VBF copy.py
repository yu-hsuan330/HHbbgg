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
    "weight", "pt", "eta", "phi", "n_jets",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_mass", "jet1_btagPNetB", "jet2_pt", "jet2_eta", "jet2_phi", "jet2_mass", "jet2_btagPNetB", 
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_mass", "jet3_btagPNetB", "jet4_pt", "jet4_eta", "jet4_phi", "jet4_mass", "jet4_btagPNetB",
    "jet5_pt", "jet5_eta", "jet5_phi", "jet5_mass", "jet5_btagPNetB", "jet6_pt", "jet6_eta", "jet6_phi", "jet6_mass", "jet6_btagPNetB",
    "lead_pt", "lead_phi", "lead_eta", "sublead_eta", "lead_mvaID", "sublead_mvaID",
    "pholead_PtOverM", "phosublead_PtOverM", "FirstJet_PtOverM", "SecondJet_PtOverM",
    "lead_bjet_pt", "lead_bjet_eta", "lead_bjet_phi", "lead_bjet_btagPNetB", 
    "sublead_bjet_pt", "sublead_bjet_eta", "sublead_bjet_phi", "sublead_bjet_btagPNetB",
    "dijet_mass", "dijet_eta", "DeltaR_jg_min",
    "absCosThetaStar_CS", "absCosThetaStar_gg", "absCosThetaStar_jj",
    "VBF_lead_jet_lheMatched", "VBF_sublead_jet_lheMatched",
    "VBF_lead_jet_pt", "VBF_lead_jet_eta", "VBF_lead_jet_phi", "VBF_lead_jet_btagPNetB",
    "VBF_sublead_jet_pt", "VBF_sublead_jet_eta", "VBF_sublead_jet_phi", "VBF_sublead_jet_btagPNetB",
    "jet1_selected_vbfjet", "jet2_selected_vbfjet", "jet3_selected_vbfjet", "jet4_selected_vbfjet", "jet5_selected_vbfjet", "jet6_selected_vbfjet"
]
Branches_custom = [
    "pt", "eta", "phi", "btagPNetB"
]
# Training parameters D7263D
Classes, ClassColors = ["b-jet pair", "VBF-jet pair", "wrong pair"], ["#22577E", "#EF5B5B", "#F97D10"]
# Classes, ClassColors = ["ggF", "bkg"], ["#EF5B5B", "#22577E"]
TestSize, RandomState = 0.5, 123

Processes = [
    {
    "Class":"b-jet pair",
    "path":["../minitree/ver0823/22*_VBFToHH.root"],
    "selection": "(lead_bjet_genMatched == 1) & (sublead_bjet_genMatched == 1)",
    "custom":[("pair1", "lead_bjet"), ("pair2", "sublead_bjet")],
    "input_weight": 1,#("weight"),
    },
    {
    "Class":"VBF-jet pair",
    "path":["../minitree/ver0823/22*_VBFToHH.root"],
    "selection":"(VBF_lead_jet_lheMatched == 1) & (VBF_sublead_jet_lheMatched == 1)", 
    "custom":[("pair1", "VBF_lead_jet"), ("pair2", "VBF_sublead_jet")],
    "input_weight": 1,#("weight"), 
    },
    {
    "Class":"wrong pair",
    "path":["../minitree/ver0823/22*_VBFToHH.root"],
    "selection":"(jet1_pt > 0) & (jet2_pt > 0) & ((abs(jet1_genFlav) != 5) | (abs(jet2_genFlav) != 5)) & ( (jet1_selected_vbfjet == 0) | (jet2_selected_vbfjet == 0))", 
    "custom":[("pair1", "jet1"), ("pair2", "jet2")],
    "input_weight": 1,#("weight"), 
    },
    {
    "Class":"wrong pair",
    "path":["../minitree/ver0823/22*_VBFToHH.root"],
    "selection":"(jet1_pt > 0) & (jet3_pt > 0) & ((abs(jet1_genFlav) != 5) | (abs(jet3_genFlav) != 5)) & ( (jet1_selected_vbfjet == 0) | (jet3_selected_vbfjet == 0))", 
    "custom":[("pair1", "jet1"), ("pair2", "jet3")],
    "input_weight": 1,#("weight"), 
    },
    {
    "Class":"wrong pair",
    "path":["../minitree/ver0823/22*_VBFToHH.root"],
    "selection":"(jet1_pt > 0) & (jet4_pt > 0) & ((abs(jet1_genFlav) != 5) | (abs(jet4_genFlav) != 5)) & ( (jet1_selected_vbfjet == 0) | (jet4_selected_vbfjet == 0))", 
    "custom":[("pair1", "jet1"), ("pair2", "jet4")],
    "input_weight": 1,#("weight"), 
    },
]

MVAs = [
    {
    "MVAtype":"test", 
    "Label":"no-weight",
    "addwei":"nowei",
    "features":["pt", "eta", "phi", "n_jets", "pair1_pt", "pair2_pt", "pair1_eta", "pair2_eta", "pair1_phi", "pair2_phi", "pair1_btagPNetB", "pair2_btagPNetB"], # "FirstJet_PtOverM", "SecondJet_PtOverM", "absCosThetaStar_CS", "absCosThetaStar_gg", "absCosThetaStar_jj"
    "features_unit":["$p_{T}^{\gamma\gamma}$", "$\eta^{\gamma\gamma}$", "$\\Phi^{\gamma\gamma}$", "$N_{jets}$", "leading $p_{T}^{b}$", "subleading $p_{T}^{b}$", "leading $\eta_{b}$", "subleading $\eta_{b}$", "leading $\Phi_{b}$", "subleading $\Phi_{b}$", "leading b btagPNetB", "subleading b btagPNetB"],
    "feature_bins":[np.linspace(0, 500, 31), np.linspace(-5, 5, 31), 15, np.linspace(2, 10, 9), np.linspace(20, 350, 31), np.linspace(20, 120, 31), 15, 15, 15, 15, 15, 15, 15, 15, 31, 31],
    "hyperopt":True,
    "ModelParams":{"tree_method": "hist", "device": "cuda", "objective": "multi:softprob", "num_class": 3,  "eval_metric": "merror", "random_state": RandomState},
    "HyperParams":{"max_depth": 1, "learning_rate": 0.1, "min_child_weight": 29.0, "min_split_loss": 1.6, "subsample": 0.65},
    }
]
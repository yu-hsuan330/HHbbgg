#=============================================#
#           training configurations           #
#           VBF pairing study                 #
#=============================================# 
import numpy as np

OutputDirName = "MultiClass_XGB_test" # All plots, models, config file will be stored here
Debug, MVAlogplot = False, False

# ROOT files
Tree = "DiphotonTree/data_125_13TeV_NOTAG"
Branches = [
    "weight", "pt", "eta", "phi", "n_jets",
    "lead_pt", "lead_phi", "lead_eta", "sublead_eta", "lead_mvaID", "sublead_mvaID",
    "pholead_PtOverM", "phosublead_PtOverM", "FirstJet_PtOverM", "SecondJet_PtOverM",
    "lead_bjet_pt", "lead_bjet_eta", "lead_bjet_phi", "lead_bjet_btagPNetB", 
    "sublead_bjet_pt", "sublead_bjet_eta", "sublead_bjet_phi", "sublead_bjet_btagPNetB",
    "dijet_mass", "dijet_eta", "DeltaR_jg_min",
    "absCosThetaStar_CS", "absCosThetaStar_gg", "absCosThetaStar_jj",

]

# Training parameters D7263D
Classes, ClassColors = ["ggF", "VBF", "bkg"], ["#22577E", "#EF5B5B", "#F97D10"]
# Classes, ClassColors = ["ggF", "bkg"], ["#EF5B5B", "#22577E"]
TestSize, RandomState = 0.5, 123

Processes = [
    {
    "Class":"ggF",
    "path":["../minitree/ver0820/22preEE_GluGluToHH.root", "../minitree/ver0820/22postEE_GluGluToHH.root"],
    "selection": "(Higgs_toGG_genMatched == 1) & (lead_bjet_genMatched == 1) & (sublead_bjet_genMatched == 1)",
    "input_weight": 1,#("weight"),
    },
    {
    "Class":"VBF",
    "path":["../minitree/ver0820/22preEE_VBFToHH.root", "../minitree/ver0820/22postEE_VBFToHH.root"],
    "selection":"(Higgs_toGG_genMatched == 1) & (lead_bjet_genMatched == 1) & (sublead_bjet_genMatched == 1)", 
    "input_weight": 1,#("weight"), 
    },
    {
    "Class":"bkg", #"../minitree/ver0820/22postEE_GGJets.root",
    "path":["../minitree/ver0820/22postEE_GGJets.root", "../minitree/ver0820/22preEE_GluGluToHH.root", "../minitree/ver0820/22postEE_GluGluToHH.root", "../minitree/ver0820/22preEE_VBFToHH.root", "../minitree/ver0814_test/22postEE_VBFToHH.root"],
    # "path":["../minitree/ver0820/22postEE_GGJets.root", "../minitree/ver0820/22preEE_GluGluToHH.root", "../minitree/ver0820/22postEE_GluGluToHH.root", "../minitree/ver0820/22preEE_VBFToHH.root", "../minitree/ver0814_test/22postEE_VBFToHH.root"],
    "selection":"(lead_bjet_genMatched == 0) & (sublead_bjet_genMatched == 0)", 
    "input_weight": 1.#("weight"), 
    }
]
    # "features_unit":["$cos(\\theta)$", "$cos(\\Theta)$", "$\\Phi$", "$p^{T}_{H}/m_{H}$", "photon resol.", "min $\\Delta R(l,\\gamma)$", "max $\\Delta R(l,\\gamma)$",
                    # "phoID MVA", "$\eta^{l1}$", "$\eta^{l2}$", "$\eta^{\gamma}$"],

MVAs = [
    {
    "MVAtype":"test", 
    "Label":"no-weight",
    "addwei":"nowei",
    "features":["pt", "eta", "phi", "n_jets", "lead_bjet_pt", "sublead_bjet_pt", "lead_bjet_eta", "sublead_bjet_eta", "lead_bjet_phi", "sublead_bjet_phi", "lead_bjet_btagPNetB", "sublead_bjet_btagPNetB"], # "FirstJet_PtOverM", "SecondJet_PtOverM", "absCosThetaStar_CS", "absCosThetaStar_gg", "absCosThetaStar_jj"
    "features_unit":["$p_{T}^{\gamma\gamma}$", "$\eta^{\gamma\gamma}$", "$\\Phi^{\gamma\gamma}$", "$N_{jets}$", "leading $p_{T}^{b}$", "subleading $p_{T}^{b}$", "leading $\eta_{b}$", "subleading $\eta_{b}$", "leading $\Phi_{b}$", "subleading $\Phi_{b}$", "leading b btagPNetB", "subleading b btagPNetB"],
    "feature_bins":[np.linspace(0, 500, 31), np.linspace(-5, 5, 31), 15, np.linspace(2, 10, 9), np.linspace(20, 350, 31), np.linspace(20, 120, 31), 15, 15, 15, 15, 15, 15, 15, 15, 31, 31],
    "hyperopt":True,
    "ModelParams":{"tree_method": "hist", "device": "cuda", "objective": "multi:softprob", "num_class": 3,  "eval_metric": "merror", "random_state": RandomState},
    "HyperParams":{"max_depth": 1, "learning_rate": 0.1, "min_child_weight": 29.0, "min_split_loss": 1.6, "subsample": 0.65},
    }
]
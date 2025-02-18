#=============================================#
#           training configurations           #
#              ggF pairing study              #
#=============================================# 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Masking, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

OutputDirName = "results/EventCate" # All plots, models, config file will be stored here
Debug, MVAlogplot = False, False

# ROOT files
Tree = "DiphotonTree/data_125_13TeV_NOTAG"
Branches = [
    "event", "lumi_xs", "weight", "n_jets",
    # photon info
    "diphoton_pt", "diphoton_eta", "diphoton_phi", "diphoton_ptOverM",
    "pho_lead_pt", "pho_sublead_pt", "pho_lead_ptOverM", "pho_sublead_ptOverM", "pho_lead_eta", "pho_sublead_eta", "pho_lead_phi", "pho_sublead_phi", 
    # b-jet info
    "bjet_lead_pt", "bjet_sublead_pt", "bjet_lead_ptOverM", "bjet_sublead_ptOverM", "bjet_lead_eta", "bjet_sublead_eta", "bjet_lead_phi", "bjet_sublead_phi", 
    "bjet_lead_btagPNetB", "bjet_sublead_btagPNetB", "bjet_lead_btagPNetQvG", "bjet_sublead_btagPNetQvG",
    "bjet_pair_pt", "bjet_pair_ptOverM", "bjet_pair_eta", "bjet_pair_phi", "bjet_pair_mass", "bjet_pair_DeltaR", "bjet_pair_DeltaPhi", 
    "bjet_pair_eta_prod", "bjet_pair_eta_diff", "bjet_pair_Cgg", 
    "bjet_true_bjet_pair", "bjet_true_vbfjet_pair",
    # vbf-jet info
    "vbfjet_lead_pt", "vbfjet_sublead_pt", "vbfjet_lead_ptOverM", "vbfjet_sublead_ptOverM", "vbfjet_lead_eta", "vbfjet_sublead_eta", "vbfjet_lead_phi", "vbfjet_sublead_phi", 
    "vbfjet_lead_btagPNetB", "vbfjet_sublead_btagPNetB", "vbfjet_lead_btagPNetQvG", "vbfjet_sublead_btagPNetQvG",
    "vbfjet_pair_pt", "vbfjet_pair_ptOverM", "vbfjet_pair_eta", "vbfjet_pair_phi", "vbfjet_pair_mass", "vbfjet_pair_DeltaR", "vbfjet_pair_DeltaPhi", 
    "vbfjet_pair_eta_prod", "vbfjet_pair_eta_diff", "vbfjet_pair_Cgg", "vbfjet_pair_Cbb", 
    "vbfjet_true_bjet_pair", "vbfjet_true_vbfjet_pair",
]

Classes, ClassColors = ["ggHH-0/1jet", "ggHH-2jet", "VBFHH-0/1jet", "VBFHH-2jet", "background"], ["#FF745E", "#EFA443", "#009FB7", "#35CE8D", "#82645E"]
TestSize, RandomState = 0.5, 123

Processes = [
    {
    "Class": "ggHH-0/1jet",
    "path": ["../study_JetPairing/update_minitree/ver0121_/22*_GluGluToHH.root"],
    "selection": "(bjet_true_bjet_pair == 1) & (vbfjet_lead_pt < 0)",
    "input_weight": ("weight","lumi_xs"),
    },
    {
    "Class": "ggHH-2jet",
    "path": ["../study_JetPairing/update_minitree/ver0121_/22*_GluGluToHH.root"],
    "selection": "(bjet_true_bjet_pair == 1) & (vbfjet_lead_pt > 0)",
    "input_weight": ("weight","lumi_xs"),
    },
    {
    "Class": "VBFHH-0/1jet",
    "path": ["../study_JetPairing/update_minitree/ver0121_/22*_VBFToHH.root"],
    "selection": "(bjet_true_bjet_pair == 1) & (vbfjet_lead_pt < 0)",
    "input_weight": ("weight","lumi_xs"),
    },
    {
    "Class": "VBFHH-2jet",
    "path": ["../study_JetPairing/update_minitree/ver0121_/22*_VBFToHH.root"],
    "selection": "(bjet_true_bjet_pair == 1) & (vbfjet_true_vbfjet_pair == 1)",
    "input_weight": ("weight","lumi_xs"),
    },
    {
    "Class": "background",
    "path": ["../study_JetPairing/update_minitree/ver0121_/22*_GGJets.root", "../study_JetPairing/update_minitree/ver0121_/22*_GJet*.root"],
    "selection": "event > 0",
    "input_weight": ("weight","lumi_xs"),
    },
]

MVAs = [
    {
        "MVAtype":"DNN", 
        "Label":"DNN",
        "addwei":"nowei",
        "features":[
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
        "features_unit":[
            "lead $p_{T}^{\gamma}/M_{\gamma\gamma}$",  "sublead $p_{T}^{\gamma}/M_{\gamma\gamma}$",         "lead $\eta_{\gamma}$",             "sublead $\eta_{\gamma}$",
            "$p_{T}^{\gamma\gamma}/M_{\gamma\gamma}$", "$\eta_{\gamma\gamma}$",                             "$\Phi_{\gamma\gamma}$",           "$N_{jets}$",
            # b-jet info
            "lead $p_{T}^{b}/M_{bb}$",      "sublead $p_{T}^{b}/M_{bb}$",       "lead $\eta_{b}$",          "sublead $\eta_{b}$",
            "lead b-jet PNetB",             "sublead b-jet PNetB",              "lead b-jet PNetQvG",       "sublead b-jet PNetQvG",
            "$p_{T}^{bb}/M_{bb}$",          "$\eta_{bb}$",                      "$\Phi_{jj}$",              "$C_{\gamma\gamma, b-jets}$",
            "$\eta_{b_1} X \eta_{b_2}$",    "$\Delta\eta(b_1, b_2)$",           "$\Delta R(b_1, b_2)$",     "$\Delta \Phi(b_1, b_2)$", 
            # vbf-jet info
            "lead $p_{T}^{j}/M_{jj}$",      "sublead $p_{T}^{j}/M_{jj}$",       "lead $\eta_{j}$",          "sublead $\eta_{j}$",
            "lead VBF-jet PNetB",           "sublead VBF-jet PNetB",            "lead VBF-jet PNetQvG",     "sublead VBF-jet PNetQvG",
            "$p_{T}^{jj}/M_{jj}$",          "$\eta_{jj}$",                      "$\Phi_{jj}$",              "$M_{jj} (GeV)$",
            "$\eta_{j_1} X \eta_{j_2}$",    "$\Delta\eta(j_1, j_2)$",           "$\Delta R(j_1, j_2)$",     "$\Delta \Phi(j_1, j_2)$", 
            "$C_{\gamma\gamma, VBF-jets}$", "$C_{bb, VBF-jets}$", 

            ],
        "feature_bins":[
            np.linspace(0, 3.5, 31),    np.linspace(0, 2, 31),      31,                         31,
            np.linspace(0, 5, 31),      np.linspace(-5, 5, 31),     15,                         np.linspace(2, 10, 9), 
            # b-jet info 
            np.linspace(0, 4, 31),      np.linspace(0, 1.5, 31),    31,                         31, 
            np.linspace(0, 1, 31),      np.linspace(0, 1, 31),      np.linspace(0, 1, 31),      np.linspace(0, 1, 31),
            np.linspace(0, 5, 31),      np.linspace(-6, 6, 31),     15,                         np.linspace(0, 1, 31),
            np.linspace(-3, 5, 31),     np.linspace(-3.5, 3.5, 31), np.linspace(0, 4, 31),      31, 
            # vbf-jet info
            np.linspace(0, 1.5, 31),    np.linspace(0, 0.7, 31),    np.linspace(-5, 5, 31),     np.linspace(-5, 5, 31), 
            np.linspace(0, 1, 31),      np.linspace(0, 1, 31),      np.linspace(0, 1, 31),      np.linspace(0, 1, 31),
            np.linspace(0, 2, 31),      np.linspace(-7, 7, 31),     np.linspace(-3.2, 3.2, 31), np.linspace(0, 5000, 31),
            np.linspace(-18, 6, 31),    np.linspace(-10, 10, 31),   np.linspace(0, 10, 31),     np.linspace(-3.2, 3.2, 31), 
            np.linspace(0, 1, 31),      np.linspace(0, 1, 31),            
        ],
        "hyperopt": False,
        "Algorithm": "DNN",
        "Scaler":"MinMaxScaler", #Scaling for features before passing to the model training
        "DNNDict":{ "ModelParams":{"epochs": 10, "batchsize": 1000, "input_dim": (42,), "output_dim": len(Classes)},
                    'model': Sequential([#Masking(mask_value=-999),
                                Input(shape=(42,)),
                                Masking(mask_value=-1),
                                Dense(128, kernel_initializer='glorot_normal', activation='relu'),
                                Dropout(0.1),
                                Dense(64, activation="relu"),
                                # Dense(32, activation="relu"),
                                Dropout(0.1),
                                Dense(len(Classes), activation="softmax")
                            ]),
                    'compile': {'loss':'categorical_crossentropy','optimizer':Adam(learning_rate=0.001), 'metrics':['accuracy']},
                    'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
                }
    },
    # {
    #     "MVAtype":"XGB", 
    #     "Label":"XGB",
    #     "addwei":"nowei",
    #     "features":[
    #         "pt", "eta", "phi", "n_jets", 
    #         "pair1_ptOverM", "pair2_ptOverM", "pair1_eta", "pair2_eta", 
    #         "pair1_btagPNetB", "pair2_btagPNetB", "pair1_btagPNetQvG", "pair2_btagPNetQvG",
    #         "pair_pt","pair_eta", "pair_phi", "pair_mass",
    #         "pair_DeltaR", "pair_DeltaPhi", "pair_eta_prod", "pair_eta_diff", 
    #         "pair_Cgg", "pair1_phi", "pair2_phi"
    #     ],
    #     "features_unit":[
    #         "$p_{T}^{\gamma\gamma}$", "$\eta_{\gamma\gamma}$", "$\\Phi_{\gamma\gamma}$", "$N_{jets}$", 
    #         "lead $p_{T}^{j}/M_{jj}$", "sublead $p_{T}^{j}/M_{jj}$", "lead $\eta_{j}$", "sublead $\eta_{j}$", 
    #         "lead jet PNetB", "sublead jet PNetB", "lead jet PNetQvG", "sublead jet PNetQvG", 
    #         "$p_{T}^{jj}$", "$\eta_{jj}$", "$\Phi_{jj}$", "$M_{jj} (GeV)$",
    #         "$\Delta R(j_1, j_2)$", "$\Delta \Phi(j_1, j_2)$", 
    #         "$\eta_{j_1} X \eta_{j_2}$", "$\Delta\eta(j_1, j_2)$", 
    #         "$C_{\gamma\gamma}$", "lead $\Phi_{j}$", "sublead $\Phi_{j}$"
    #     ],
    #     "feature_bins":[
    #         np.linspace(0, 500, 31), np.linspace(-5, 5, 31), 15, np.linspace(2, 10, 9), 
    #         np.linspace(0, 2.5, 31), np.linspace(0, 1.5, 31), 15, 15, 
    #         np.linspace(0, 1, 21), np.linspace(0, 1, 21), np.linspace(0, 1, 21), np.linspace(0, 1, 21),
    #         np.linspace(0, 500, 31), np.linspace(-5, 5, 31), 15, np.linspace(0, 3000, 31),
    #         31, 31, 31, 31,
    #         31, 15, 15      
    #     ],
    #     "Algorithm": "XGB",
    #     "hyperopt":True,
    #     "Scaler":"MinMaxScaler", #Scaling for features before passing to the model training
    #     "XGBDict":{
    #         "ModelParams":{"tree_method": "hist", "device": "cuda:0", "objective": "multi:softprob", "num_class": len(Classes),  "eval_metric": "merror", "random_state": RandomState},
    #         "HyperParams":{"max_depth": 2, "learning_rate": 0.1, "min_child_weight": 12.0, "min_split_loss": 0.0, "subsample": 0.7000000000000001}
    #     }
    #     },

]

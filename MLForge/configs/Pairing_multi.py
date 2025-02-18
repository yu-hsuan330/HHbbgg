#=============================================#
#           training configurations           #
#              ggF pairing study              #
#=============================================# 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Masking, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

OutputDirName = "MultiClass_DNN_Pairing_v4" # All plots, models, config file will be stored here
Debug, MVAlogplot = False, False

# ROOT files
Tree = "DiphotonTree/data_125_13TeV_NOTAG"
Branches = [
    "pair1_pt", "pair2_pt", "pair1_ptOverM", "pair2_ptOverM", "pair1_eta", "pair2_eta", "pair1_phi", "pair2_phi", 
    "pair1_btagPNetB", "pair2_btagPNetB", "pair1_btagPNetQvG", "pair2_btagPNetQvG",
    "pair_pt", "pair_eta", "pair_phi", "pair_mass", "pair_DeltaR", "pair_DeltaPhi", "pair_eta_prod", "pair_eta_diff", "pair_Cgg",
    "weight", "pt", "eta", "phi", "n_jets", "lumi_xs", "b_jetPair", "vbf_jetPair", "WrongPair",
    "lead_pt", "lead_phi", "lead_eta", "sublead_eta", "lead_mvaID", "sublead_mvaID",
    "nonRes_pholead_PtOverM", "nonRes_phosublead_PtOverM", "nonRes_FirstJet_PtOverM", "nonRes_SecondJet_PtOverM",
    "nonRes_lead_bjet_pt", "nonRes_lead_bjet_eta", "nonRes_lead_bjet_phi", "nonRes_lead_bjet_btagPNetB", 
    "nonRes_sublead_bjet_pt", "nonRes_sublead_bjet_eta", "nonRes_sublead_bjet_phi", "nonRes_sublead_bjet_btagPNetB",
    "nonRes_dijet_mass", "nonRes_dijet_eta", "nonRes_DeltaR_jg_min",
    # "absCosThetaStar_CS", "absCosThetaStar_gg", "absCosThetaStar_jj",
]

Classes, ClassColors = ["b-jet pair", "VBF-jet pair", "wrong pair"], ["#22577E", "#EF5B5B", "#F97D10"]
TestSize, RandomState = 0.5, 123

Processes = [
    {
    "Class": "b-jet pair",
    "path": ["../minitree/ver0121_pair/22*_VBFToHH_jet*.root", "../minitree/ver0121_pair/22*_GluGluToHH_jet*.root"],
    "selection": "(b_jetPair == 1)",
    "input_weight": ("weight","lumi_xs"),
    },
    {
    "Class": "VBF-jet pair",
    "path": ["../minitree/ver0121_pair/22*_VBFToHH_jet*.root"],
    "selection": "(vbf_jetPair == 1)",
    "input_weight": ("weight","lumi_xs"),
    },
    {
    "Class": "wrong pair",
    "path": ["../minitree/ver0121_pair/22*_VBFToHH_jet*.root", "../minitree/ver0121_pair/22*_GluGluToHH_jet*.root"],#, "../minitree/ver0121_pair/22*GJet*.root"],
    "selection": "(WrongPair == 1)",
    "input_weight": ("weight","lumi_xs"),
    },
]

MVAs = [
    {
        "MVAtype":"DNN", 
        "Label":"DNN-VBF",
        "addwei":"nowei",
        "features":[
             
            "pair1_ptOverM", "pair2_ptOverM", "pair1_eta", "pair2_eta", 
            "pair1_btagPNetB", "pair2_btagPNetB", "pair1_btagPNetQvG", "pair2_btagPNetQvG",
            "n_jets","pair_pt","pair_eta", "pair_mass",
            "pair_DeltaR", "pair_DeltaPhi", "pair_eta_prod", "pair_eta_diff", 
            "pair_Cgg", "pair1_phi", "pair2_phi"
        ],
        "features_unit":[
            
            "lead $p_{T}^{j}/M_{jj}$", "sublead $p_{T}^{j}/M_{jj}$", "lead $\eta_{j}$", "sublead $\eta_{j}$", 
            "lead jet PNetB", "sublead jet PNetB", "lead jet PNetQvG", "sublead jet PNetQvG", 
            "$N_{jets}$", "$p_{T}^{jj}$", "$\eta_{jj}$", "$M_{jj} (GeV)$",
            "$\Delta R(j_1, j_2)$", "$\Delta \Phi(j_1, j_2)$", 
            "$\eta_{j_1} X \eta_{j_2}$", "$\Delta\eta(j_1, j_2)$", 
            "$C_{\gamma\gamma}$", "lead $\Phi_{j}$", "sublead $\Phi_{j}$"
        ],
        "feature_bins":[ 
            np.linspace(0, 2.5, 31), np.linspace(0, 1.5, 31), 15, 15, 
            np.linspace(0, 1, 21), np.linspace(0, 1, 21), np.linspace(0, 1, 21), np.linspace(0, 1, 21),
            np.linspace(2, 10, 9), np.linspace(0, 500, 31), np.linspace(-5, 5, 31), np.linspace(0, 3000, 31),
            31, 31, 31, 31,
            31, 15, 15      
        ],
        "hyperopt": False,
        "Algorithm": "DNN",
        "Scaler":"MinMaxScaler", #Scaling for features before passing to the model training
        "DNNDict":{ "ModelParams":{"epochs": 200, "batchsize": 2500, "input_dim": (19,), "output_dim": len(Classes)},
                    'model': Sequential([
                                Input(shape=(19,)),
                                Masking(mask_value=-1),
                                Dense(64, kernel_initializer='glorot_normal', activation='relu'),
                                Dense(32, activation="relu"),
                                Dense(24, activation="relu"),
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

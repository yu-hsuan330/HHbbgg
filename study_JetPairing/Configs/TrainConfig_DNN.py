#=============================================#
#           training configurations           #
#           VBF pairing study                 #
#=============================================# 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

OutputDirName = "MultiClass_DNN_test" # All plots, models, config file will be stored here
Debug, MVAlogplot = False, False

# ROOT files
Tree = "DiphotonTree/data_125_13TeV_NOTAG"
Branches = [
    "weight", "pt", "eta", "phi", "n_jets",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_mass", "jet2_pt", "jet2_eta", "jet2_phi", "jet2_mass",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_mass", "jet4_pt", "jet4_eta", "jet4_phi", "jet4_mass",
    "jet5_pt", "jet5_eta", "jet5_phi", "jet5_mass", "jet6_pt", "jet6_eta", "jet6_phi", "jet6_mass",

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
    "selection": "lead_bjet_genMatched == lead_bjet_genMatched",
    "new_branches":[],
    "input_weight": 1,#("weight"),
    },
    {
    "Class":"VBF",
    "path":["../minitree/ver0820/22preEE_VBFToHH.root", "../minitree/ver0820/22postEE_VBFToHH.root"],
    "selection":"lead_bjet_genMatched == lead_bjet_genMatched", 
    "new_branches":[],
    "input_weight": 1,#("weight"), 
    },
    {
    "Class":"bkg", #"../minitree/ver0820/22postEE_GGJets.root",
    "path":["../minitree/ver0820/22postEE_GGJets.root"],
    # "path":["../minitree/ver0820/22postEE_GGJets.root", "../minitree/ver0820/22preEE_GluGluToHH.root", "../minitree/ver0820/22postEE_GluGluToHH.root", "../minitree/ver0820/22preEE_VBFToHH.root", "../minitree/ver0814_test/22postEE_VBFToHH.root"],
    "selection":"(lead_bjet_genMatched == 0) & (sublead_bjet_genMatched == 0)", 
    "new_branches":[],
    "input_weight": 1.#("weight"), 
    }
]
    # "features_unit":["$cos(\\theta)$", "$cos(\\Theta)$", "$\\Phi$", "$p^{T}_{H}/m_{H}$", "photon resol.", "min $\\Delta R(l,\\gamma)$", "max $\\Delta R(l,\\gamma)$",
                    # "phoID MVA", "$\eta^{l1}$", "$\eta^{l2}$", "$\eta^{\gamma}$"],

MVAs = [
    {
    "MVAtype":"test", 
    "Algorithm": "DNN",
    "Label":"no-weight",
    "addwei":"nowei",
    "features":[
        "jet1_pt", "jet1_eta", "jet1_phi", "jet1_mass", 
        "jet2_pt", "jet2_eta", "jet2_phi", "jet2_mass",
        "jet3_pt", "jet3_eta", "jet3_phi", "jet3_mass", 
        "jet4_pt", "jet4_eta", "jet4_phi", "jet4_mass",
        "jet5_pt", "jet5_eta", "jet5_phi", "jet5_mass", 
        "jet6_pt", "jet6_eta", "jet6_phi", "jet6_mass",
        "n_jets", 
    ], # "FirstJet_PtOverM", "SecondJet_PtOverM", "absCosThetaStar_CS", "absCosThetaStar_gg", "absCosThetaStar_jj"
    # "features_unit":["$p_{T}^{\gamma\gamma}$", "$\eta^{\gamma\gamma}$", "$\\Phi^{\gamma\gamma}$", "$N_{jets}$", "leading $p_{T}^{b}$", "subleading $p_{T}^{b}$", "leading $\eta_{b}$", "subleading $\eta_{b}$", "leading $\Phi_{b}$", "subleading $\Phi_{b}$", "leading b btagPNetB", "subleading b btagPNetB"],
    "features_unit":[
        "jet1_pt", "jet1_eta", "jet1_phi", "jet1_mass", 
        "jet2_pt", "jet2_eta", "jet2_phi", "jet2_mass",
        "jet3_pt", "jet3_eta", "jet3_phi", "jet3_mass", 
        "jet4_pt", "jet4_eta", "jet4_phi", "jet4_mass",
        "jet5_pt", "jet5_eta", "jet5_phi", "jet5_mass", 
        "jet6_pt", "jet6_eta", "jet6_phi", "jet6_mass",
        "n_jets", 
    ],
    "feature_bins":[
        np.linspace(20, 400, 41), np.linspace(-5, 5, 31), np.linspace(-3.3, 3.3, 21), np.linspace(0, 50, 21), 
        np.linspace(20, 200, 31), np.linspace(-5, 5, 31), np.linspace(-3.3, 3.3, 21), np.linspace(0, 30, 21),
        np.linspace(20, 130, 31), np.linspace(-5, 5, 31), np.linspace(-3.3, 3.3, 21), np.linspace(0, 20, 21),
        np.linspace(20, 100, 31), np.linspace(-5, 5, 31), np.linspace(-3.3, 3.3, 21), np.linspace(0, 20, 21),
        np.linspace(20, 70, 31), np.linspace(-5, 5, 31), np.linspace(-3.3, 3.3, 21), np.linspace(0, 20, 21),
        np.linspace(20, 70, 31), np.linspace(-5, 5, 31), np.linspace(-3.3, 3.3, 21), np.linspace(0, 20, 21),
        np.linspace(2, 10, 9), 
    ],
    "hyperopt":True,
    "ModelParams":{"tree_method": "hist", "device": "cuda", "objective": "multi:softprob", "num_class": 3,  "eval_metric": "merror", "random_state": RandomState},
    "HyperParams":{"max_depth": 1, "learning_rate": 0.1, "min_child_weight": 29.0, "min_split_loss": 1.6, "subsample": 0.65},
    "Scaler":"MinMaxScaler", #Scaling for features before passing to the model training
    "DNNDict":{'epochs':5, 'batchsize':500,
                'model': Sequential([Dense(64, kernel_initializer='glorot_normal', activation='relu'),
                                     Dense(32, activation="relu"),
                                     Dense(24, activation="relu"),
                                     Dropout(0.1),
                                     Dense(len(Classes),activation="softmax")]),
                'compile':{'loss':'categorical_crossentropy','optimizer':Adam(learning_rate=0.001), 'metrics':['accuracy']},
                'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
               }
    }
]
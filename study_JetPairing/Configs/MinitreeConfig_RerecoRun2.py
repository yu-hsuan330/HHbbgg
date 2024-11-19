#==========================================================#
# All sample information and configuration are listed here #
#==========================================================#
branches = ["costheta", "cosTheta", "Phi", "costheta_", "cosTheta_", "Phi_", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1","lepEta1", "lepEta2", "phoEta1", "totwei", "totSF", "refit_mllg","phoSCEta1"]
features = ["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1","lepEta1", "lepEta2", "phoEta1"]
features_data = ["costheta_", "cosTheta_", "Phi_", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1","lepEta1", "lepEta2", "phoEta1"]
Tree = "TH"

samples = [
    {"Production": "Signal",
     "Weight": ("totwei","totSF"),
     "Color": "#EF5B5B", 
     "Label": "signal",
     "epath": ["../minitrees/Legacy16_ele_*_125GeV.root", "../minitrees/Rereco17_ele_*_125GeV.root", "../minitrees/Rereco18_ele_*_125GeV.root"],
     "upath": ["../minitrees/Legacy16_mu_*_125GeV.root", "../minitrees/Rereco17_mu_*_125GeV.root", "../minitrees/Rereco18_mu_*_125GeV.root"],
    #  "path": ["../minitrees/Legacy16_ele_*_125GeV.root", "../minitrees/Rereco17_ele_*_125GeV.root", "../minitrees/Rereco18_ele_*_125GeV.root","../minitrees/Legacy16_mu_*_125GeV.root", "../minitrees/Rereco17_mu_*_125GeV.root", "../minitrees/Rereco18_mu_*_125GeV.root"]
    },
    {"Production": "SMZg",
     "Weight": ("totwei","totSF"),
     "Color": "#37A3D2",
     "Label": "SM Zg", 
     "epath": ["../minitrees/Legacy16_ele_SMZg.root", "../minitrees/Rereco17_ele_SMZg.root", "../minitrees/Rereco18_ele_SMZg.root"],
     "upath": ["../minitrees/Legacy16_mu_SMZg.root",  "../minitrees/Rereco17_mu_SMZg.root",  "../minitrees/Rereco18_mu_SMZg.root"],
    #  "path": ["../minitrees/Legacy16_ele_SMZg.root", "../minitrees/Rereco17_ele_SMZg.root", "../minitrees/Rereco18_ele_SMZg.root","../minitrees/Legacy16_mu_SMZg.root",  "../minitrees/Rereco17_mu_SMZg.root",  "../minitrees/Rereco18_mu_SMZg.root"]
    }, 
    {"Production": "TTbar",
     "Weight": ("totwei","totSF"),
     "Color": "#EC8C6F",
     "Label": "TTbar", 
     "epath": ["../minitrees/Legacy16_ele_TTbar.root", "../minitrees/Rereco17_ele_TTbar.root", "../minitrees/Rereco18_ele_TTbar.root"],
     "upath": ["../minitrees/Legacy16_mu_TTbar.root",  "../minitrees/Rereco17_mu_TTbar.root", "../minitrees/Rereco18_mu_TTbar.root"],
    #  "path": ["../minitrees/Legacy16_ele_TTbar.root", "../minitrees/Rereco17_ele_TTbar.root", "../minitrees/Rereco18_ele_TTbar.root","../minitrees/Legacy16_mu_TTbar.root",  "../minitrees/Rereco17_mu_TTbar.root", "../minitrees/Rereco18_mu_TTbar.root"],

    }, 
    {"Production": "DYJets",
     "Weight": ("totwei","totSF"),
     "Color": "#F3C568",
     "Label": "DY+Jets", 
     "epath": ["../minitrees/Legacy16_ele_DYJetsToLL.root", "../minitrees/Rereco17_ele_DYJetsToLL.root", "../minitrees/Rereco18_ele_DYJetsToLL.root"],
     "upath": ["../minitrees/Legacy16_mu_DYJetsToLL.root",  "../minitrees/Rereco17_mu_DYJetsToLL.root",  "../minitrees/Rereco18_mu_DYJetsToLL.root"],
    #  "path": ["../minitrees/Legacy16_ele_DYJetsToLL.root", "../minitrees/Rereco17_ele_DYJetsToLL.root", "../minitrees/Rereco18_ele_DYJetsToLL.root","../minitrees/Legacy16_mu_DYJetsToLL.root",  "../minitrees/Rereco17_mu_DYJetsToLL.root",  "../minitrees/Rereco18_mu_DYJetsToLL.root"],

    },
    {"Production": "Data",
     "Weight": ("totwei","totSF"),
     "Color": "#000000",
     "Label": "Data", 
     "epath": ["../minitrees/data/Legacy16_ele_data*.root", "../minitrees/data/Rereco17_ele_data*.root", "../minitrees/data/Rereco18_ele_data*.root"],
     "upath": ["../minitrees/data/Legacy16_mu_data*.root",  "../minitrees/data/Rereco17_mu_data*.root",  "../minitrees/data/Rereco18_mu_data*.root"],
     "path": ["../minitrees/data/Legacy16_ele_data*.root", "../minitrees/data/Rereco17_ele_data*.root", "../minitrees/data/Rereco18_ele_data*.root", "../minitrees/data/Legacy16_mu_data*.root",  "../minitrees/data/Rereco17_mu_data*.root",  "../minitrees/data/Rereco18_mu_data*.root"],

    },
]

MVAlogplot=False
import tensorflow as tf
import pandas as pd
import numpy as np
import awkward as ak
import pickle
import uproot
import argparse
from coffea.nanoevents.methods import candidate
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator

import warnings
warnings.filterwarnings("error", module="coffea.*")
    
def pltSty(ax, xName = "x-axis", yName = "y-axis", TitleSize = 17, LabelSize = 16, TickSize = 13, MajTickLength = 7, MinTickLength = 4, yAuto = True):
    ax.set_xlabel(xName, fontsize = LabelSize, loc = "right")
    ax.set_ylabel(yName, fontsize = LabelSize, loc = "top")
    # ax.text(1, 1, "(13 TeV)", horizontalalignment = "right", verticalalignment = "bottom", transform = ax.transAxes, fontsize = TitleSize)
    ax.text(0, 1.01, "CMS", horizontalalignment = "left", verticalalignment = "bottom", transform = ax.transAxes, fontsize = TitleSize * 1.3, fontweight = "bold")
    ax.text(TitleSize * 0.01, 1.015, "work-in-progress", horizontalalignment = "left", verticalalignment = "bottom", transform = ax.transAxes, fontsize = TitleSize)

    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if yAuto :
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction = "in", length = MajTickLength, labelsize = TickSize, top = True, right = True)
    ax.tick_params(direction = "in", length = MinTickLength, which = "minor", labelsize = TickSize, top = True, right = True)

if __name__ == "__main__":
    
    events_new = uproot.concatenate(["/home/cosine/HHbbgg/MLForge_FollowUp/JetPairing/model_prediction/samples/Pairing_vbf_wRatio/22preEE_GGJets.root:DiphotonTree/data_125_13TeV_NOTAG"], library="ak")
    events_old = uproot.concatenate(["/home/cosine/HHbbgg/MLForge_FollowUp/JetPairing/model_prediction/samples/Pairing_vbf_woRatio/22preEE_GGJets.root:DiphotonTree/data_125_13TeV_NOTAG"], library="ak")
    events = pd.read_parquet(
        "/home/cosine/HHbbgg/minitree_parquet/ver0518/mc/merged/22preEE_GGJets/NOTAG_merged.parquet", # columns=new_columns, filters=filters
    )
    events_new = events_new[(events_new["diphoton_mass"] > 100) & (events_new["diphoton_mass"] < 180) & (events_new["vbfjet_pair_mass"] > 0)] # & (events_new["vbfjet_pair_mass"] > 0)
    events_old = events_old[(events_old["diphoton_mass"] > 100) & (events_old["diphoton_mass"] < 180) & (events_old["vbfjet_pair_mass"] > 0)]
    events = events[(events["nonRes_dijet_mass"] > 70) & (events["nonRes_dijet_mass"] < 190) & (events["mass"] > 100) & (events["mass"] < 180) & (events["VBF_dijet_mass"] > 0)] #  & (events["VBF_dijet_mass"] > 0)
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.hist(events_old["bjet_pair_mass"], label="DNN(orginal)", weights=events_old["weight"], bins=np.linspace(70, 190, 31), density=False, histtype='step', alpha=0.7, linewidth=4)
    ax.hist(events_new["bjet_pair_mass"], label="DNN(new)", weights=events_new["weight"], bins=np.linspace(70, 190, 31), density=False, histtype='step', alpha=0.7, linewidth=4)
    ax.hist(events["nonRes_dijet_mass"], label="cut-based", weights=events["weight"], bins=np.linspace(70, 190, 31), density=False, histtype='step', alpha=0.7, linewidth=4)

    pltSty(ax, xName="H(bb) mass [GeV]", yName="Events", yAuto=False)
    plt.yscale("log")
    ax.legend(title="", loc="best", title_fontsize=12, fontsize=12, frameon=False)

    
    fig.savefig("./GGJets_bbjj.pdf", bbox_inches='tight')
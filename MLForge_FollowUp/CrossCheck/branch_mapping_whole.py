import tensorflow as tf
import pandas as pd
import numpy as np
import awkward as ak
import time
import pickle
import uproot
import argparse
from coffea.nanoevents.methods import candidate
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator
from Config import *

import warnings
# warnings.filterwarnings("error", module="coffea.*")

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

def get_parser():
    
    parser = argparse.ArgumentParser(prog="ML-prediction", description="Jet pairing - prediction", epilog="Good luck!")
    parser.add_argument("-e", "--era", type=str, required=True, help="Era of the sample")
    return parser

if __name__ == "__main__":
    #* Parse the arguments
    parser = get_parser()
    args = parser.parse_args()
    
    for sample_1, sample_2 in zip(samples_1, samples_2):
        print("Start processing file: ", sample_1["name"], " and ", sample_2["name"])
        start_time = time.time()
        
        events_1 = uproot.concatenate(sample_1[args.era], library="ak")    
        # events_2 = uproot.concatenate(sample_2[args.era], library="ak")

        events_2 = uproot.concatenate(sample_2[args.era], ["event", "jet1_lheMatched", "jet2_lheMatched","jet3_lheMatched","jet4_lheMatched","jet5_lheMatched","jet6_lheMatched",
                    "jet1_selected_bjet", "jet2_selected_bjet","jet3_selected_bjet","jet4_selected_bjet","jet5_selected_bjet","jet6_selected_bjet",
                    "jet1_selected_vbfjet", "jet2_selected_vbfjet","jet3_selected_vbfjet","jet4_selected_vbfjet","jet5_selected_vbfjet","jet6_selected_vbfjet",
                    "jet1_btagPNetQvG", "jet2_btagPNetQvG","jet3_btagPNetQvG","jet4_btagPNetQvG","jet5_btagPNetQvG","jet6_btagPNetQvG"], library="ak")

        events_1 = ak.Array([events_1[:]])
        events_2 = ak.Array([events_2[:]])

        pair = ak.cartesian({"1": events_1[["event"]], "2": events_2}, axis=1, nested=True)
        pair = pair[(pair["1"]["event"] == pair["2"]["event"])]
        pair = ak.firsts(pair, axis=2)
    
        events_1 = ak.flatten(events_1)
        pair = ak.flatten(pair)

        events_1["matched"] = ~ak.is_none(pair["1"]["event"])
        
        EventCate = np.load(f"./predictions_for_samples/{args.era}/{sample_1['name']}/y_after_random_search_best1.npy")
        EventCate = EventCate[:]
        # print(EventCate[:])
        events_1["DNN_class"] = ak.argmax(EventCate[:], axis=1, mask_identity=False)
        events_1["DNN0"] = EventCate[:, 0]
        events_1["DNN1"] = EventCate[:, 1]
        events_1["DNN2"] = EventCate[:, 2] 
        events_1["DNN3"] = EventCate[:, 3]
        events_1["DNN4"] = EventCate[:, 4]
        
        for var in ["jet1_lheMatched", "jet2_lheMatched","jet3_lheMatched","jet4_lheMatched","jet5_lheMatched","jet6_lheMatched",
                    "jet1_selected_bjet", "jet2_selected_bjet","jet3_selected_bjet","jet4_selected_bjet","jet5_selected_bjet","jet6_selected_bjet",
                    "jet1_selected_vbfjet", "jet2_selected_vbfjet","jet3_selected_vbfjet","jet4_selected_vbfjet","jet5_selected_vbfjet","jet6_selected_vbfjet",
                    "jet1_btagPNetQvG", "jet2_btagPNetQvG","jet3_btagPNetQvG","jet4_btagPNetQvG","jet5_btagPNetQvG","jet6_btagPNetQvG"]:
            
            events_1[var] = ak.fill_none(pair["2"][var], -999)
        
        #* Store the selected pair to ROOT file
        with uproot.recreate(f"/home/cosine/HHbbgg/MLForge_FollowUp/samples/sample_from_Nitishi_add/{args.era}_{sample_1['name']}.root") as new_file:
            if "data" in sample_1["name"]:
                new_file[TreeName_data] = events_1
            else:
               new_file[TreeName_mc] = events_1
               
        print("Time elapsed: ", time.time() - start_time)
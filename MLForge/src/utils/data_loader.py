import uproot
import pandas as pd
import dask.dataframe as dd
from dask import delayed

# import ROOT
import numpy as np

def df_from_rootfiles_pairing(processes, treepath, branches, Branches_custom, debug=False):
      
    def get_df(Class, file, wt, selection, treepath=None, branches="", custom=[]):
        # Load root file 
        df = uproot.concatenate(file, branches, cut=selection, library="pd")
        if debug: df = df.head(1000)
        
        df["input_weight"] = 1.
        df["Class"] = Class
        
        # add global weights
        if debug: print(type(wt))
        if type(wt) == type(('wei1','wei2')):
            for i in wt:
                df["input_weight"] *= df[i]
        elif type(wt) == type("hello"):
            df["input_weight"] = df[wt]
        elif (type(wt) == type(0.1)) or (type(wt) == type(1)):
            df["input_weight"] = wt
        else:
            print("CAUTION: weight should be a branch name or a number or a tuple... Assigning the weight as 1")   

        if debug: print(file, df["input_weight"].size, df["input_weight"].abs().sum())     

        # add customized branches
        for pair, name in custom:
            for i in Branches_custom:
                df[f"{pair}_{i}"] = df[f"{name}_{i}"]

        return df

    dfs=[]
    for process in processes:
        if isinstance(process['path'], list):
            if isinstance(process['input_weight'], list):
                for onefile, onewt in zip(process['path'], process['input_weight']):
                    dfs.append(delayed(get_df)(process['Class'], onefile, process['input_weight'], process['selection'], treepath, branches, process["custom"]))
            else:            
                for onefile in process['path']:
                    dfs.append(delayed(get_df)(process['Class'], onefile, process['input_weight'], process['selection'], treepath, branches, process["custom"]))
        elif isinstance(process['path'], tuple) and len(process['path']) == 2:
            listoffiles=[process['path'][0]+'/'+f for f in os.listdir(process['path'][0]) if f.endswith(process['path'][1])]
            if debug: print(listoffiles)
            for onefile in listoffiles:
                dfs.append(delayed(get_df)(process['Class'], onefile, process['input_weight'], process['selection'], treepath, branches, process["custom"]))
        elif isinstance(process['path'], str):
            dfs.append(delayed(get_df)(process['Class'], process['path'], process['input_weight'], process['selection'], treepath, branches, process["custom"]))
        else:
            print("There is some problem with process path specification. Only string, list or tuple allowed")
    if debug: print("Creating dask graph!")
    if debug: print("Testing single file first")
    
    daskframe = dd.from_delayed(dfs)
    if debug: print("Finally, getting data from")
    output = daskframe.compute()
    output.reset_index(inplace = True, drop = True)
    return output
    
def df_from_rootfiles(processes, treepath, branches, debug=False):
      
    def get_df(Class, file, wt, selection, treepath=None, branches=""):
        # Load root file 
        df = uproot.concatenate(file, branches, cut=selection.strip(), library="pd")
        if debug: df = df.head(1000)
        # df["bjet_true_bjet_pair"] = df["bjet_true_bjet_pair"].astype(bool)
        # df["bjet_true_vbfjet_pair"] = df["bjet_true_vbfjet_pair"].astype(bool)

        df["input_weight"] = 1.
        df["Class"] = Class
        
        # add global weights
        if debug: print(type(wt))
        if type(wt) == type(('wei1','wei2')):
            for i in wt:
                df["input_weight"] *= df[i]
        elif type(wt) == type("hello"):
            df["input_weight"] = df[wt]
        elif (type(wt) == type(0.1)) or (type(wt) == type(1)):
            df["input_weight"] = wt
        else:
            print("CAUTION: weight should be a branch name or a number or a tuple... Assigning the weight as 1")   

        if debug: print(file, df["input_weight"].size, df["input_weight"].abs().sum())     


        return df

    dfs=[]
    for process in processes:
        if isinstance(process['path'], list):
            if isinstance(process['input_weight'], list):
                for onefile, onewt in zip(process['path'], process['input_weight']):
                    dfs.append(delayed(get_df)(process['Class'], onefile, process['input_weight'], process['selection'], treepath, branches))
            else:            
                for onefile in process['path']:
                    dfs.append(delayed(get_df)(process['Class'], onefile, process['input_weight'], process['selection'], treepath, branches))
        elif isinstance(process['path'], str):
            dfs.append(delayed(get_df)(process['Class'], process['path'], process['input_weight'], process['selection'], treepath, branches))
        else:
            print("There is some problem with process path specification. Only string, list or tuple allowed")
    if debug: print("Creating dask graph!")
    if debug: print("Testing single file first")
    
    daskframe = dd.from_delayed(dfs)
    if debug: print("Finally, getting data from")
    output = daskframe.compute()
    output.reset_index(inplace = True, drop = True)
    return output
    
def df_balance_rwt(Mdf, SumWeightCol="InstWt", NewWeightCol="NewWt", Classes=[""], debug=False) -> pd.Series:
    Mdf[NewWeightCol] = 1.
    sum_w, sum_w_, wei = [1.0]*len(Classes), [1.0]*len(Classes), [1.0]*len(Classes)
    
    # calculate the weighted sum of each class
    for i, k in enumerate(Classes):
        sum_w[i] = Mdf[SumWeightCol][Mdf.Class == k].sum()
   
    # reweight to the final category class[-1]
    for i, k in enumerate(Classes): 
        wei[i] = sum_w[-1] / sum_w[i]
        Mdf.loc[Mdf.Class == k, NewWeightCol] = wei[i] * Mdf[SumWeightCol][Mdf.Class == k]
        sum_w_[i] = Mdf[NewWeightCol][Mdf.Class == k].sum()
        if debug: print("Class = %s, n = %.2f, balanced n = %.2f" %(k, sum_w[i], sum_w_[i]), flush=True)
    
    return Mdf[NewWeightCol]

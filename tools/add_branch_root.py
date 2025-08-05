import os
import pandas as pd
import numpy as np
import uproot
def all_ttrees(directory, prefix=""):
    """
    Recursively yield (full_name, tree_object) for every TTree
    in the given uproot directory, including subdirectories.
    """
    for key, obj in directory.items():              # keys like b'Tree1;1'
        cname = key.decode().split(";")[0]          # strip ";cycle"
        full  = f"{prefix}{cname}"
        if isinstance(obj, uproot.behaviors.TTree.TTree):
            yield full, obj
        elif isinstance(obj, uproot.reading.ReadOnlyDirectory):
            yield from all_ttrees(obj, prefix=full + "/")
            
if __name__ == "__main__":
    
    path = "/home/cosine/HHbbgg/minitree/0704/22postEE"
    ext = "_fit"
    for sample in os.listdir(f"{path}/root"):
        for file in os.listdir(f"{path}/root/{sample}"):
            os.makedirs(f"{path}{ext}/root/{sample}", exist_ok=True)
            with uproot.open(f"{path}/root/{sample}/{file}") as fin, uproot.recreate(f"{path}{ext}/root/{sample}/{file}") as fout:
                fin_keys = fin.keys()
                
                lumi_xs = 1
                lumi = -999
                if "postEE" in path:
                    lumi = 26337
                elif "preEE" in path:
                    lumi = 8000
                
                bbgg = (2*0.5824*2.270E-03)
                if "GluGlutoHHto2B2G-kl-1p00-kt-1p00-c2-0p00" in file:
                    lumi_xs = lumi*bbgg*0.03413
                elif "GluGlutoHHto2B2G-kl-0p00-kt-1p00-c2-0p00" in file:
                    lumi_xs = lumi*bbgg*0.07575
                elif "GluGlutoHHto2B2G-kl-2p45-kt-1p00-c2-0p00" in file:
                    lumi_xs = lumi*bbgg*0.01477
                elif "GluGlutoHHto2B2G-kl-5p00-kt-1p00-c2-0p00" in file:
                    lumi_xs = lumi*bbgg*0.09965
                elif "VBFHHto2B2G-CV-1-C2V-1-C3-1" in file:
                    lumi_xs = lumi*bbgg*0.001886
                elif "VBFHHto2B2G-CV-1-C2V-0-C3-1" in file:
                    lumi_xs = lumi*bbgg*0.0270800
                elif "VBFHHto2B2G-CV-m1p83-C2V-3p57-C3-m3p39" in file:
                    lumi_xs = lumi*bbgg*0.0149850
                elif "VBFHHto2B2G-CV-1p74-C2V-1p37-C3-14p4" in file:
                    lumi_xs = lumi*bbgg*0.3777832
                elif "VBFHHto2B2G-CV-m0p758-C2V-1p44-C3-m19p3" in file:
                    lumi_xs = lumi*bbgg*0.3340766
                elif "VBFHHto2B2G-CV-m0p012-C2V-0p030-C3-10p2" in file:
                    lumi_xs = lumi*bbgg*0.0000120
                elif "GluGluHtoGG_M125" in file:
                    lumi_xs = lumi*51.96*2.270E-03
                elif "VBFHtoGG_M125" in file:
                    lumi_xs = lumi*4.067*2.270E-03
                for tree_name in fin_keys[1:]:
                    tree = fin[tree_name[:-2]]
                    arrays = tree.arrays(library="np")
                    arrays["totwei"] = arrays["weight"]*lumi_xs
                    arrays["lumi_xs"] = np.ones(len(arrays["weight"]))*lumi_xs
                    
                    fout[tree_name[:-2]] = arrays 

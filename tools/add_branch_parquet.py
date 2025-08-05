import os
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq

if __name__ == "__main__":
    
    path = "/home/cosine/HHbbgg/minitree/0702_merge_data"
    for sample in os.listdir(f"{path}/merged"):
        for file in os.listdir(f"{path}/merged/{sample}"):

            events = pd.read_parquet(
            f"{path}/merged/{sample}/{file}", 
            # columns=new_columns, filters=filters
            engine='pyarrow'
            ) 
            lumi_xs = 1
            if "GluGlu" in file:
                lumi_xs = 8.0*0.03413*(2*0.5824*2.270E-03)
            if "VBF" in file:
                lumi_xs = 8.0*0.001886*(2*0.5824*2.270E-03)
            
            events["lumi_xs"] = lumi_xs
            events["totwei"] = events["weight"] * events["lumi_xs"]
            os.makedirs(f"{path}_/merged/{sample}", exist_ok=True)
            # print(events["totwei"])
            events.to_parquet(f"{path}_/merged/{sample}/{file}", engine='pyarrow')
            
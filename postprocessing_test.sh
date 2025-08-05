year=22postEE
parquet_folder=parquet_file
minitree_folder=minitree
mkdir -p ${minitree_folder}
for sample in QCD_Pt30_MGG40to80 GJet_Pt20_MGG40to80
do 
    python3 /home/cosine/GitLab/HiggsDNA/scripts/postprocessing/merge_parquet.py --source ${parquet_folder}/${year}_${sample}/nominal/ --target ${parquet_folder}/${year}_${sample}/
    python3 /home/cosine/GitLab/HiggsDNA/scripts/postprocessing/convert_parquet_to_root.py ${parquet_folder}/${year}_${sample}/merged.parquet ${minitree_folder}/${year}_${sample}.root mc 
done

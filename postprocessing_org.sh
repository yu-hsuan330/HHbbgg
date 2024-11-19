year=22postEE
parquet_folder=parquet_file_genWeight
minitree_folder=minitree_genWeight
mkdir -p ${minitree_folder}
for sample in QCD_HT70to100 QCD_HT100to200 QCD_HT200to400 QCD_HT400to600 QCD_HT600to800 QCD_HT800to1000 QCD_HT1000to1200 QCD_HT1200to1500 QCD_HT1500to2000 QCD_HT2000
do 
    python3 /home/cosine/GitLab/HiggsDNA/scripts/postprocessing/merge_parquet.py --source ${parquet_folder}/${year}_${sample}/nominal/ --target ${parquet_folder}/${year}_${sample}/
    python3 /home/cosine/GitLab/HiggsDNA/scripts/postprocessing/convert_parquet_to_root.py ${parquet_folder}/${year}_${sample}/merged.parquet ${minitree_folder}/${year}_${sample}.root mc 
done


for sample in GluGluToHH VBFToHH GJet_Pt20to40 GJet_Pt40 GGJets QCD_Pt30to40 QCD_Pt40 GluGluHtoGG_M125 VBFHtoGG_M125 VHtoGG_M125 ttHtoGG_M125
do 
    python3 /home/cosine/GitLab/HiggsDNA/scripts/postprocessing/merge_parquet.py --source ${parquet_folder}/${year}_${sample}/nominal/ --target ${parquet_folder}/${year}_${sample}/
    python3 /home/cosine/GitLab/HiggsDNA/scripts/postprocessing/convert_parquet_to_root.py ${parquet_folder}/${year}_${sample}/merged.parquet ${minitree_folder}/${year}_${sample}.root mc 
done


# for data_sample in 22preEE_EGammaC 22preEE_EGammaD 22postEE_EGammaE 22postEE_EGammaF 22postEE_EGammaG
# do 
#     python3 /home/cosine/GitLab/HiggsDNA/scripts/postprocessing/merge_parquet.py --source ${parquet_folder}/${data_sample}/nominal/ --target ${parquet_folder}/${data_sample}/ --is-data
#     python3 /home/cosine/GitLab/HiggsDNA/scripts/postprocessing/convert_parquet_to_root.py ${parquet_folder}/${data_sample}/merged.parquet ${minitree_folder}/data/${data_sample}.root data
# done
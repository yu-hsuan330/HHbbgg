year=22preEE
ver_folder=ver1108

mkdir -p minitree/${ver_folder}/data

#
# python HiggsDNA/scripts/run_analysis.py --json-analysis json_RunFile/run_MC_22postEE_sig.json --skipCQR --skipJetVetoMap --dump parquet/ver0814_test

# post-processing
for sample in GluGluToHH VBFToHH GJet_Pt20_MGG40to80 GJet_Pt20to40_MGG80 GJet_Pt40_MGG80 GGJets QCD_Pt30_MGG40to80 QCD_Pt30to40_MGG80 QCD_Pt40_MGG80 GluGluHtoGG_M125 VBFHtoGG_M125 VHtoGG_M125 ttHtoGG_M125
# DYto2L-2Jets_MLL-50
do 
    python3 /home/cosine/GitLab/HiggsDNA/scripts/postprocessing/merge_parquet.py --source parquet/${ver_folder}/${year}_${sample}/nominal/ --target parquet/${ver_folder}/${year}_${sample}/
    python3 /home/cosine/GitLab/HiggsDNA/scripts/postprocessing/convert_parquet_to_root.py parquet/${ver_folder}/${year}_${sample}/merged.parquet minitree/${ver_folder}/${year}_${sample}.root mc 
done


# for data_sample in 22postEE_EGammaE 22postEE_EGammaF 22postEE_EGammaG #22preEE_EGammaC 22preEE_EGammaD
# do 
#     python3 /home/cosine/GitLab/HiggsDNA/scripts/postprocessing/merge_parquet.py --source parquet/${ver_folder}/${data_sample}/nominal/ --target parquet/${ver_folder}/${data_sample}/ --is-data
#     python3 /home/cosine/GitLab/HiggsDNA/scripts/postprocessing/convert_parquet_to_root.py parquet/${ver_folder}/${data_sample}/merged.parquet minitree/${ver_folder}/data/${data_sample}.root data
# done
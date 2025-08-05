# run HiggsDNA
# python higgs_dna/scripts/run_analysis.py --json-analysis ../../HHbbgg/Shared_file/run_MC_22_VBFHH.json --dump ../../HHbbgg/Shared_file/parquet --doFlow-corrections --fiducialCuts store_flag --Smear-sigma-m --doDeco --executor iterative --nano-version 12 --skipbadfiles

python ../../GitLab/HiggsDNA/higgs_dna/scripts/postprocessing/prepare_output_file.py --input ./parquet/22 --merge --syst --varDict variation.json

# for sample in 22preEE_VBFHHto2B2G_CV_1_C2V_1_C3_1 22postEE_VBFHHto2B2G_CV_1_C2V_1_C3_1 
# do
#     python ../../GitLab/HiggsDNA/higgs_dna/scripts/postprocessing/prepare_output_file.py --input ./parquet/22 --merge #--syst --varDict variation.json
# done

# nohup python higgs_dna/scripts/run_analysis.py --json-analysis ../../HHbbgg/Shared_file/run_MC_postBPix_1.json --dump ../../HHbbgg/Shared_file/parquet/postBPix --doFlow-corrections --fiducialCuts store_flag --Smear-sigma-m --doDeco --executor futures --nano-version 13 --skipbadfiles &> 0513_v13_4.txt&

dir_to_HiggsDNA_dump="/home/cosine/HHbbgg/minitree/0623_v3_add_pairing/22preEE"
path_to_catDict="/home/cosine/GitLab/HiggsDNA/higgs_dna/category.json"

# dir_to_HiggsDNA_dump="/home/cosine/HHbbgg/minitree/data_test"
# path_to_catDict="/home/cosine/GitLab/HiggsDNA/higgs_dna/scripts/postprocessing/config_jsons/HHbbgg/VBFHH_cats.json"

#* Prepare output file for HiggsDNA dump
# usage: python prepare_output_file --input <dir_to_HiggsDNA_dump> --merge --varDict <path_to_varDict> --root --syst --cats --catDict <path_to_catDict> --output <path_to_output_dir>

#* MC samples
# python3 higgs_dna/scripts/postprocessing/prepare_output_file.py --input ${dir_to_HiggsDNA_dump} --type data --merge --root --output ${dir_to_HiggsDNA_dump} #--verbose DEBUG

#* data samples
python3 higgs_dna/scripts/postprocessing/prepare_output_file.py --input ${dir_to_HiggsDNA_dump} --merge --root --cats --catDict ${path_to_catDict} --output ${dir_to_HiggsDNA_dump} #--verbose DEBUG

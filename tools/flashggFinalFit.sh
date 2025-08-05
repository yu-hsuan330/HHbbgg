path=/home/cosine/HHbbgg/minitree/0725/22preEE_cat/root
# path=/home/cosine/HHbbgg/minitree/0702_merge_data_/root

#* set CMSSW environment
# cd ~/HHbbgg/CMSSW_14_1_0_pre4/src/
# cmsenv
# cd flashggFinalFit/
# source setup.sh

#* ROOT file -> workspace
# cd ~/HHbbgg/CMSSW_14_1_0_pre4/src/flashggFinalFit/Trees2WS/

# python3 RunWSScripts.py --inputDir ${path} --inputConfig config_bbgg.py --year 2022preEE --mode trees2ws --batch local
# python3 RunWSScripts.py --inputDir ${path}/Data/ --inputConfig config_bbgg.py --year 2022preEE --mode trees2ws_data --batch local

#* signal fit
# python3 ~/HHbbgg/CMSSW_14_1_0_pre4/src/flashggFinalFit/tools/ConfigTool.py -a set -k inputWSDir -v ${path}/ws_mc --dict sig --inplace

# cd ~/HHbbgg/CMSSW_14_1_0_pre4/src/flashggFinalFit/Signal/
# python3 RunSignalScripts.py --inputConfig config_bbgg.py --mode fTest --modeOpts="--doPlots --nGaussMax 5 --xvar CMS_hgg_mass"
# python3 RunSignalScripts.py --inputConfig config_bbgg.py --mode signalFit --groupSignalFitJobsByCat --modeOpts="--doPlots --skipSystematics --skipVertexScenarioSplit --replacementThreshold 0 --xvar CMS_hgg_mass"
# python3 RunSignalScripts.py --inputConfig config_bbgg.py --mode packageSignal --modeOpts="--mergeYears --exts=bbgg_2022preEE"

#* background fit
# python3 ~/HHbbgg/CMSSW_14_1_0_pre4/src/flashggFinalFit/tools/ConfigTool.py -a set -k inputWS -v ${path}/ws_data/allData_2022preEE.root --dict bkg --inplace

# cd ~/HHbbgg/CMSSW_14_1_0_pre4/src/flashggFinalFit/Background/
# python3 RunBackgroundScripts.py  --inputConfig config_bbgg.py  --mode fTestParallel

#* datacard
cd ~/HHbbgg/CMSSW_14_1_0_pre4/src/flashggFinalFit/Datacard/
# # # python3 RunYields.py --inputWSDirMap 2022preEE=/home/cosine/HHbbgg/minitree/0623_v3_add_pairing/22preEE/root/ws_mc --sigModelWSDir ../Signal/outdir_bbgg_2022preEE --sigModelExt bbgg_2022preEE_GGHH_2022preEE --bkgModelWSDir ../Background/outdir_bbgg --cats auto --procs auto --batch local --queue espresso --ext test
python3 RunYields.py --inputWSDirMap 2022preEE=${path}/ws_mc --sigModelWSDir ~/HHbbgg/CMSSW_14_1_0_pre4/src/flashggFinalFit/Signal/outdir_bbgg_2022preEE --sigModelExt bbgg_2022preEE --bkgModelWSDir ~/HHbbgg/CMSSW_14_1_0_pre4/src/flashggFinalFit/Background/outdir_bbgg --cats vbftag --procs auto --batch local --queue espresso --ext test --mergeYears
python3 makeDatacard.py --ext test --years 2022preEE --analysis bbgg --HH --doMCStatUncertainty

#include <vector>
#include <string>
// #include <sstream>
#include <iostream>
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDataSource.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"
// #include "TLatex.h"
#include "TRandom3.h"
#include "TChain.h"
#include "TTreeIndex.h"
#include "TLorentzVector.h"
#include "TMVA/TreeInference/Forest.hxx"
#include "TMVA/RInferenceUtils.hxx"
#include "TMVA/RReader.hxx"
#include "TMVA/Reader.h"
#include "TXMLEngine.h"

#include "boost/filesystem.hpp" // cling cannot include <filessystem.> (bug) 
#include "nlohmann/json.hpp"

using namespace std;
using namespace ROOT::VecOps;
using namespace TMVA::Experimental;
namespace fs = boost::filesystem;

using RNode = ROOT::RDF::RNode;
using VecI_t = const ROOT::RVec<int>&;
using VecF_t = const ROOT::RVec<float>&; // using cRVecF = const ROOT::RVecF &;
using VecS_t = const ROOT::RVec<size_t>&;

void test(){
    String tree = "DiphotonTree/vbfhh_125_13TeV_untag";
    String filename = "/home/cosine/HHbbgg/minitree/0702_merge_data/root/VBFHHto2B2G_CV_1_C2V_1_C3_1/output_VBFHH_M125_13TeV_madgraph_pythia8.root"
    ROOT::RDataFrame df(tree, filename);
    float ggHH_xs = 0.03413*(2*0.5824*2.270E-03)
    float VBFHH_xs = 0.001886
    auto df1 = df.Define("lumi", to_string(8.0))
                 .Define("xs", to_string(0.001886))
                 .Define("totwei", "weight*lumi*")
    // auto countEvent = df.Sum("n_jets");
    // cout<< Form("nJet = %f",countEvent.GetValue()) <<endl;
    df1.Snapshot(tree, filename);
}

// gghh 0.03413
// vbfhh 0.001886

// BR_H_bb = 0.5824
// BR_H_gg = 2.270E-03
// BR_H_bbgg = 2*BR_H_bb*BR_H_gg
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

    ROOT::RDataFrame df("DiphotonTree/data_125_13TeV_NOTAG","/home/cosine/HHbbgg/minitree_test/22postEE_VBFToHH.root");
    auto countEvent = df.Sum("n_jets");
    
    cout<< Form("nJet = %f",countEvent.GetValue()) <<endl;

}
 
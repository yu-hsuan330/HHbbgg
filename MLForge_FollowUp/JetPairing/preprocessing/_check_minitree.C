#include <vector>
#include <string>

using namespace std;
using namespace ROOT::VecOps;


using RNode = ROOT::RDF::RNode;
using VecI_t = const ROOT::RVec<int>&;
using VecF_t = const ROOT::RVec<float>&; // using cRVecF = const ROOT::RVecF &;
using VecS_t = const ROOT::RVec<size_t>&;

TLorentzVector setTLVec(float pt, float eta, float phi, float mass){
    TLorentzVector v; v.SetPtEtaPhiM(pt, eta, phi, mass);
    return v;
};
// auto DelPtRel(ROOT::RVec<TLorentzVector> vec, TLorentzVector lhe){
//     ROOT::RVec<float> val;
//     for(int i=0; i<vec.size(); i++){
//         val.push_back( (vec[i].Pt()-lhe.Pt())/lhe.Pt() );
//     }
//     return val;
// }
auto DelPtRel(TLorentzVector reco, TLorentzVector lhe){
    return (reco.Pt()-lhe.Pt())/lhe.Pt();
}
auto DeltaR(ROOT::RVec<TLorentzVector> vec, TLorentzVector lhe){
    ROOT::RVec<float> val;
    for(int i=0; i<vec.size(); i++){
        val.push_back(lhe.DeltaR(vec[i]));
    }
    return val;
}
void printRVec(ROOT::RVec<float>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}
// auto nMatched_by_lhe(ROOT::RVec<Double_t> vec){
//     int sum = 0;
//     for(int i=0; i<vec.size(); i++){
//         if(vec[i] > -999) sum += 1;
//     }
//     return sum;
// }

auto nMatched_bFlav(ROOT::RVec<Double_t> vec){
    int sum = 0;
    for(int i=0; i<vec.size(); i++){
        if(abs(int(vec[i])) == 5) sum += 1;
    }
    return sum;
}

auto nMatched_bool(ROOT::RVec<Double_t> vec){
    int sum = 0;
    for(int i=0; i<vec.size(); i++){
        if(vec[i] > -999) sum += vec[i];
    }
    return sum;
}

// auto nMatched(ROOT::RVec<Double_t> vec){
//     int sum = 0;
//     for(int i=0; i<vec.size(); i++){
//         if(vec[i] > -999) sum += vec[i];
//     }
//     return sum;
// }
// auto nMatched_(ROOT::RVec<Double_t> vec){
//     int sum = 0;
//     for(int i=0; i<vec.size(); i++){
//         if(vec[i] > -999) sum += 1;
//     }
//     return sum;
// }
void check_minitree(){
    string version = "ver1009";
    string fileName = "22preEE_VBFToHH"; //22preEE_GluGluToHH 22preEE_VBFToHH 22postEE_GluGluToHH 22postEE_VBFToHH

    ROOT::EnableImplicitMT();
    ROOT::RDataFrame df("DiphotonTree/data_125_13TeV_NOTAG", Form("/home/cosine/HHbbgg/minitree/%s/%s.root", version.c_str(), fileName.c_str()));
    auto df1 = df
                //.Range(1)
                .Define("jet1",                     "setTLVec(jet1_pt, jet1_eta, jet1_phi, jet1_mass)")
                .Define("jet2",                     "setTLVec(jet2_pt, jet2_eta, jet2_phi, jet2_mass)")
                .Define("jet3",                     "setTLVec(jet3_pt, jet3_eta, jet3_phi, jet3_mass)")
                .Define("jet4",                     "setTLVec(jet4_pt, jet4_eta, jet4_phi, jet4_mass)")
                .Define("jet5",                     "setTLVec(jet5_pt, jet5_eta, jet5_phi, jet5_mass)")
                .Define("jet6",                     "setTLVec(jet6_pt, jet6_eta, jet6_phi, jet6_mass)")
                .Define("jet",                      "ROOT::VecOps::RVec<TLorentzVector> v = {jet1, jet2, jet3, jet4, jet5, jet6}; return v;")

                .Define("lhe1",                     "setTLVec(lhe_vbf_parton1_pt, lhe_vbf_parton1_eta, lhe_vbf_parton1_phi, lhe_vbf_parton1_mass)")
                .Define("lhe2",                     "setTLVec(lhe_vbf_parton2_pt, lhe_vbf_parton2_eta, lhe_vbf_parton2_phi, lhe_vbf_parton2_mass)")
                .Define("lhe",                      "ROOT::RVec<TLorentzVector> v = {lhe1, lhe2}; return v;")
                .Define("is_lheMatched1",           "return lhe1.DeltaR(jet);")
                .Define("lheMatch_vec",             "ROOT::RVec<Double_t> v = {jet1_lheMatched, jet2_lheMatched, jet3_lheMatched, jet4_lheMatched, jet5_lheMatched, jet6_lheMatched}; return v;")
                .Define("genMatch_vec",             "ROOT::RVec<Double_t> v = {jet1_genMatched, jet2_genMatched, jet3_genMatched, jet4_genMatched, jet5_genMatched, jet6_genMatched}; return v;")
                .Define("genFlav_vec",              "ROOT::RVec<Double_t> v = {jet1_genFlav, jet2_genFlav, jet3_genFlav, jet4_genFlav, jet5_genFlav, jet6_genFlav}; return v;")
                .Define("genFlavH_vec",             "ROOT::RVec<Double_t> v = {jet1_genFlavH, jet2_genFlavH, jet3_genFlavH, jet4_genFlavH, jet5_genFlavH, jet6_genFlavH}; return v;")
                ;

    auto isbjet = [](Double_t genFlav, bool selected_bjet) { if((abs(int(genFlav)) == 5) && (selected_bjet == true)) return 1; return 0; };
    auto isvjet = [](Double_t genFlav, Double_t lheMatched, bool selected_vbfjet) { if((abs(int(genFlav)) != 5) && (int(lheMatched) == 1) && (selected_vbfjet == true)) return 1; return 0; };
    auto isboth = [](Double_t genFlav, Double_t lheMatched) { if((abs(int(genFlav)) == 5) && (int(lheMatched) == 1)) return 1; return 0; };

    auto df2 = df1
                .Define("jet1_isbjet",  isbjet, {"jet1_genFlav", "jet1_selected_bjet"})
                .Define("jet2_isbjet",  isbjet, {"jet2_genFlav", "jet2_selected_bjet"})
                .Define("jet3_isbjet",  isbjet, {"jet3_genFlav", "jet3_selected_bjet"})
                .Define("jet4_isbjet",  isbjet, {"jet4_genFlav", "jet4_selected_bjet"})
                .Define("jet5_isbjet",  isbjet, {"jet5_genFlav", "jet5_selected_bjet"})
                .Define("jet6_isbjet",  isbjet, {"jet6_genFlav", "jet6_selected_bjet"})

                .Define("jet1_isvjet",  isvjet, {"jet1_genFlav", "jet1_lheMatched", "jet1_selected_vbfjet"})
                .Define("jet2_isvjet",  isvjet, {"jet2_genFlav", "jet2_lheMatched", "jet2_selected_vbfjet"})
                .Define("jet3_isvjet",  isvjet, {"jet3_genFlav", "jet3_lheMatched", "jet3_selected_vbfjet"})
                .Define("jet4_isvjet",  isvjet, {"jet4_genFlav", "jet4_lheMatched", "jet4_selected_vbfjet"})
                .Define("jet5_isvjet",  isvjet, {"jet5_genFlav", "jet5_lheMatched", "jet5_selected_vbfjet"})
                .Define("jet6_isvjet",  isvjet, {"jet6_genFlav", "jet6_lheMatched", "jet6_selected_vbfjet"})

                .Define("jet1_isboth",  isboth, {"jet1_genFlav", "jet1_lheMatched"})
                .Define("jet2_isboth",  isboth, {"jet2_genFlav", "jet2_lheMatched"})
                .Define("jet3_isboth",  isboth, {"jet3_genFlav", "jet3_lheMatched"})
                .Define("jet4_isboth",  isboth, {"jet4_genFlav", "jet4_lheMatched"})
                .Define("jet5_isboth",  isboth, {"jet5_genFlav", "jet5_lheMatched"})
                .Define("jet6_isboth",  isboth, {"jet6_genFlav", "jet6_lheMatched"})

                .Define("n_matched_both",           "jet1_isboth+jet2_isboth+jet3_isboth+jet4_isboth+jet5_isboth+jet6_isboth")

                .Define("n_selected_true_bjet",     "jet1_isbjet+jet2_isbjet+jet3_isbjet+jet4_isbjet+jet5_isbjet+jet6_isbjet")
                .Define("n_selected_bjet",          "jet1_selected_bjet+jet2_selected_bjet+jet3_selected_bjet+jet4_selected_bjet+jet5_selected_bjet+jet6_selected_bjet")
                
                .Define("n_selected_true_vbfjet",   "jet1_isvjet+jet2_isvjet+jet3_isvjet+jet4_isvjet+jet5_isvjet+jet6_isvjet")
                .Define("n_selected_vbfjet",        "jet1_selected_vbfjet+jet2_selected_vbfjet+jet3_selected_vbfjet+jet4_selected_vbfjet+jet5_selected_vbfjet+jet6_selected_vbfjet")

                .Define("n_matched_vbfjet",         "nMatched_bool(lheMatch_vec)")
                .Define("n_matched_genjet",         "nMatched_bool(genMatch_vec)")
                .Define("n_matched_bjet",           "nMatched_bFlav(genFlav_vec)")
                .Define("n_matched_bjet_FlavH",     "nMatched_bFlav(genFlavH_vec)")

                // .Filter("n_matched_both > 0",       "n_matched_both > 0")
                ;

    auto columnNames = df.GetColumnNames();
    columnNames.push_back("n_selected_true_bjet");
    columnNames.push_back("n_selected_bjet");
    columnNames.push_back("n_selected_true_vbfjet");
    columnNames.push_back("n_selected_vbfjet");

    columnNames.push_back("n_matched_vbfjet");
    columnNames.push_back("n_matched_genjet");
    columnNames.push_back("n_matched_bjet");
    columnNames.push_back("n_matched_bjet_FlavH");
    columnNames.push_back("n_matched_both");

    // {"n_selected_true_bjet", "n_selected_bjet", "n_matched_vbfjet", "n_matched_genjet", "n_matched_bjet", "n_matched_bjet_FlavH"}
    df2.Snapshot("Events", Form("%s_test.root", fileName.c_str()), columnNames);
    
    // auto DeltaR1 = df1.Take<RVec<float>>("MinDeltaR1");
    // printRVec(DeltaR1);
    // for(int i=0; i<DeltaR1.size(); i++){
    //     cout<< DeltaR1[i]<<endl;
    // }

    // string var = "MinDeltaR1";
    // auto hist = df1.Histo1D<float>(var.c_str());
    
    // TCanvas pl("pl","pl",700,700); 
    // pl.cd();
    // hist->Draw("HIST");

    // pl.Print(Form("%s.pdf", var.c_str()));
	// pl.Clear();

    df1.Report()->Print();
}
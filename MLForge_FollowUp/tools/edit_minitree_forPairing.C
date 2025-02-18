#include <vector>
#include <string>

using namespace std;
using namespace ROOT::VecOps;


using RNode = ROOT::RDF::RNode;
using VecI_t = const ROOT::RVec<int>&;
using VecF_t = const ROOT::RVec<float>&; // using cRVecF = const ROOT::RVecF &;
using VecS_t = const ROOT::RVec<size_t>&;

TLorentzVector TLVec(float pt, float eta, float phi, float mass){
    TLorentzVector v; v.SetPtEtaPhiM(pt, eta, phi, mass);
    return v;
};

auto Cgg(float jet_eta_diff, float jet_eta_sum, float gg_eta){
    return exp(-4 / pow(jet_eta_diff, 2) * pow(gg_eta - (jet_eta_sum) / 2, 2));
    // np.exp(-4 / (VBFjet_eta_diff) ** 2 * (higgs_eta - (VBFjet_eta_sum) / 2) ** 2)
};

RNode addLHE(RNode df, int hasLHE){
    if(hasLHE){
        return df;
    }
    auto df1 = df
                .Define("jet1_lheMatched",        "return -999;")
                .Define("jet2_lheMatched",        "return -999;")
                .Define("jet3_lheMatched",        "return -999;")
                .Define("jet4_lheMatched",        "return -999;")
                .Define("jet5_lheMatched",        "return -999;")
                .Define("jet6_lheMatched",        "return -999;");
    return df1;
}
void add_VBFVar(string inpath, string infile, string outpath, string pair1, string pair2, float lumi_xs){

    // ROOT::EnableImplicitMT();
    ROOT::RDataFrame df("DiphotonTree/data_125_13TeV_NOTAG", Form("%s/%s.root", inpath.c_str(), infile.c_str()));
    cout<< "hasLHE: " << df.HasColumn("jet1_lheMatched") << endl;

    auto df1 = addLHE(df, df.HasColumn("jet1_lheMatched"));
    auto df_ = df1
                .Define("lumi_xs",                  to_string(lumi_xs))
                .Define("n_selected_bjet",          "jet1_selected_bjet+jet2_selected_bjet+jet3_selected_bjet+jet4_selected_bjet+jet5_selected_bjet+jet6_selected_bjet")
                .Filter("n_selected_bjet == 2",     "in 6 jet collection")
                
                .Define("b_jetPair",                Form("(abs(%s_genFlav) == 5) && (abs(%s_genFlav) == 5)", pair1.c_str(), pair2.c_str()))
                .Define("vbf_jetPair",              Form("(%s_lheMatched == 1) && (abs(%s_genFlav) != 5) && (%s_lheMatched == 1) && (abs(%s_genFlav) != 5)", pair1.c_str(), pair1.c_str(), pair2.c_str(), pair2.c_str()))
                .Define("WrongPair",                "(b_jetPair == 0) && (vbf_jetPair == 0)")

                .Filter(Form("%s_pt > 0 && %s_pt > 0", pair1.c_str(), pair2.c_str()),   "isExistPair")
                
                .Define("pair1",                    Form("TLVec(%s_pt, %s_eta, %s_phi, %s_mass)", pair1.c_str(), pair1.c_str(), pair1.c_str(), pair1.c_str()))
                .Define("pair2",                    Form("TLVec(%s_pt, %s_eta, %s_phi, %s_mass)", pair2.c_str(), pair2.c_str(), pair2.c_str(), pair2.c_str()))
                .Define("pair",                     "pair1 + pair2")

                // .Define("bb_selection",             "abs(pair1.Eta()) < 2.5 &&  abs(pair2.Eta()) < 2.5")
                // .Define("jj_selection",             "pair1.Pt() > 40 && pair2.Pt() > 30")
                
                //* store training variables
                .Define("pair1_pt",                 "pair1.Pt()")
                .Define("pair1_eta",                "pair1.Eta()")
                .Define("pair1_phi",                "pair1.Phi()")
                .Define("pair1_mass",               "pair1.M()")
                .Define("pair1_btagPNetB",          Form("%s_btagPNetB", pair1.c_str()))
                .Define("pair1_btagPNetQvG",        Form("%s_btagPNetQvG", pair1.c_str()))
                .Define("pair1_ptOverM",            "pair1.Pt()/pair.M()")
                .Define("pair1_selected_bjet",      Form("%s_selected_bjet", pair1.c_str()))
                .Define("pair1_selected_vbfjet",    Form("%s_selected_vbfjet", pair1.c_str()))

                .Define("pair2_pt",                 "pair2.Pt()")
                .Define("pair2_eta",                "pair2.Eta()")
                .Define("pair2_phi",                "pair2.Phi()")
                .Define("pair2_mass",               "pair2.M()")
                .Define("pair2_btagPNetB",          Form("%s_btagPNetB", pair2.c_str()))
                .Define("pair2_btagPNetQvG",        Form("%s_btagPNetQvG", pair2.c_str()))
                .Define("pair2_ptOverM",            "pair2.Pt()/pair.M()")
                .Define("pair2_selected_bjet",      Form("%s_selected_bjet", pair2.c_str()))
                .Define("pair2_selected_vbfjet",    Form("%s_selected_vbfjet", pair2.c_str()))

                .Define("pair_pt",                  "pair.Pt()")
                .Define("pair_eta",                 "pair.Eta()")
                .Define("pair_phi",                 "pair.Phi()")
                .Define("pair_mass",                "pair.M()")

                .Define("pair_DeltaR",              "pair1.DeltaR(pair2)")
                .Define("pair_DeltaPhi",            "pair1.DeltaPhi(pair2)")                
                .Define("pair_eta_prod",            "pair1.Eta() * pair2.Eta()")
                .Define("pair_eta_diff",            "pair1.Eta() - pair2.Eta()")
                .Define("pair_eta_sum",             "pair1.Eta() + pair2.Eta()")
                .Define("pair_Cgg",                 "Cgg(pair_eta_diff, pair_eta_sum, eta)")

                .Define("jet1",                     "TLVec(jet1_pt, jet1_eta, jet1_phi, jet1_mass)")
                .Define("jet2",                     "TLVec(jet2_pt, jet2_eta, jet2_phi, jet2_mass)")
                .Define("max_mjj",                  "(jet1 + jet2).M()");


    df_.Report()->Print();
    system(Form("mkdir -p %s", outpath.c_str()));
    df_.Snapshot("DiphotonTree/data_125_13TeV_NOTAG", Form("%s/%s_%s%s.root",outpath.c_str(), infile.c_str(), pair1.c_str(), pair2.c_str()));
}

void edit_minitree_forPairing(){
    std::map<std::string, float> files = {
        // {"22preEE_GluGluToHH",              7980.4  *   0.03443},
        // {"22postEE_GluGluToHH",             26671.7 *   0.03443},
        // {"22preEE_VBFToHH",                 7980.4  *   0.00192},
        // {"22postEE_VBFToHH",                26671.7 *   0.00192},
        // {"22preEE_GGJets",                  7980.4  *   88.75},
        // {"22postEE_GGJets",                 26671.7 *   88.75},
        {"22preEE_GJet_Pt20to40_MGG80",     7980.4  *   242.5},
        {"22postEE_GJet_Pt20to40_MGG80",    26671.7 *   242.5},
        {"22preEE_GJet_Pt40_MGG80",         7980.4  *   919.1},
        {"22postEE_GJet_Pt40_MGG80",        26671.7 *   919.1},
        {"22preEE_GJet_Pt20_MGG40to80",     7980.4  *   3284},
        {"22postEE_GJet_Pt20_MGG40to80",   26671.7  *   3284}
    
        // "22preEE_QCD_Pt30to40_MGG80", "22postEE_QCD_Pt30to40_MGG80", 
        // "22preEE_QCD_Pt40_MGG80", "22postEE_QCD_Pt40_MGG80",
        // "22preEE_QCD_Pt30_MGG40to80", "22postEE_QCD_Pt30_MGG40to80",
        // 7980.4*25950, 26671.7*25950, 
        // 7980.4*124700, 26671.7*124700,
        // 7980.4*252200, 26671.7*252200
    };

    vector<string> jet = {
        "jet1", "jet2", "jet3", "jet4", "jet5", "jet6"
    };

    for(int idx1=0; idx1<jet.size(); idx1++){
        for(int idx2=idx1+1; idx2<jet.size(); idx2++){
            for(const auto& n: files){    
                // std::cout << "key: " << n.first << " value: " << n.second << "\n";
                add_VBFVar("../../minitree/ver0121", n.first, "../../minitree/ver0121_pair", jet[idx1], jet[idx2], n.second);
            }
        }
    }
    
}
 
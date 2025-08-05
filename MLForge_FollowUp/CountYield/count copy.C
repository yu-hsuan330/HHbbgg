#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDataSource.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"

using RNode = ROOT::RDF::RNode;
using VecI_t = const ROOT::RVec<int>&;

RNode JetReform(RNode df){
    auto df1 = df
            .Define("jet_genFlav",          "ROOT::RVec<int> v = {int(jet1_genFlav), int(jet2_genFlav), int(jet3_genFlav), int(jet4_genFlav), int(jet5_genFlav), int(jet6_genFlav)}; return v;")
            .Define("jet_lheMatched",       "ROOT::RVec<int> v = {int(jet1_lheMatched), int(jet2_lheMatched), int(jet3_lheMatched), int(jet4_lheMatched), int(jet5_lheMatched), int(jet6_lheMatched)}; return v;")
            .Define("selected_bjet",        "ROOT::RVec<int> v = {int(jet1_selected_bjet), int(jet2_selected_bjet), int(jet3_selected_bjet), int(jet4_selected_bjet), int(jet5_selected_bjet), int(jet6_selected_bjet)}; return v;")
            .Define("selected_vbfjet",      "ROOT::RVec<int> v = {int(jet1_selected_vbfjet), int(jet2_selected_vbfjet), int(jet3_selected_vbfjet), int(jet4_selected_vbfjet), int(jet5_selected_vbfjet), int(jet6_selected_vbfjet)}; return v;")
            .Define("is_true_bjet",         "(selected_bjet == 1) && (abs(jet_genFlav) == 5)")
            .Define("is_true_vbfjet",       "(selected_vbfjet == 1) && (abs(jet_genFlav) != 5) && (jet_lheMatched == 1)")
            .Define("true_bjet_pair",       "Sum(is_true_bjet) == 2")
            .Define("true_vbfjet_pair",     "Sum(is_true_vbfjet) == 2");
    
    // df1.Snapshot("TH", "test.root", {"jet_genFlav"});
            // df1.Snapshot("TH", "test.root", {"jet_genFlav", "jet_lheMatched", "selected_bjet", "selected_vbfjet", "is_true_bjet", "is_true_vbfjet", "true_bjet_pair", "true_vbfjet_pair"});
            return df1;

}

void GetYield(string file_path_preEE, string file_path_postEE, string xs, string weight, string selection_ggHH, string selection_VBFHH){
    ROOT::RDataFrame df_preEE2("DiphotonTree/data_125_13TeV_NOTAG", file_path_preEE.c_str());
    ROOT::RDataFrame df_postEE2("DiphotonTree/data_125_13TeV_NOTAG", file_path_postEE.c_str());
    auto df_preEE = JetReform(df_preEE2);
    auto df_postEE = JetReform(df_postEE2);
    // df.Filter();
    auto df_preEE1 = df_preEE.Define("weight_lumi_xs", Form("weight*7980.4*%s", xs.c_str()));
    auto df_postEE1 = df_postEE.Define("weight_lumi_xs", Form("weight*26671.7*%s", xs.c_str()));
    // auto df_preEE1 = df_preEE.Define("weight_lumi_xs", "return 1;");
    // auto df_postEE1 = df_postEE.Define("weight_lumi_xs", "return 1;");

    auto ggHH_preEE = df_preEE1.Filter(selection_ggHH.c_str()).Sum("weight_lumi_xs");
    auto ggHH_postEE = df_postEE1.Filter(selection_ggHH.c_str()).Sum("weight_lumi_xs");

    auto VBFHH_preEE = df_preEE1.Filter(selection_VBFHH.c_str()).Sum("weight_lumi_xs");
    auto VBFHH_postEE = df_postEE1.Filter(selection_VBFHH.c_str()).Sum("weight_lumi_xs");

    cout<<" - ggHH cate./VBF cate.: "<<ggHH_preEE.GetValue()+ggHH_postEE.GetValue()<<", "<<VBFHH_preEE.GetValue()+VBFHH_postEE.GetValue()<<endl;
    // cout<<" - total event: "<<ggHH_preEE.GetValue()+ggHH_postEE.GetValue()<<", "<<ggHH_preEE.GetValue()<<", "<<ggHH_postEE.GetValue()<<endl;
    // cout<<" - total event: "<<VBFHH_preEE.GetValue()+VBFHH_postEE.GetValue()<<", "<<VBFHH_preEE.GetValue()<<", "<<VBFHH_postEE.GetValue()<<endl;

}

void count(){
    // ROOT::EnableImplicitMT(10);

    ifstream json_files("./config.json"); //config_purity
    auto conf = nlohmann::json::parse(json_files);

    // vector<string> method = {"my_my", "pair_Nitish", "CutBased", "Nitish", "Nitish_cutbased"};
    vector<string> method = {"Cutbased", "Nitish", }; //"pair_Cutbased", "pair_Nitish", "my_my"
    vector<string> production = {"ggHH", "VBFHH"};
    for(int i = 0; i < method.size(); i++){
        for(int j = 0; j < production.size(); j++){
            cout << method[i] << " " << production[j];
            auto file_path_preEE = conf[method[i]][production[j]]["22preEE"]["file"];
            auto file_path_postEE = conf[method[i]][production[j]]["22postEE"]["file"];
            auto xs = conf[method[i]][production[j]]["xs"];
            auto weight = conf[method[i]][production[j]]["weight"];
            auto selection_ggHH = conf[method[i]][production[j]]["selection_ggHH"];
            auto selection_VBFHH = conf[method[i]][production[j]]["selection_VBFHH"];

            GetYield(file_path_preEE, file_path_postEE, xs, weight, selection_ggHH, selection_VBFHH);
        }
    }
    // auto file_path_preEE = conf["CutBased"]["ggHH"]["22preEE"]["file"];
    // auto file_path_postEE = conf["CutBased"]["ggHH"]["22postEE"]["file"];
    // auto xs = conf["CutBased"]["ggHH"]["xs"];
    // auto weight = conf["CutBased"]["ggHH"]["weight"];
    // auto selection = conf["CutBased"]["ggHH"]["selection"];
    // cout<<file_path_preEE<<endl;  

    

    // auto abseta_range = j["NUM_TrackerMuons_DEN_genTracks"]["abseta_pt"]["binning"][0]["binning"];
    
    // TStopwatch time; time.Start();

}
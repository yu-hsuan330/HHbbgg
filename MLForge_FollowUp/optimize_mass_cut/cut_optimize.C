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

vector<double> GetYield(string file_path_preEE, string file_path_postEE, string xs, string weight, string selection_ggHH, string selection_VBFHH, string mjj_branch, string mass_cut){
    ROOT::RDataFrame df_preEE("DiphotonTree/data_125_13TeV_NOTAG", file_path_preEE.c_str());
    ROOT::RDataFrame df_postEE("DiphotonTree/data_125_13TeV_NOTAG", file_path_postEE.c_str());
    // auto df_preEE = JetReform(df_preEE2); //Filter("true_bjet_pair")
    // auto df_postEE = JetReform(df_postEE2);
    
    auto df_preEE1 = df_preEE.Define("weight_lumi_xs", Form("weight*7980.4*%s", xs.c_str()));
    auto df_postEE1 = df_postEE.Define("weight_lumi_xs", Form("weight*26671.7*%s", xs.c_str()));
    // auto df_preEE1 = df_preEE.Define("weight_lumi_xs", "return 1;");
    // auto df_postEE1 = df_postEE.Define("weight_lumi_xs", "return 1;");

    auto ggHH_preEE = df_preEE1.Filter(selection_ggHH.c_str()).Sum("weight_lumi_xs");
    auto ggHH_postEE = df_postEE1.Filter(selection_ggHH.c_str()).Sum("weight_lumi_xs");

    auto VBFHH_preEE = df_preEE1.Filter(selection_VBFHH.c_str()).Filter(Form("%s>%s", mjj_branch.c_str(), mass_cut.c_str())).Sum("weight_lumi_xs");
    auto VBFHH_postEE = df_postEE1.Filter(selection_VBFHH.c_str()).Filter(Form("%s>%s", mjj_branch.c_str(), mass_cut.c_str())).Sum("weight_lumi_xs");

    vector<double> results = {
        ggHH_preEE.GetValue()+ggHH_postEE.GetValue(), 
        VBFHH_preEE.GetValue()+VBFHH_postEE.GetValue()
    };
    // cout << " - ggHH cate./VBF cate.: " << ggHH_preEE.GetValue()+ggHH_postEE.GetValue() << ", " << VBFHH_preEE.GetValue()+VBFHH_postEE.GetValue() << endl;
    return results;
}

void cut_optimize(){
    // ROOT::EnableImplicitMT(10);

    ifstream json_files("./config.json"); //config_purity
    auto conf = nlohmann::json::parse(json_files);

    // vector<string> method = {"my_my", "pair_Nitish", "CutBased", "Nitish", "Nitish_cutbased"};
    // vector<string> method = {"Cutbased", "Nitish", "pair_Cutbased_discard", "pair_Nitish_discard", "pair_Cutbased", "pair_Nitish", "my_my"}; //
    vector<string> method = {"Cutbased", "pair_Cutbased", "Nitish", "pair_Nitish", "Yu", "pair_Yu", "pair_Yu_before"}; //
    vector<string> production = {"ggHH", "VBFHH"};

    for(int i = 0; i < method.size(); i++){
        vector<float> mass_cut;
        vector<double> cate_ggHH[2]; // 0: ggHH sample, 1: VBFHH sample
        vector<double> cate_VBFHH[2]; // 0: ggHH sample, 1: VBFHH sample

        for(int j = 0; j < production.size(); j++){
            cout << method[i] << " " << production[j]<<endl;
            auto file_path_preEE = conf[method[i]][production[j]]["22preEE"]["file"];
            auto file_path_postEE = conf[method[i]][production[j]]["22postEE"]["file"];
            auto xs = conf[method[i]][production[j]]["xs"];
            auto weight = conf[method[i]][production[j]]["weight"];
            auto selection_ggHH = conf[method[i]][production[j]]["selection_ggHH"];
            auto selection_VBFHH = conf[method[i]][production[j]]["selection_VBFHH"];
            auto mjj_branch = conf[method[i]][production[j]]["mjj_branch"];

            for(int k = 0; k < 2; k++){
                float mass_cut_value = k*400;
                mass_cut.push_back(mass_cut_value);
                
                vector<double> temp = GetYield(file_path_preEE, file_path_postEE, xs, weight, selection_ggHH, selection_VBFHH, mjj_branch, to_string(mass_cut_value));
                if(k == 0){
                    cate_ggHH[j].push_back(temp[0]);
                } 
                else{
                    // the removed events should add back to ggHH cate.
                    cate_ggHH[j].push_back(temp[0]+(cate_VBFHH[j][0]-temp[1]));
                }   
                cate_VBFHH[j].push_back(temp[1]);
            }
            for(int rrr = 0; rrr < cate_ggHH[j].size(); rrr++){
                cout << cate_ggHH[j][rrr] << " " << cate_VBFHH[j][rrr] << endl;
            }
        }

        // TCanvas canvas("c", "c", 800, 600);
        // canvas.SetLogy();

        // TGraph graph1(mass_cut.size(), mass_cut.data(), mass_cut.data());
        // TGraph graph2(mass_cut.size(), mass_cut.data(), mass_cut.data()); 

        // // ggFHH cate.
        // for(int rrr=0; rrr < mass_cut.size(); rrr++){
        //     graph1.SetPointY(rrr, cate_ggHH[0][rrr]);
        //     graph2.SetPointY(rrr, cate_ggHH[1][rrr]);
        // }

        // graph1.SetTitle("ggFHH cate.");
        // graph1.Draw("AP");
        // graph2.Draw("P");
        // graph1.SetMarkerStyle(20);
        // graph2.SetMarkerStyle(21);
        // graph1.SetMaximum(1);
        // graph1.SetMinimum(0);

        // TLegend legend(0.2, 0.75, 0.4, 0.91);
        // legend.AddEntry(&graph1, "ggFHH sample", "p");
        // legend.AddEntry(&graph2, "VBFHH sample", "p");
        // legend.Draw();

        // canvas.SaveAs(Form("%s_cate_ggFHH.pdf", method[i].c_str()));
        // canvas.Clear();

        // TGraph graph3(mass_cut.size(), mass_cut.data(), mass_cut.data());
        // TGraph graph4(mass_cut.size(), mass_cut.data(), mass_cut.data()); 
        // // VBFHH cate.
        // for(int rrr=0; rrr < mass_cut.size(); rrr++){
        //     graph3.SetPointY(rrr, cate_VBFHH[0][rrr]);
        //     graph4.SetPointY(rrr, cate_VBFHH[1][rrr]);
        // }
        // TCanvas canvas2("c", "c", 800, 600);
        // canvas2.SetLogy();

        // graph3.SetTitle("VBFHH cate.");
        // graph3.Draw("AP");
        // graph4.Draw("P");
        // graph3.SetMarkerStyle(20);
        // graph4.SetMarkerStyle(21);
        // graph3.SetMaximum(1);
        // graph3.SetMinimum(0);

        // TLegend legend1(0.2, 0.75, 0.4, 0.91);
        // legend1.AddEntry(&graph3, "ggFHH sample", "p");
        // legend1.AddEntry(&graph4, "VBFHH sample", "p");
        // legend1.Draw();

        // canvas2.SaveAs(Form("%s_cate_VBFHH.pdf", method[i].c_str()));
    }

//  0  0.646    0.260427
// 50  0.656426 0.250001
// 100 0.690801 0.215627
// 150 0.727675 0.178752
// 200 0.756963 0.149464
// 250 0.78026  0.126168
// 300 0.798986 0.107441
// 350 0.813784 0.0926432
// 400 0.826294 0.0801335
// 450 0.836483 0.0699436
// 500 0.844716 0.0617111
// 550 0.851727 0.0546997
// 600 0.857805 0.048622
// 650 0.86305 0.0433769
// 700 0.867958 0.0384689

}
#include "sigmaEff.h"
#include "tdrstyle.C"
#include "CMS_lumi.C"
#include "nlohmann/json.hpp"

void draw_DataMC(TChain *ch, vector<string> Samples, vector<float> xs, float lumi, string var="", string unit="", int binning=50, int min=0, int max=0){
    const int sample_size = Samples.size();

    setTDRStyle();
	TCanvas pl("pl","pl",700,700); 
	pl.cd();

	TH1F *hmc_sig = new TH1F("hmc_sig","", binning, min, max);
	TH1F *hmc = new TH1F("hmc","", binning, min, max);
	TH1F *hdata = new TH1F("hdata","", binning, min, max);
    
    TH1F *hist[sample_size];
    for(int i=0; i<sample_size; i++){
        hist[i] = new TH1F(Samples[i].c_str(), "", binning, min, max);
        if(Samples[i] == "GluGluToHH" || Samples[i] == "VBFToHH"){
            ch[i].Draw(Form("%s>>+hmc_sig",var.c_str()),Form("weight*1000*%f*%f", lumi, xs[i]));
        }
        if(Samples[i] == "data"){
            ch[i].Draw(Form("%s>>hdata",var.c_str()));
        }
        ch[i].Draw(Form("%s>>hist[%d]",var.c_str(), i),Form("weight*1000*%f*%f", lumi, xs[i]));

    }

    delete hmc_sig; delete hmc; delete hdata; delete hist[sample_size];

}

void draw_test(){
    // input the setting json file
    
    ifstream json_sample("../SampleList.json");
    ifstream json_plot("draw_setting.json");
    auto infile = nlohmann::json::parse(json_sample);
    auto pltVar = nlohmann::json::parse(json_plot);
    
    // check how many samples are included
    vector<string> samples = pltVar["sample_list"];
    const int sample_size = samples.size();
    string tree = "DiphotonTree/data_125_13TeV_NOTAG";
    string year = pltVar["year"];
    vector<float> xsec; xsec.clear();
    cout<<infile["GluGluToHH"]["file"][year]<<endl;
    
    TChain *ch[sample_size];

    for(int i=0; i<sample_size; i++){
        ch[i] = new TChain(tree.c_str());
        vector<string> file = infile[samples[i]]["file"][year];
        xsec.push_back(infile[samples[i]]["xs"]);
        // read all the root file
        for(int j=0; j<file.size(); j++){
            ch[i]->Add(file[j].c_str());
        }
        // 
    }
    draw_DataMC(ch[sample_size], samples, xsec, infile["data"]["lumi"][year], "pt", "unit", 100, 0, 50);



}
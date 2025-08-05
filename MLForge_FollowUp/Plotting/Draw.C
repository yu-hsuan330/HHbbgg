// #include "sigmaEff.h"
#include "tdrstyle.C"
#include "CMS_lumi.C"
#include "nlohmann/json.hpp"
void SetHistColor(TH1F *hist, string color, string type="fill", int line = 1){
	if(type == "fill"){
		hist->SetFillColor(TColor::GetColor(color.c_str()));
		hist->SetLineColor(TColor::GetColor(color.c_str()));
	}
	else if(type == "line"){
		hist->SetLineColor(TColor::GetColor(color.c_str()));
		hist->SetLineWidth(4);
		hist->SetLineStyle(line);
	}
}

void draw_MultiHist(std::map<std::string, TH1F> hist, vector<string> histName, TString var, vector<string> unit, float Yscale, string dir){

    TCanvas pl("pl","pl",700,700); 
	pl.cd();

	hist[histName[0]].GetXaxis()->SetTitle(unit[0].c_str());
	hist[histName[0]].GetYaxis()->SetTitle(unit[1].c_str());
	
	float maxY = -99;
	for(int i=0; i<histName.size(); i++){
		// find out the maximum of all histogram
		float max = hist[histName[i]].GetBinContent(hist[histName[i]].GetMaximumBin());
		if(maxY < max) maxY = max;
	}
	// scale y maximum
	hist[histName[0]].SetMaximum(Yscale*maxY);
	
	for(int i=0; i<histName.size(); i++){
		cout<<histName[i]<<Form(":%f", hist[histName[i]].Integral(-1,-1))<<endl;

		// draw style
		string style = (i==0) ? "hist" : "histSAME";
		if(unit[1] == "a.u.") hist[histName[i]].DrawNormalized(style.c_str());
		else hist[histName[i]].Draw(style.c_str());
		
	}

	// legend setting
	TLegend *le1 = new TLegend(0.6,0.65,0.92,0.87);
	le1->SetFillStyle(0);
	le1->SetBorderSize(0);  

	for(int i=0; i<histName.size(); i++){
    	le1->AddEntry(&hist[histName[i]],histName[i].c_str(),"l");
	}

    CMS_lumi(&pl);

    le1->Draw();

	TString output;
	system(Form("mkdir -p %s", dir.c_str()));
	output=Form("./%s/%s.pdf", dir.c_str(), var.ReplaceAll('/', '_').Data());
	pl.Print(output);
	pl.Clear();
}

void MultiHist(string json_file="./SampleList.json", string dir="test"){
    ifstream json_sample(json_file.c_str());
    auto infile = nlohmann::json::parse(json_sample);

    auto hist_list = infile["MultiHist"];
	auto var_list = infile["var_list"];
	vector<string> histNames = infile["MultiHist_list"];
	string era = infile["era"];

	// variable loop
	for(nlohmann::json::iterator it_var = var_list.begin(); it_var != var_list.end(); ++it_var){
		std::map<std::string, TH1F> histmap;
		auto varName = it_var.value()["var"];
		string varTitle = it_var.key();
		int binning = it_var.value()["nbin"];
		vector<float> range = it_var.value()["range"];
		vector<string> unit = it_var.value()["unit"]; 

		// histogram loop
		for(nlohmann::json::iterator it = hist_list.begin(); it != hist_list.end(); ++it){
			
			string histName = it.key();
			vector<string> var = varName[histName];
			
			string tree = it.value()["tree"];
			string weight = it.value()["weight"];
			string histColor = it.value()["histColor"];
			int histLine = it.value()["histLine"];
			string xs = it.value()["xs"];

			TH1F hist(histName.c_str(), "", binning, range[0], range[1]);	
			
			if(era == "22"){
				float lumi = infile["lumi"]["22preEE"];
				string file = it.value()["file"]["22preEE"];
				TChain ch1(tree.c_str());
				ch1.Add(file.c_str());
				ch1.Draw(Form("%s>>%s", var[0].c_str(), histName.c_str()), Form("%s*%s*%f*%s", var[1].c_str() ,weight.c_str(), lumi, xs.c_str()));
				
				lumi = infile["lumi"]["22postEE"];
				file = it.value()["file"]["22postEE"];
				TChain ch2(tree.c_str());
				ch2.Add(file.c_str());
				ch2.Draw(Form("%s>>+%s", var[0].c_str(), histName.c_str()), Form("%s*%s*%f*%s", var[1].c_str(), weight.c_str(), lumi, xs.c_str()));
			}	
			else{
				float lumi = infile["lumi"][era];
				string file = it.value()["file"][era];
				TChain ch(tree.c_str());
				ch.Add(file.c_str());
				ch.Draw(Form("%s>>%s", var[0].c_str(), histName.c_str()), Form("%s*%s*%f*%s", var[1].c_str(), weight.c_str(), lumi, xs.c_str()));
			}	
				


			
			SetHistColor(&hist, histColor, "line", histLine);

			histmap[histName] = hist;
		}

		// draw the histograms
		// histmap, histNames, varTitle, unit, Yscale, dir
		draw_MultiHist(histmap, histNames, varTitle, unit, range[2], dir);

	}
}

void Draw(){
    // string json_file="./Conf/MultiHist_VBF.json";
    string json_file="./Conf/MultiHist.json";
	setTDRStyle();
	MultiHist(json_file, "test");
}

// ,
        // "ggFcate_hgg_mass":{
        //     "var":{
        //         "MutiClass_pair": ["diphoton_mass", "1"],
        //         "MultiClass": ["CMS_hgg_mass", "1"]
        //     },
        //     "unit":["m_{#gamma#gamma} (GeV)", "a.u."],
        //     "range":[100, 140, 1.3],
        //     "nbin": 20
        // }
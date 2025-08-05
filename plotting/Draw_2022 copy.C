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
    float lumi = infile["lumi"][era];

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
			string var = varName[histName];
			string file = it.value()["file"][era];
			string tree = it.value()["tree"];
			string weight = it.value()["weight"];
			string histColor = it.value()["histColor"];
			int histLine = it.value()["histLine"];
			float xs = it.value()["xs"];

			TChain ch(tree.c_str());
			ch.Add(file.c_str());
			
			TH1F hist(histName.c_str(), "", binning, range[0], range[1]);	
			ch.Draw(Form("%s>>%s", var.c_str(), histName.c_str()), Form("%s*%f*%f", weight.c_str(), lumi, xs));
			SetHistColor(&hist, histColor, "line", histLine);

			histmap[histName] = hist;
		}

		// draw the histograms
		// histmap, histNames, varTitle, unit, Yscale, dir
		draw_MultiHist(histmap, histNames, varTitle, unit, range[2], dir);

	}
}

void drawDataMC(std::map<std::string, TH1F> hist, TString var, string unit, float min, float max, string dir){

    setTDRStyle();
    TCanvas pl("pl","pl",700,700); 
	pl.cd();
    
    TPad *pad1 = new TPad("pad1", "pad1", 0, 0.25, 1, 1);
	pad1->SetTopMargin(0.1);
	pad1->SetBottomMargin(0.015);
	pad1->SetRightMargin(0.05);
	pad1->SetLeftMargin(0.125);
	pad1->Draw("A");
	pad1->cd();
	
    THStack *hs = new THStack("hs","");
	// hs->Add(&hist["HGG"]);
	hs->Add(&hist["VBFHtoGG_M125"]);
	hs->Add(&hist["ttHtoGG_M125"]);
	hs->Add(&hist["VHtoGG_M125"]);
	hs->Add(&hist["GluGluHtoGG_M125"]);
	hs->Add(&hist["GGJets"]);	
	// hs->Add(&hist["GJet_Pt40"]);
	// hs->Add(&hist["GJet_Pt20to40"]);
	hs->Add(&hist["DD_QCDGJets"]);
    // hs->Add(&hist["QCD_Pt40"]);
    // hs->Add(&hist["QCD_Pt30_MGG40to80"]);
    // hs->Add(&hist["QCD_Pt30to40"]);
	
	hs->SetMinimum(0.1);

	float maxY = (hist["mc"].GetBinContent(hist["mc"].GetMaximumBin()));
	if(maxY < hist["data"].GetBinContent(hist["data"].GetMaximumBin())) maxY = hist["data"].GetBinContent(hist["data"].GetMaximumBin());
	if(maxY < 100*hist["mc_sig"].GetBinContent(hist["mc_sig"].GetMaximumBin())) maxY = 100*hist["mc_sig"].GetBinContent(hist["mc_sig"].GetMaximumBin());
	
	if(var.Contains("eta") || var.Contains("phi")) hs->SetMaximum(2*maxY);
    else hs->SetMaximum(1.4*maxY);
    
	if(var.Contains("Cbb") || var.Contains("Cgg") || var.Contains("PNetB")){
		pad1->SetLogy(1);
		// hs->SetMinimum(0);
	} 
	hs->SetMaximum(1000000);
	pad1->SetLogy(1);
	SetHistColor(&hist["GGJets"], "#0B66C1");
	SetHistColor(&hist["HGG"], "#BCC2AD");
	SetHistColor(&hist["VHtoGG_M125"], "#37A3D2");
	SetHistColor(&hist["GJet_Pt20to40"], "#37A3D2");
	SetHistColor(&hist["DD_QCDGJets"], "#B8C5D6");
	SetHistColor(&hist["VBFHtoGG_M125"], "#F3C568");
	SetHistColor(&hist["GluGluHtoGG_M125"], "#28AFB0");
	SetHistColor(&hist["ttHtoGG_M125"], "#EC8C6F");

	hs->Draw("hist");

    hs->GetXaxis()->SetLabelSize(0);
	hs->GetYaxis()->SetTitle("Events");
	hs->GetYaxis()->SetTitleOffset(1.3);
	hs->GetYaxis()->SetTitleSize(0.05);
	hs->GetYaxis()->SetLabelSize(0.04);
	
	
    hist["GluGluToHH"].SetLineColor(TColor::GetColor("#FC4622"));
	hist["GluGluToHH"].SetLineWidth(3);
	hist["GluGluToHH"].Scale(1000);
	// hist["GluGluToHH"].SetLineStyle(2);
    hist["GluGluToHH"].Draw("histSAME");

	hist["VBFToHH"].SetLineColor(TColor::GetColor("#9AE3FE"));
	hist["VBFToHH"].SetLineWidth(3);
	hist["VBFToHH"].Scale(10000);
	// hist["VBFToHH"].SetLineStyle(2);
    hist["VBFToHH"].Draw("histSAME");
	
    hist["data"].SetMarkerSize(1);
	hist["data"].SetMarkerStyle(20);
	hist["data"].SetLineColor(1);
	hist["data"].Draw("PESAME");
	
    CMS_lumi(pad1);
    pad1->Modified();

	TLegend *le1 = new TLegend(0.3,0.65,0.92,0.87);
	le1->SetNColumns(2);
	le1->SetFillStyle(0);
	le1->SetBorderSize(0);
    // le1->AddEntry(&hist["QCD_Pt40"], "QCD (Pt40 MGG80)","f");

    // le1->AddEntry(&hist["QCD_Pt30to40"],"QCD (Pt30-40 MGG80)","f");
	le1->AddEntry(&hist["DD_QCDGJets"],"DD background","f");
	le1->AddEntry(&hist["GGJets"], "#gamma#gamma+jets","f");
    // le1->AddEntry(&hist["QCD_Pt30_MGG40to80"], "QCD (Pt30 MGG40-80)","f");

	// le1->AddEntry(&hist["GJet_Pt40"], "#gamma+jets","f");
	// le1->AddEntry(&hist["HGG"], "H#rightarrow#gamma#gamma","f");
	le1->AddEntry(&hist["GluGluHtoGG_M125"], "ggH(#gamma#gamma)","f");
	le1->AddEntry(&hist["VBFHtoGG_M125"], "VBFH(#gamma#gamma)","f");
	le1->AddEntry(&hist["VHtoGG_M125"], "VH(#gamma#gamma)","f");
	le1->AddEntry(&hist["ttHtoGG_M125"], "ttH(#gamma#gamma)","f");
	le1->AddEntry(&hist["GluGluToHH"], "ggHH#times1000","L");
	le1->AddEntry(&hist["VBFToHH"], "VBFHH#times10000","L");

	le1->AddEntry(&hist["data"], "Data","PE");	

    le1->Draw();
	pl.cd(); 

	TH1 *hrMC = (TH1F*)hist["mc"].Clone();
	TH1 *hrdata = (TH1F*)hist["data"].Clone();
	hrdata->Divide(hrMC);

	TPad *pad2 = new TPad("pad2","pad2", 0, 0, 1, 0.25);

	pad2->SetTopMargin(0.03);
	pad2->SetBottomMargin(0.35);
	pad2->SetRightMargin(0.05);
	pad2->SetLeftMargin(0.125);

	pad2->Draw("A");
	pad2->cd();

	hrdata->SetMarkerSize(1);
	hrdata->SetMarkerStyle(20);
	hrdata->SetLineColor(1);
	hrdata->GetYaxis()->SetTitle("Data/MC");
	hrdata->GetYaxis()->SetNdivisions(505);
	hrdata->GetYaxis()->CenterTitle();
	hrdata->GetYaxis()->SetTitleOffset(0.46);
	hrdata->GetYaxis()->SetTitleSize(0.135);
	hrdata->GetYaxis()->SetLabelSize(0.12);
	
	hrdata->GetXaxis()->SetTitle(unit.c_str());
	//hrdata->GetXaxis()->SetNdivisions(505);
	hrdata->GetXaxis()->SetTitleOffset(0.9);
	hrdata->GetXaxis()->SetTitleSize(0.15);
	hrdata->GetXaxis()->SetLabelSize(0.12);
	hrdata->SetMinimum(-0.5+(hrdata->GetBinContent(hrdata->GetMinimumBin())));
	if(hrdata->GetMinimum()<0) hrdata->SetMinimum(0);

	hrdata->SetMaximum(0.3+(hrdata->GetBinContent(hrdata->GetMaximumBin())));
	// hrdata->SetMinimum(0.5);
	
	if(hrdata->GetMaximum()>2) hrdata->SetMaximum(2);
	
	hrdata->Draw("PE");
	TLine *line1 = new TLine(min,1,max,1);
	TLine *line2 = new TLine(min,0.5,max,0.5);
	TLine *line3 = new TLine(min,1.5,max,1.5);
	TLine *line4 = new TLine(min,2,max,2);
	TLine *line5 = new TLine(min,3,max,3);
	line1->SetLineWidth(2);
	line1->SetLineStyle(3);
	line2->SetLineWidth(2);
	line2->SetLineStyle(3); 
	line3->SetLineWidth(2);
	line3->SetLineStyle(3);
	line4->SetLineWidth(2);
	line4->SetLineStyle(3);
	line5->SetLineWidth(2);
	line5->SetLineStyle(3);
	line1->Draw("SAME");

	if(hrdata->GetMinimum()<0.5) line2->Draw("SAME");
	if(hrdata->GetMaximum()>1) line3->Draw("SAME");
	// if(hrdata->GetMaximum()>2) line4->Draw("SAME");
	// if(hrdata->GetMaximum()>3) line5->Draw("SAME");

	TString output;
	system(Form("mkdir -p %s", dir.c_str()));
	output=Form("./%s/%s.pdf", dir.c_str(), var.ReplaceAll('/', '_').Data());
	pl.Print(output);
	delete hrdata; delete hrMC;
	pl.Clear();
}

void make_DataMC(std::map<std::string, TH1F> &histmap, string json_file="./SampleList.json", string var="pt", int binning=50, float min=0, float max=400){
    ifstream json_sample(json_file.c_str());
    auto infile = nlohmann::json::parse(json_sample);
    vector<string> era = {"22preEE", "22postEE"};
    string tree = infile["tree"]["Data"];
    string file = "";
	string MC_weight = infile["weight"]["MC"];
	string Data_weight = infile["weight"]["data"];
    
	string basic_selection = "(CMS_hgg_mass > 100 && CMS_hgg_mass < 180 && nonRes_dijet_mass > 0 && lead_mvaID > -0.7 && sublead_mvaID > -0.7)";
    
	//* Data
    TChain data(tree.c_str());
	for(int j = 0; j < 2; j++){
		file = infile["Data"]["file"][era[j]];
		data.Add(file.c_str());
	}

	TH1F hdata("hdata","", binning, min, max);
	
	string additional_selection = "";
	if(var.find("CMS_hgg_mass") != std::string::npos) additional_selection = "(CMS_hgg_mass < 115 || CMS_hgg_mass > 135)";
	else if(var.find("nonRes_dijet_mass") != std::string::npos) additional_selection = "(nonRes_dijet_mass < 105 || nonRes_dijet_mass > 145)";
	else additional_selection = "1";
	data.Draw(Form("%s>>+hdata", var.c_str()), Form("%s*%s*%s", basic_selection.c_str(), additional_selection.c_str(), Data_weight.c_str()));
	
	//* MC
    // check how many samples are included
    const int sample_size = infile["MC"].size();
	
	TH1F hmc("hmc","", binning, min, max);
    // MC --> histigram 
    int i = 0;
    auto MC = infile["MC"];
    tree = infile["tree"]["MC"];
    for(nlohmann::json::iterator it = MC.begin(); it != MC.end(); ++it) {
        string sample = it.key();
		float xs = it.value()["xs"];
		TH1F hist(sample.c_str(), "", binning, min, max);

		for(int j = 0; j < 2; j++){
			float lumi = infile["Data"]["lumi"][era[j]];
			file = it.value()["file"][era[j]];
			TChain ch(tree.c_str());
			ch.Add(file.c_str());

			// fill hist
			if(sample == "DD_QCDGJets"){
				ch.Draw(Form("%s>>+%s", var.c_str(), sample.c_str()), Form("%s*%s", basic_selection.c_str(), MC_weight.c_str()));
				ch.Draw(Form("%s>>+hmc", var.c_str()), Form("%s*%s", basic_selection.c_str(), MC_weight.c_str()));
			}
			else{
				ch.Draw(Form("%s>>+%s", var.c_str(), sample.c_str()), Form("%s*%s*%f*%f", basic_selection.c_str(), MC_weight.c_str(), lumi, xs));
				ch.Draw(Form("%s>>+hmc", var.c_str()), Form("%s*%s*%f*%f", basic_selection.c_str(), MC_weight.c_str(), lumi, xs));
			}
		}

        histmap[sample] = hist;		
        i++;
    }
    histmap["data"] = hdata;
    histmap["mc"] = hmc;
}


void Draw_2022(){
    // string json_file="./Conf/MultiHist_1009_matching.json";
    string SampleList = "./Conf/SampleList_v2.json";
	
	setTDRStyle();

	ifstream json_sample("./var.json");
    auto infile = nlohmann::json::parse(json_sample);
	for(nlohmann::json::iterator it = infile.begin(); it != infile.end(); ++it) {
		std::map<std::string, TH1F> histmap;
		string varName = it.key();
		string unit = it.value()["unit"];
		int binning = it.value()["nbin"];
		float min = it.value()["min"];
		float max = it.value()["max"];
		make_DataMC(histmap, SampleList, varName, binning, min, max);
		drawDataMC(histmap, varName, unit, min, max, "22_0612");
	}
	
	// vector<vector<int>> hist_range = {};
	// std::map<std::string, TH1F> histmap;
	// make_DataMC(histmap, json_file, "nonRes_dijet_mass", 48, 70, 190);
	// drawDataMC(histmap, "nonRes_dijet_mass", "H(bb) mass [GeV]", 70, 190, "22");

	// make_DataMC(histmap, json_file, "CMS_hgg_mass", 32, 100, 180);
	// drawDataMC(histmap, "CMS_hgg_mass", "H(#gamma#gamma) mass [GeV]", 100, 180, "22");

	// make_DataMC(histmap, json_file, "nonRes_dijet_pt", 51, 0, 500);
	// drawDataMC(histmap, "nonRes_dijet_pt", "dijet p_{T} [GeV]", 0, 500, "22");

	// make_DataMC(histmap, json_file, "pt", 51, 0, 400);
	// drawDataMC(histmap, "pt", "diphoton p_{T} [GeV]", 0, 400, "22");

}
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
	// pl.SetLogy();
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
	// hist[histName[0]].SetMaximum(1000000);
	// hist[histName[0]].SetMinimum(40000);
	hist[histName[0]].SetMinimum(10);
	
	
	for(int i=0; i<histName.size(); i++){
		cout<<histName[i]<<Form(":%f", hist[histName[i]].Integral(-1,-1))<<endl;

		// draw style
		string style = (i==0) ? "hist" : "histSAME";
		if(unit[1] == "a.u.") hist[histName[i]].DrawNormalized(style.c_str());
		else hist[histName[i]].Draw(style.c_str());
		
	}

	// legend setting
	TLegend *le1 = new TLegend(0.58,0.68,0.92,0.87);
	// TLegend *le1 = new TLegend(0.58,0.63,0.92,0.87);
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
	hs->Add(&hist["GGJets"]);	
	hs->Add(&hist["GJet_Pt40"]);
	hs->Add(&hist["GJet_Pt20to40"]);
	hs->Add(&hist["GJet_Pt20_MGG40to80"]);
    hs->Add(&hist["QCD_Pt40"]);
    hs->Add(&hist["QCD_Pt30_MGG40to80"]);
    hs->Add(&hist["QCD_Pt30to40"]);

	hs->Add(&hist["HGG"]);

	float maxY = (hist["mc"].GetBinContent(hist["mc"].GetMaximumBin()));
	if(maxY < hist["data"].GetBinContent(hist["data"].GetMaximumBin())) maxY = hist["data"].GetBinContent(hist["data"].GetMaximumBin());
	if(maxY < 100*hist["mc_sig"].GetBinContent(hist["mc_sig"].GetMaximumBin())) maxY = 100*hist["mc_sig"].GetBinContent(hist["mc_sig"].GetMaximumBin());
	
	if(var.Contains("eta") || var.Contains("phi")) hs->SetMaximum(2*maxY);
    else hs->SetMaximum(1.4*maxY);
    
	if(var.Contains("Cbb") || var.Contains("Cgg") || var.Contains("PNetB")){
		pad1->SetLogy(1);
		// hs->SetMinimum(0);
	} 

	SetHistColor(&hist["GGJets"], "#0B66C1");
	SetHistColor(&hist["HGG"], "#BCC2AD");
	SetHistColor(&hist["GJet_Pt40"], "#37A3D2");
	SetHistColor(&hist["GJet_Pt20to40"], "#37A3D2");
	SetHistColor(&hist["GJet_Pt20_MGG40to80"], "#37A3D2");
	SetHistColor(&hist["QCD_Pt40"], "#F3C568");
	SetHistColor(&hist["QCD_Pt30to40"], "#70877F");
	SetHistColor(&hist["QCD_Pt30_MGG40to80"], "#EC8C6F");

	hs->Draw("hist");

    hs->GetXaxis()->SetLabelSize(0);
	hs->GetYaxis()->SetTitle("Events");
	hs->GetYaxis()->SetTitleOffset(1.3);
	hs->GetYaxis()->SetTitleSize(0.05);
	hs->GetYaxis()->SetLabelSize(0.04);
	
	
    hist["mc_sig"].SetLineColor(TColor::GetColor("#EF5B5B"));
	hist["mc_sig"].SetLineWidth(3);
	hist["mc_sig"].Scale(100);
    hist["mc_sig"].Draw("histSAME");

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
	le1->AddEntry(&hist["data"],"Data","PE");	
    le1->AddEntry(&hist["QCD_Pt40"],"QCD (Pt40 MGG80)","f");

	le1->AddEntry(&hist["mc_sig"],"signal#times100","L");
    le1->AddEntry(&hist["QCD_Pt30to40"],"QCD (Pt30-40 MGG80)","f");

	le1->AddEntry(&hist["GGJets"],"#gamma#gamma+jets","f");
    le1->AddEntry(&hist["QCD_Pt30_MGG40to80"],"QCD (Pt30 MGG40-80)","f");

	le1->AddEntry(&hist["GJet_Pt40"],"#gamma+jets","f");
	le1->AddEntry(&hist["HGG"],"H#rightarrow#gamma#gamma","f");

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
			// ch.Draw(Form("%s>>%s", var.c_str(), histName.c_str()));
			SetHistColor(&hist, histColor, "line", histLine);

			histmap[histName] = hist;
		}

		// draw the histograms
		// histmap, histNames, varTitle, unit, Yscale, dir
		draw_MultiHist(histmap, histNames, varTitle, unit, range[2], dir);

	}

	


}

void make_DataMC(std::map<std::string, TH1F> &histmap, string json_file="./SampleList.json", string era="22postEE", string var="pt", int binning=50, float min=0, float max=400){
    ifstream json_sample(json_file.c_str());
    auto infile = nlohmann::json::parse(json_sample);
    

    string tree = infile["tree"]["Data"];
    string file = infile["Data"]["file"][era];
	string MC_weight = infile["weight"]["MC"];
	string Data_weight = infile["weight"]["data"];
    float lumi = infile["Data"]["lumi"][era];
    
    //* Data
    TChain data(tree.c_str());
    data.Add(file.c_str());

	TH1F hdata("hdata","", binning, min, max);
    if(var.find("CMS_hgg_mass") != std::string::npos) data.Draw(Form("%s>>hdata", var.c_str()), Form("%s*(CMS_hgg_mass<115||CMS_hgg_mass>135)", Data_weight.c_str()));
	else if(var.find("dijet_mass") != std::string::npos) data.Draw(Form("%s>>hdata", var.c_str()), Form("%s*(dijet_mass<105||dijet_mass>145)", Data_weight.c_str()));
	else data.Draw(Form("%s>>hdata", var.c_str()), Data_weight.c_str());
    //* MC
    // check how many samples are included
    const int sample_size = infile["MC"].size();
	
   	TH1F hmc_sig("hmc_sig","", binning, min, max);
	TH1F hmc("hmc","", binning, min, max);
	TH1F hHGG("hHGG","", binning, min, max);
	TH1F hQCDHT("hQCDHT","", binning, min, max);

    // MC --> histigram 
    int i = 0;
    auto MC = infile["MC"];
    tree = infile["tree"]["MC"];
    for (nlohmann::json::iterator it = MC.begin(); it != MC.end(); ++it) {
        file = it.value()["file"][era];
        string sample = it.key();
        float xs = it.value()["xs"];
		
		TChain ch(tree.c_str());
		ch.Add(file.c_str());
        
        // fill hist
		TH1F hist(sample.c_str(), "", binning, min, max);
		//TODO optimize!!
		ch.Draw(Form("%s>>%s", var.c_str(), sample.c_str()), Form("%s*%f*%f", MC_weight.c_str(), lumi, xs));
        histmap[sample] = hist;

        // fill overall hist
        if(sample == "GluGluToHH" || sample == "VBFToHH"){
            ch.Draw(Form("%s>>+hmc_sig", var.c_str()), Form("%s*%f*%f", MC_weight.c_str(), lumi, xs));
        }
        if(sample.find("M125") != std::string::npos){
            ch.Draw(Form("%s>>+hHGG", var.c_str()), Form("%s*%f*%f", MC_weight.c_str(), lumi, xs));
        }
		if(sample.find("QCD_HT") != std::string::npos){
			ch.Draw(Form("%s>>+hQCDHT", var.c_str()), Form("%s*%f*%f", MC_weight.c_str(), lumi, xs));
		}
		ch.Draw(Form("%s>>+hmc", var.c_str()), Form("%s*%f*%f", MC_weight.c_str(), lumi, xs));

        i++;
    }

    histmap["data"] = hdata;
    histmap["mc"] = hmc;
    histmap["mc_sig"] = hmc_sig;
    histmap["HGG"] = hHGG;
    histmap["QCD_HT"] = hQCDHT;
}


void Draw(){
    string json_file="./MultiHist_Jetpairing_vbfjet1115.json"; //MultiHist_Jetpairing_vbfjet MultiHist_Jetpairing_withoutNorm MultiHist_Jetpairing_bjet1115
	setTDRStyle();
	MultiHist(json_file, "ver1115");
}

//MultiHist_Jetpairing_DNNscore_jet4
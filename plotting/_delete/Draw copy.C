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

void drawMultiHist(std::map<std::string, TH1F> hist, vector<string> histName, vector<string> histColor, vector<int> histLine, TString var, string unit, float min, float max, string dir){


    TCanvas pl("pl","pl",700,700); 
	pl.cd();
	
	TLegend *le1 = new TLegend(0.6,0.65,0.92,0.87);
	// le1->SetNColumns(2);
	le1->SetFillStyle(0);
	le1->SetBorderSize(0);

	hist[histName[0]].GetYaxis()->SetTitle("a.u.");
	hist[histName[0]].GetXaxis()->SetTitle(unit.c_str());

	float maxY = -99;
	for(int i=0; i<histName.size(); i++){
		if(maxY < hist[histName[i]].GetBinContent(hist[histName[i]].GetMaximumBin())) maxY = hist[histName[i]].GetBinContent(hist[histName[i]].GetMaximumBin());
	}

	if(var.Contains("eta") || var.Contains("phi")){
		hist[histName[0]].SetMaximum(1.3*maxY);
		if(var.Contains("dijet")) hist[histName[0]].SetMaximum(2*maxY);
		if(var.Contains("lead_bjet")) hist[histName[0]].SetMaximum(2*maxY);
	}
	else if(var.Contains("dijet")) hist[histName[0]].SetMaximum(2*maxY);
	else if(var.Contains("lead_bjet")) hist[histName[0]].SetMaximum(2*maxY);
    else hist[histName[0]].SetMaximum(2*maxY);
    
	vector<string> leg = {"RECO", "calib RECO", "GEN"};
	for(int i=0; i<histName.size(); i++){
		SetHistColor(&hist[histName[i]], histColor[i], "line", histLine[i]);
		if(i==0) hist[histName[i]].DrawNormalized("hist");
		else hist[histName[i]].DrawNormalized("histSAME");

    	le1->AddEntry(&hist[histName[i]],histName[i].c_str(),"l");

	}

    CMS_lumi(&pl);


	// le1->AddEntry(&hist["GluGluToHH"],"GluGluToHH","f");	
    // le1->AddEntry(&hist["GluGluToHH_calib"],"calib GluGluToHH","f");
	// le1->AddEntry(&hist["VBFToHH"],"VBFToHH","f");	
    // le1->AddEntry(&hist["VBFToHH_calib"],"calib VBFToHH","f");
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

void make_hist_pt(std::map<std::string, TH1F> &histmap, string json_file="./SampleList.json", string era="22postEE", string var="pt", int binning=50, float min=0, float max=400){
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
	vector<string> var_pt = {"pt", "pt_Calib", "gen_pt"}; //

    for (nlohmann::json::iterator it = MC.begin(); it != MC.end(); ++it) {
        file = it.value()["file"][era];
        string sample = it.key();
        float xs = it.value()["xs"];
		
		TChain ch(tree.c_str());
		ch.Add(file.c_str());
        
		for(int j=0; j<var_pt.size(); j++){
			TH1F hist(var_pt[j].c_str(), "", binning, min, max);	
			// ch.Draw(Form("fabs(%s%s-%sgen_pt)/%sgen_pt>>%s", var.c_str(), var_pt[j].c_str(), var.c_str(), var.c_str(), var_pt[j].c_str()), Form("%s*%f*%f", MC_weight.c_str(), lumi, xs));
			ch.Draw(Form("%s%s>>%s", var.c_str(), var_pt[j].c_str(), var_pt[j].c_str()), Form("%s*%f*%f", MC_weight.c_str(), lumi, xs));
			histmap[var_pt[j]] = hist;
		}		
        i++;
    }
}
void make_hist_pt_(std::map<std::string, TH1F> &histmap, string json_file="./SampleList.json", string era="22postEE", string var="pt", int binning=50, float min=0, float max=400){
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
	vector<string> var_pt = {"pt", "genjet_pt", "gen_pt"}; //

    for (nlohmann::json::iterator it = MC.begin(); it != MC.end(); ++it) {
        file = it.value()["file"][era];
        string sample = it.key();
        float xs = it.value()["xs"];
		
		TChain ch(tree.c_str());
		ch.Add(file.c_str());
        
		// for(int j=0; j<var_pt.size(); j++){
			TH1F hist(sample.c_str(), "", binning, min, max);	
			// ch.Draw(Form("fabs(%s%s-%sgen_pt)/%sgen_pt>>%s", var.c_str(), sample.c_str(), var.c_str(), var.c_str(), sample.c_str()), Form("%s*%f*%f", MC_weight.c_str(), lumi, xs));
			ch.Draw(Form("%s%s>>%s", var.c_str(), sample.c_str(), sample.c_str()), Form("%s*%f*%f", MC_weight.c_str(), lumi, xs));
			if(sample == "eta") histmap["Jet"] = hist;
			else histmap["genJet"] = hist;
		
		TH1F hist1("genPart", "", binning, min, max);	
		ch.Draw(Form("%sgen_phi>>genPart", var.c_str()), Form("%s*%f*%f", MC_weight.c_str(), lumi, xs));
		histmap["genPart"] = hist1;

		// }		
        i++;
    }
}
void make_hist_gen(std::map<std::string, TH1F> &histmap, string json_file="./SampleList.json", string era="22postEE", string var="pt", string var2="pt", int binning=50, float min=0, float max=400){
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
        TH1F hist(sample.c_str(), "", binning, min, max);	
		ch.Draw(Form("%s>>%s", var.c_str(), sample.c_str()), Form("%s*%f*%f", MC_weight.c_str(), lumi, xs));
		histmap["RECO"] = hist;

	    TH1F hist2("h2", "", binning, min, max);	
		ch.Draw(Form("%s>>%s", var2.c_str(), "h2"), Form("%s*%f*%f", MC_weight.c_str(), lumi, xs));
		histmap["GEN"] = hist2;	
        i++;
    }
}
void make_hist(std::map<std::string, TH1F> &histmap, string json_file="./SampleList.json", string era="22postEE", string var="pt", int binning=50, float min=0, float max=400){
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
    // string json_file="./SampleList.json";
    string json_file="./SampleList_gen.json";
    
	ifstream json_sample(json_file.c_str());
    auto infile = nlohmann::json::parse(json_sample);
    vector<string> histName = infile["setting"]["histName"];
    vector<string> histColor = infile["setting"]["histColor"];
	vector<int> histLine = infile["setting"]["LineStyle"];
	string era = infile["setting"]["era"];
    // vector<vector<string>> var = {
	// 	// gg part
	// 	{"pt", "p_{T}^{#gamma#gamma} (GeV)"},
	// 	{"CMS_hgg_mass", "m_{#gamma#gamma} (GeV)"},
    //     {"eta", "#eta_{#gamma#gamma}"},
    //     {"phi", "#phi_{#gamma#gamma}"},		
		
	// 	{"lead_pt", "lead #gamma p_{T} (GeV)"},
	// 	{"sublead_pt", "sublead #gamma p_{T} (GeV)"},
    //     {"lead_eta", "lead #gamma #eta"},
    //     {"sublead_eta", "sublead #gamma #eta"},
    //     {"lead_phi", "lead #gamma #phi"},
    //     {"sublead_phi", "sublead #gamma #phi"},		

	// 	// bb part
    //     {"dijet_pt", "p_{T}^{bb} (GeV)"},
    //     {"dijet_mass", "m_{bb} (GeV)"},
    //     {"dijet_eta", "#eta_{bb}"},
    //     {"dijet_phi", "#phi_{bb}"},

    //     {"lead_bjet_pt", "lead b p_{T}(GeV)"},
    //     {"sublead_bjet_pt", "sublead b p_{T}(GeV)"},
    //     {"lead_bjet_eta", "lead b #eta"},
    //     {"sublead_bjet_eta", "sublead b #eta"},
    //     {"lead_bjet_phi", "lead b #phi"},
    //     {"sublead_bjet_phi", "sublead b #phi"},
    //     {"lead_bjet_btagPNetB", "lead b btagPNetB"},
    //     {"sublead_bjet_btagPNetB", "sublead b btagPNetB"},		

	// 	// bbgg part
    //     {"HHbbggCandidate_pt", "p_{T}^{HH} (GeV)"},
    //     {"HHbbggCandidate_mass", "m_{HH} (GeV)"},
    //     {"HHbbggCandidate_eta", "#eta_{HH}"},
    //     {"HHbbggCandidate_phi", "#phi_{HH}"},
		
	// 	// VBF jets part
	// 	{"VBF_first_jet_pt", "VBF lead jet p_{T}"},
	// 	{"VBF_second_jet_pt", "VBF sublead jet p_{T}"},
	// 	{"VBF_first_jet_pt/VBF_dijet_mass", "VBF lead jet p_{T}/m_{jj}"},
	// 	{"VBF_second_jet_pt/VBF_dijet_mass", "VBF sublead jet p_{T}/m_{jj}"},
	// 	{"VBF_dijet_pt", "VBF dijet p_{T}"},

	// 	{"VBF_first_jet_eta", "VBF lead jet #eta"},
	// 	{"VBF_second_jet_eta", "VBF sublead jet #eta"},
	// 	{"VBF_dijet_eta", "VBF dijet #eta"},

	// 	{"VBF_first_jet_phi", "VBF lead jet #phi"},
	// 	{"VBF_second_jet_phi", "VBF sublead jet #phi"},
	// 	{"VBF_dijet_phi", "VBF dijet #phi"},

	// 	{"VBF_dijet_mass", "VBF dijet m_{jj}"},

	// 	{"VBF_jet_eta_prod", "#eta_{j1}#times#eta_{j2}"},
	// 	{"abs(VBF_jet_eta_diff)", "|#Delta#eta|"},
	// 	{"VBF_DeltaR_jb_min", "min #Delta R(j, b)"},
	// 	{"VBF_DeltaR_jg_min", "min #Delta R(j, #gamma)"},
	// 	{"VBF_Cgg", "Centrality C_{#gamma#gamma}"},
	// 	{"VBF_Cbb", "Centrality C_{bb}"},
	// 	{"VBF_first_jet_btagDeepFlav_QG", "VBF lead jet QGL"},		
	// 	{"VBF_second_jet_btagDeepFlav_QG", "VBF sublead jet QGL"},	
		
	// };
	// vector<vector<float>> range = {
	// 	// gg part pt-mass-eta-phi
    //     {40, 0, 400},
    //     {40, 110, 150},
    //     {25, -5, 5},
    //     {10, -3.2, 3.2},

    //     {40, 35, 200},
    //     {40, 25, 120},
    //     {20, -2.6, 2.6},
    //     {20, -2.6, 2.6},
    //     {20, -3.2, 3.2},
    //     {20, -3.2, 3.2},

	// 	// bb part
    //     {40, 0, 300},
    //     {24, 70, 190},
    //     {25, -5, 5},
    //     {20, -3.2, 3.2},

    //     {50, 25, 250},
    //     {50, 25, 120},
    //     {20, -2.7, 2.7},
    //     {20, -2.7, 2.7},
    //     {20, -3.2, 3.2},
    //     {20, -3.2, 3.2},
    //     {20, 0, 1},
    //     {20, 0, 1},

	// 	// bbgg part
    //     {40, 0, 300},
    //     {40, 150, 700},
    //     {40, -6, 6},
    //     {20, -3.2, 3.2},

	// 	// VBF jets part
    //     {40, 40, 350},
    //     {40, 30, 140},
    //     {40, 0, 2.5},
    //     {40, 0, 1.5},
    //     {40, 0, 400},

    //     {25, -5, 5},
    //     {25, -5, 5},
    //     {25, -5, 5},

    //     {10, -3.2, 3.2},
    //     {10, -3.2, 3.2},
    //     {10, -3.2, 3.2},

	// 	{50, 0, 2000},

	// 	{30, -15, 15},
    //     {20, 0, 8},
    //     {40, 0, 5},
    //     {40, 0, 5},
	// 	{40, 0, 1},
	// 	{40, 0, 1},
	// 	{30, 0, 1},
	// 	{30, 0, 1},
	// };

    // vector<vector<string>> var_ = {
	// 	// gg part
	// 	{"lead_pt", "lead_gen_pt", "lead #gamma p_{T} (GeV)"},
	// 	{"sublead_pt", "sublead_gen_pt", "sublead #gamma p_{T} (GeV)"},
    //     {"lead_eta", "lead_gen_eta", "lead #gamma #eta"},
    //     {"sublead_eta", "sublead_gen_eta", "sublead #gamma #eta"},
    //     {"lead_phi", "lead_gen_phi", "lead #gamma #phi"},
    //     {"sublead_phi", "sublead_gen_phi", "sublead #gamma #phi"},		

	// 	// bb part
    //     {"lead_bjet_pt", "lead_bjet_gen_pt", "lead b p_{T}(GeV)"},
    //     {"fabs(lead_bjet_pt-lead_bjet_gen_pt)/lead_bjet_gen_pt", "fabs(lead_bjet_pt-lead_bjet_gen_pt)/lead_bjet_gen_pt", "lead b |p_{T}^{RECO}-p_{T}^{GEN}|/p_{T}^{GEN}"},
    //     {"sublead_bjet_pt", "sublead_bjet_gen_pt", "sublead b p_{T}(GeV)"},
    //     {"fabs(sublead_bjet_pt-sublead_bjet_gen_pt)/sublead_bjet_gen_pt", "fabs(sublead_bjet_pt-sublead_bjet_gen_pt)/sublead_bjet_gen_pt", "sublead b |p_{T}^{RECO}-p_{T}^{GEN}|/p_{T}^{GEN}"},

    //     {"lead_bjet_eta", "lead_bjet_gen_eta", "lead b #eta"},
    //     {"sublead_bjet_eta", "sublead_bjet_gen_eta", "sublead b #eta"},
    //     {"lead_bjet_phi", "lead_bjet_gen_phi", "lead b #phi"},
    //     {"sublead_bjet_phi", "sublead_bjet_gen_phi", "sublead b #phi"},


	// 	// bbgg part
    //     // {"HHbbggCandidate_pt", "p_{T}^{HH} (GeV)"},
    //     // {"HHbbggCandidate_mass", "m_{HH} (GeV)"},
    //     // {"HHbbggCandidate_eta", "#eta_{HH}"},
    //     // {"HHbbggCandidate_phi", "#phi_{HH}"},
		
	// 	// VBF jets part
	// 	// {"VBF_first_jet_pt", "VBF lead jet p_{T}"},
	// 	// {"VBF_second_jet_pt", "VBF sublead jet p_{T}"},
	// 	// {"VBF_first_jet_pt/VBF_dijet_mass", "VBF lead jet p_{T}/m_{jj}"},
	// 	// {"VBF_second_jet_pt/VBF_dijet_mass", "VBF sublead jet p_{T}/m_{jj}"},
	// 	// {"VBF_dijet_pt", "VBF dijet p_{T}"},

	// 	// {"VBF_first_jet_eta", "VBF lead jet #eta"},
	// 	// {"VBF_second_jet_eta", "VBF sublead jet #eta"},
	// 	// {"VBF_dijet_eta", "VBF dijet #eta"},

	// 	// {"VBF_first_jet_phi", "VBF lead jet #phi"},
	// 	// {"VBF_second_jet_phi", "VBF sublead jet #phi"},
	// 	// {"VBF_dijet_phi", "VBF dijet #phi"},

	// 	// {"VBF_dijet_mass", "VBF dijet m_{jj}"},

	// 	// {"VBF_jet_eta_prod", "#eta_{j1}#times#eta_{j2}"},
	// 	// {"abs(VBF_jet_eta_diff)", "|#Delta#eta|"},
	// 	// {"VBF_DeltaR_jb_min", "min #Delta R(j, b)"},
	// 	// {"VBF_DeltaR_jg_min", "min #Delta R(j, #gamma)"},
	// 	// {"VBF_Cgg", "Centrality C_{#gamma#gamma}"},
	// 	// {"VBF_Cbb", "Centrality C_{bb}"},
	// 	// {"VBF_first_jet_btagDeepFlav_QG", "VBF lead jet QGL"},		
	// 	// {"VBF_second_jet_btagDeepFlav_QG", "VBF sublead jet QGL"},	
		
	// };
	// vector<vector<float>> range_ = {
	// 	// gg 

    //     {40, 35, 200},
    //     {40, 25, 120},
    //     {20, -2.6, 2.6},
    //     {20, -2.6, 2.6},
    //     {20, -3.2, 3.2},
    //     {20, -3.2, 3.2},

	// 	// bb part

    //     {50, 25, 250},
    //     {25, 0, 1},
    //     {50, 25, 120},
    //     {25, 0, 1},
    //     {20, -2.7, 2.7},
    //     {20, -2.7, 2.7},
    //     {20, -3.2, 3.2},
    //     {20, -3.2, 3.2},


	// 	// bbgg part
    //     // {40, 0, 300},
    //     // {40, 150, 700},
    //     // {40, -6, 6},
    //     // {20, -3.2, 3.2},

	// 	// VBF jets part
    //     // {40, 40, 350},
    //     // {40, 30, 140},
    //     // {40, 0, 2.5},
    //     // {40, 0, 1.5},
    //     // {40, 0, 400},

    //     // {25, -5, 5},
    //     // {25, -5, 5},
    //     // {25, -5, 5},

    //     // {10, -3.2, 3.2},
    //     // {10, -3.2, 3.2},
    //     // {10, -3.2, 3.2},

	// 	// {50, 0, 2000},

	// 	// {30, -15, 15},
    //     // {20, 0, 8},
    //     // {40, 0, 5},
    //     // {40, 0, 5},
	// 	// {40, 0, 1},
	// 	// {40, 0, 1},
	// 	// {30, 0, 1},
	// 	// {30, 0, 1},
	// };

	setTDRStyle();
	
	// for(int i=0; i<var_.size(); i++){
	// 	std::map<std::string, TH1F> histmap;
	// 	make_hist_gen(histmap, json_file, era, var_[i][0], var_[i][1], range_[i][0], range_[i][1], range_[i][2]);
	// 	drawMultiHist(histmap, histName, histColor, histLine, var_[i][0], var_[i][2], range_[i][1], range_[i][2], "gen_");
	// }
	




	vector<vector<string>> var_ = {
		{"lead_bjet_", "lead b #phi"},
        {"sublead_bjet_", "sublead b #phi"},
		// {"lead_bjet_", "lead b #eta"},
        // {"sublead_bjet_", "sublead b #eta"},
        // {"lead_bjet_", "lead b | p_{T}^{RECO}-p_{T}^{GEN} | / p_{T}^{GEN}"},
        // {"sublead_bjet_", "sublead b | p_{T}^{RECO}-p_{T}^{GEN} | / p_{T}^{GEN}"},
		// {"VBF_first_jet_", "VBF lead jet p_{T}"},
		// {"VBF_second_jet_", "VBF sublead jet p_{T}"},
	};
	vector<vector<float>> range_ = {
		{20, -3.2, 3.2},
		{20, -3.2, 3.2},
		// {20, -2.7, 2.7},
		// {20, -2.7, 2.7},
        // {30, 0, 0.9},
        // {30, 0, 0.9},
        // {50, 25, 250},
        // {50, 25, 120},
        // {50, 0, 1},
        // {50, 0, 1},
 
	};

	for(int i=0; i<var_.size(); i++){
		std::map<std::string, TH1F> histmap;
		make_hist_pt_(histmap, json_file, era, var_[i][0], range_[i][0], range_[i][1], range_[i][2]);
		drawMultiHist(histmap, histName, histColor, histLine, var_[i][0], var_[i][1], range_[i][1], range_[i][2], "match_");
	}


	// for(int i=0; i<var.size(); i++){
    // // for(int i=0; i<10; i++){
    //     std::map<std::string, TH1F> histmap;

    //     make_hist(histmap, json_file, era, var[i][0], range[i][0], range[i][1], range[i][2]);
    //     // drawDataMC2(histmap, var[i][0], var[i][1], range[i][1], range[i][2], "DataMC");
    //     drawMultiHist(histmap, histName, histColor, histLine, var[i][0], var[i][1], range[i][1], range[i][2], "calib");
    // }
    


}
// #include "sigmaEff.h"
#include "tdrstyle.C"
#include "CMS_lumi.C"
#include "nlohmann/json.hpp"
void SetHistColor(TH1F *hist, string color){
	// hist->SetFillColor(TColor::GetColor(color.c_str()));
	hist->SetLineColor(TColor::GetColor(color.c_str()));
	hist->SetLineWidth(5);
}

void drawQCD(std::map<std::string, TH1F> hist, TString var, string unit, float min, float max, string dir){

    setTDRStyle();
    TCanvas pl("pl","pl",700,700); 
	pl.cd();
	pl.SetLeftMargin(0.16);
	pl.SetBottomMargin(0.12);

	float maxY = (hist["QCD_PT"].GetBinContent(hist["QCD_PT"].GetMaximumBin()));
	if(maxY < hist["QCD_HT"].GetBinContent(hist["QCD_HT"].GetMaximumBin())) maxY = hist["QCD_HT"].GetBinContent(hist["QCD_HT"].GetMaximumBin());
	
	if(var.Contains("eta") || var.Contains("phi")) hist["QCD_PT"].SetMaximum(2*maxY);
    else hist["QCD_PT"].SetMaximum(1.4*maxY);
    
	if(var.Contains("Cbb") || var.Contains("Cgg")) gPad->SetLogy(1);	

	SetHistColor(&hist["QCD_PT"], "#37A3D2");
	SetHistColor(&hist["QCD_HT"], "#EC8C6F");


	hist["QCD_PT"].GetXaxis()->SetTitleOffset(1.);
	hist["QCD_PT"].GetXaxis()->SetTitleSize(0.05);
	hist["QCD_PT"].GetXaxis()->SetLabelSize(0.04);
	hist["QCD_PT"].GetXaxis()->SetTitle(unit.c_str());

	hist["QCD_PT"].GetYaxis()->SetTitleOffset(1.6);
	hist["QCD_PT"].GetYaxis()->SetTitleSize(0.05);
	hist["QCD_PT"].GetYaxis()->SetLabelSize(0.04);
	hist["QCD_PT"].GetYaxis()->SetTitle("Events");

	hist["QCD_PT"].Draw("hist");
    hist["QCD_HT"].Draw("histSAME");
	
    CMS_lumi(&pl);

	TLegend *le1 = new TLegend(0.63,0.7,0.92,0.87);
	le1->SetFillStyle(0);
	le1->SetBorderSize(0);
	le1->AddEntry(&hist["QCD_PT"],"QCD PT","l");
    le1->AddEntry(&hist["QCD_HT"],"QCD HT","l");

    le1->Draw();
	
	TString output;
	system(Form("mkdir -p %s", dir.c_str()));
	output=Form("./%s/%s.pdf", dir.c_str(), var.ReplaceAll('/', '_').Data());
	pl.Print(output);
	pl.Clear();
}

void make_hist(std::map<std::string, TH1F> &histmap, string json_file="./SampleList.json", string era="22postEE", string var="pt", int binning=50, float min=0, float max=400){
    ifstream json_sample(json_file.c_str());
    auto infile = nlohmann::json::parse(json_sample);
    
	string file;
    string tree = infile["tree"]["Data"];
    float lumi = infile["Data"]["lumi"][era];

    //* MC
    // check how many samples are included
    const int sample_size = infile["MC"].size();

	TH1F hQCDHT("hQCDHT","", binning, min, max);
	TH1F hQCDPT("hQCDPT","", binning, min, max);

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
        
		string weight = "weight";

        // fill overall hist

		if(sample.find("QCD_HT") != std::string::npos){
			ch.Draw(Form("%s>>+hQCDHT", var.c_str()), Form("%s*%f*%f", weight.c_str(), lumi, xs));
		}
		if(sample.find("QCD_Pt") != std::string::npos){
			ch.Draw(Form("%s>>+hQCDPT", var.c_str()), Form("%s*%f*%f", weight.c_str(), lumi, xs));
		}
        i++;
    }

    histmap["QCD_HT"] = hQCDHT;
    histmap["QCD_PT"] = hQCDPT;
}


void draw_QCD(){
    string json_file="./SampleList_QCD.json";
    string era = "22postEE";

    vector<vector<string>> var = {
		// gg part
		{"pt", "p_{T}^{#gamma#gamma} (GeV)"},
		{"CMS_hgg_mass", "m_{#gamma#gamma} (GeV)"},
        {"eta", "#eta_{#gamma#gamma}"},
        {"phi", "#phi_{#gamma#gamma}"},		
		
		{"lead_pt", "lead #gamma p_{T} (GeV)"},
		{"sublead_pt", "sublead #gamma p_{T} (GeV)"},
        {"lead_eta", "lead #gamma #eta"},
        {"sublead_eta", "sublead #gamma #eta"},
        {"lead_phi", "lead #gamma #phi"},
        {"sublead_phi", "sublead #gamma #phi"},		

		// bb part
        {"dijet_pt", "p_{T}^{bb} (GeV)"},
        {"dijet_mass", "m_{bb} (GeV)"},
        {"dijet_eta", "#eta_{bb}"},
        {"dijet_phi", "#phi_{bb}"},

        {"lead_bjet_pt", "lead b p_{T}(GeV)"},
        {"sublead_bjet_pt", "sublead b p_{T}(GeV)"},
        {"lead_bjet_eta", "lead b #eta"},
        {"sublead_bjet_eta", "sublead b #eta"},
        {"lead_bjet_phi", "lead b #phi"},
        {"sublead_bjet_phi", "sublead b #phi"},
		
		// // bbgg part
        {"HHbbggCandidate_pt", "p_{T}^{HH} (GeV)"},
        {"HHbbggCandidate_mass", "m_{HH} (GeV)"},
        {"HHbbggCandidate_eta", "#eta_{HH}"},
        {"HHbbggCandidate_phi", "#phi_{HH}"},
		
		// // VBF jets part
		{"VBF_first_jet_pt", "VBF lead jet p_{T}"},
		{"VBF_second_jet_pt", "VBF sublead jet p_{T}"},
		{"VBF_first_jet_pt/VBF_dijet_mass", "VBF lead jet p_{T}/m_{jj}"},
		{"VBF_second_jet_pt/VBF_dijet_mass", "VBF sublead jet p_{T}/m_{jj}"},
		{"VBF_dijet_pt", "VBF dijet p_{T}"},

		{"VBF_first_jet_eta", "VBF lead jet #eta"},
		{"VBF_second_jet_eta", "VBF sublead jet #eta"},
		{"VBF_dijet_eta", "VBF dijet #eta"},

		{"VBF_first_jet_phi", "VBF lead jet #phi"},
		{"VBF_second_jet_phi", "VBF sublead jet #phi"},
		{"VBF_dijet_phi", "VBF dijet #phi"},

		{"VBF_dijet_mass", "VBF dijet m_{jj}"},

		{"VBF_jet_eta_prod", "#eta_{j1}#times#eta_{j2}"},
		{"abs(VBF_jet_eta_diff)", "|#Delta#eta|"},
		{"VBF_DeltaR_jb_min", "min #Delta R(j, b)"},
		{"VBF_DeltaR_jg_min", "min #Delta R(j, #gamma)"},
		{"VBF_Cgg", "Centrality C_{#gamma#gamma}"},
		{"VBF_Cbb", "Centrality C_{bb}"},
		{"VBF_first_jet_btagDeepFlav_QG", "VBF lead jet QGL"},		
		{"VBF_second_jet_btagDeepFlav_QG", "VBF sublead jet QGL"},	
		
	};
	vector<vector<float>> range = {
		// gg part pt-mass-eta-phi
        {40, 0, 240},
        {20, 110, 150},
        {25, -5, 5},
        {10, -3.2, 3.2},

        {40, 35, 200},
        {40, 25, 120},
        {20, -2.6, 2.6},
        {20, -2.6, 2.6},
        {20, -3.2, 3.2},
        {20, -3.2, 3.2},

		
		// bb part
        {25, 0, 200},
        {13, 70, 200},
        {25, -5, 5},
        {20, -3.2, 3.2},

        {50, 25, 250},
        {50, 25, 120},
        {20, -2.7, 2.7},
        {20, -2.7, 2.7},
        {20, -3.2, 3.2},
        {20, -3.2, 3.2},

		// bbgg part
        {40, 0, 300},
        {40, 150, 700},
        {40, -6, 6},
        {20, -3.2, 3.2},

		// VBF jets part
        {40, 40, 350},
        {40, 30, 140},
        {40, 0, 2.5},
        {40, 0, 1.5},
        {40, 0, 400},

        {25, -5, 5},
        {25, -5, 5},
        {25, -5, 5},

        {10, -3.2, 3.2},
        {10, -3.2, 3.2},
        {10, -3.2, 3.2},

		{50, 0, 2000},

		{30, -15, 15},
        {20, 0, 8},
        {40, 0, 5},
        {40, 0, 5},
		{40, 0, 1},
		{40, 0, 1},
		{30, 0, 1},
		{30, 0, 1},
	};

    for(int i=0; i<var.size(); i++){
    // for(int i=0; i<3; i++){
        std::map<std::string, TH1F> histmap;

        make_hist(histmap, json_file, era, var[i][0], range[i][0], range[i][1], range[i][2]);
        drawQCD(histmap, var[i][0], var[i][1], range[i][1], range[i][2], "QCD_test");
    }
    


}
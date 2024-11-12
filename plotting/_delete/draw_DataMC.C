#include <iostream>
#include <vector>
#include <fstream>
#include <TString.h>
#include "TROOT.h"
#include <typeinfo>
#include <algorithm>
#include <TChain.h>
#include "TFile.h"
#include <TH1F.h>
#include "TLorentzVector.h"
#include "TLatex.h"
#include "TColor.h"
#include "TLegend.h"
#include "TPad.h"
#include "TLine.h"
#include "sigmaEff.h"
#include "tdrstyle.C"
#include "CMS_lumi.C"
using namespace std;
using VecI_t = const ROOT::RVec<int>&;
using VecF_t = const ROOT::RVec<float>&;

void drawDataMC(TString var="", TString unit="",int binning=0, double min=0, float max=0, TString dir=""){
	TString tree; 
	//* MC signal + background
	tree = "DiphotonTree/data_125_13TeV_NOTAG";
	TChain GluGluToHH(tree);
	TChain VBFToHH(tree);
	TChain GjetPt40(tree);
	TChain GjetPt20to40(tree);
	TChain GGjets(tree);
	TChain GluGluHtoGG_M125(tree);
	TChain VBFHtoGG_M125(tree);
	TChain VHtoGG_M125(tree);
	TChain ttHtoGG_M125(tree);

	// TChain GGjets(tree);
	// TChain GGjets(tree);
	// TChain GGjets(tree);
	// TChain ttbar(tree);
	
	//* data
	tree = "DiphotonTree/Data_13TeV_NOTAG";
	TChain data(tree);

	GluGluToHH.Add("../_minitree_no/22_postEE_GluGluToHH.root");
	VBFToHH.Add("../_minitree_no/22_postEE_VBFToHH.root");

	GjetPt40.Add("../_minitree_no/22_postEE_GjetPt40.root");
	GjetPt20to40.Add("../_minitree_no/22_postEE_GjetPt20to40.root");
	GGjets.Add("../_minitree_no/22_postEE_GGJets.root");

	GluGluHtoGG_M125.Add("../_minitree_no/22_postEE_GluGluHtoGG_M125.root");
	VBFHtoGG_M125.Add("../_minitree_no/22_postEE_VBFHtoGG_M125.root");
	VHtoGG_M125.Add("../_minitree_no/22_postEE_VHtoGG_M125.root");
	ttHtoGG_M125.Add("../_minitree_no/22_postEE_ttHtoGG_M125.root");
	
	// ttbar.Add(Form("../../_minitree_nos_%s/%s*_TTbar.root", dir.Data(), era.Data()));
	
	data.Add("../_minitree_no/data/EGamma_Run2022E.root");
	data.Add("../_minitree_no/data/EGamma_Run2022F.root");
	data.Add("../_minitree_no/data/EGamma_Run2022G.root");
	
	setTDRStyle();
	TCanvas *pl = new TCanvas("pl","pl",700,700); 
	pl->cd();
	
	TH1F *hggF = new TH1F("hggF","",binning,min,max);
	TH1F *hVBF = new TH1F("hVBF","",binning,min,max);
	TH1F *hZH  = new TH1F("hZH" ,"",binning,min,max);
	TH1F *hWmH = new TH1F("hWmH","",binning,min,max);
	TH1F *hWpH = new TH1F("hWpH","",binning,min,max);
	TH1F *httH = new TH1F("httH","",binning,min,max);
	TH1F *hZg  = new TH1F("hZg" ,"",binning,min,max);
	TH1F *hGjetPt40 = new TH1F("hGjetPt40","",binning,min,max);
	TH1F *hGjetPt20to40 = new TH1F("hGjetPt20to40","",binning,min,max);
	TH1F *hGGjets = new TH1F("hGGjets","",binning,min,max);
	TH1F *hHGG = new TH1F("hHGG","",binning,min,max);
	
	TH1F *hmc_sig = new TH1F("hmc_sig","",binning,min,max);
	TH1F *hmc = new TH1F("hmc","",binning,min,max);
	TH1F *hdata = new TH1F("hdata","",binning,min,max);

	// (VBFtag==0 && Leptag==0)* ((mllg>115&&mllg<120)||(mllg>130&&mllg<135)) (fabs(phoSCEta1)<1.479||(fabs(phoSCEta1)>1.479&&phoCalIDMVA1>-0.59))
	// TString SF="26.6717*1000*weight_nominal";
	TString SF="26.6717*1000*weight";
	if(var.Contains("VBF")) SF="26.6717*1000*weight*(VBF_first_jet_pt>0)";

	GluGluToHH.Draw(Form("%s>>hmc_sig",var.Data()),Form("%s*0.03443",SF.Data()));
	GluGluToHH.Draw(Form("%s>>hmc",var.Data()),Form("%s*0.03443",SF.Data()));
	VBFToHH.Draw(Form("%s>>+hmc_sig",var.Data()),Form("%s*0.00192",SF.Data()));
	VBFToHH.Draw(Form("%s>>+hmc",var.Data()),Form("%s*0.00192",SF.Data()));

	GjetPt20to40.Draw(Form("%s>>hGjetPt20to40",var.Data()),Form("%s*242.5",SF.Data()));
	GjetPt20to40.Draw(Form("%s>>+hmc",var.Data()),Form("%s*242.5",SF.Data()));
	GjetPt40.Draw(Form("%s>>hGjetPt40",var.Data()),Form("%s*919.1",SF.Data()));
	GjetPt40.Draw(Form("%s>>+hmc",var.Data()),Form("%s*919.1",SF.Data()));

	GGjets.Draw(Form("%s>>hGGjets",var.Data()),Form("%s*88.75/fabs(weight)/45977908.0",SF.Data()));
	GGjets.Draw(Form("%s>>+hmc",var.Data()),Form("%s*88.75/fabs(weight)/45977908.0",SF.Data()));

	GluGluHtoGG_M125.Draw(Form("%s>>hHGG",var.Data()),Form("%s*52.23*2.270E-03",SF.Data()));
	GluGluHtoGG_M125.Draw(Form("%s>>+hmc",var.Data()),Form("%s*52.23*2.270E-03",SF.Data()));
	VBFHtoGG_M125.Draw(Form("%s>>+hHGG",var.Data()),Form("%s*4.078*2.270E-03",SF.Data()));
	VBFHtoGG_M125.Draw(Form("%s>>+hmc",var.Data()),Form("%s*4.078*2.270E-03",SF.Data()));
	VHtoGG_M125.Draw(Form("%s>>+hHGG",var.Data()),Form("%s*2.401*2.270E-03",SF.Data()));
	VHtoGG_M125.Draw(Form("%s>>+hmc",var.Data()),Form("%s*2.401*2.270E-03",SF.Data()));
	ttHtoGG_M125.Draw(Form("%s>>+hHGG",var.Data()),Form("%s*0.570*2.270E-03",SF.Data()));
	ttHtoGG_M125.Draw(Form("%s>>+hmc",var.Data()),Form("%s*0.570*2.270E-03",SF.Data()));
		
	data.Draw(Form("%s>>hdata",var.Data()),""); 
	if(var.Contains("CMS_hgg_mass")) data.Draw(Form("%s>>hdata",var.Data()),Form("(%s<115||%s>135)",var.Data(),var.Data())); 
	if(var.Contains("HHbbggCandidate_mass")) data.Draw(Form("%s>>hdata",var.Data()),Form("(%s<350)",var.Data())); 
	if(var.Contains("dijet_mass")) data.Draw(Form("%s>>hdata",var.Data()),Form("(%s<105||%s>145)",var.Data(),var.Data())); 
	if(var.Contains("VBF")) data.Draw(Form("%s>>hdata",var.Data()),"(VBF_first_jet_pt>0)");  

	// if(var.Contains("mllg")) data.Draw(Form("%s>>hdata",var.Data()),Form("(%s<120||%s>130)",var.Data(),var.Data()));//(VBFtag==0 && Leptag==0)
	// else if(var.Contains("IDMVA")) data.Draw(Form("%s>>hdata",var.Data()),"");
	// else data.Draw(Form("%s>>hdata",var.Data()),""); 
	//cout<<httbar->Integral(-1,-1)/(hZjets->Integral(-1,-1)+httbar->Integral(-1,-1))<<endl;
	// cout<<hmc_sig->Integral(-1,-1)<<endl;
	cout<<"MC:"<<hmc->Integral(-1,-1)<<", Data:"<<hdata->Integral(-1,-1)<<", data/mc ratio:"<<hdata->Integral(-1,-1)/hmc->Integral(-1,-1)<<endl;

	/*--the main plot--*/
	TPad *pad1 = new TPad("pad1", "pad1", 0, 0.25, 1, 1);
	pad1->SetTopMargin(0.1);
	pad1->SetBottomMargin(0.015);
	pad1->SetRightMargin(0.05);
	pad1->SetLeftMargin(0.125);
	pad1->Draw("A");
	pad1->cd();
	
	THStack *hs = new THStack("hs","");
	
	// hVBF->SetLineColor(TColor::GetColor("#084887"));
	// hVBF->SetLineWidth(3);
	// hVBF->SetLineStyle(7);
	// hVBF->Scale(1000000);

	//hmc_sig->SetFillColor(TColor::GetColor("#F5CDA6"));	
	hmc_sig->SetLineColor(TColor::GetColor("#EF5B5B"));
	hmc_sig->SetLineWidth(3);
	hmc_sig->Scale(50);

	hGjetPt40->SetFillColor(TColor::GetColor("#F3C568"));
	hGjetPt40->SetLineColor(TColor::GetColor("#F3C568"));
	hGGjets->SetFillColor(TColor::GetColor("#37A3D2"));
	hGGjets->SetLineColor(TColor::GetColor("#37A3D2")); //E7D09D
	hGjetPt20to40->SetFillColor(TColor::GetColor("#EC8C6F"));
	hGjetPt20to40->SetLineColor(TColor::GetColor("#EC8C6F"));
	hHGG->SetFillColor(TColor::GetColor("#054A91"));
	hHGG->SetLineColor(TColor::GetColor("#054A91"));
	//hZjets->GetYaxis()->SetRangeUser(0,3E6);
	//if(var=="mll") pad1->SetLogy();

	hdata->SetMarkerSize(1);
	hdata->SetMarkerStyle(20);
	hdata->SetLineColor(1);
	hs->Add(hHGG);
	hs->Add(hGGjets);	
	hs->Add(hGjetPt40);
	hs->Add(hGjetPt20to40);
	
	// hs->SetMaximum(maxx*(hmc->GetBinContent(hmc->GetMaximumBin())));

	float maxY = (hmc->GetBinContent(hmc->GetMaximumBin()));
	if(maxY < hdata->GetBinContent(hdata->GetMaximumBin())) maxY = hdata->GetBinContent(hdata->GetMaximumBin());
	if(maxY < hmc_sig->GetBinContent(hmc_sig->GetMaximumBin())) maxY = hmc_sig->GetBinContent(hmc_sig->GetMaximumBin());
	hs->SetMaximum(1.4*maxY);
	if(var.Contains("eta")) hs->SetMaximum(2*maxY);
	if(var.Contains("phi")) hs->SetMaximum(2*maxY);

	// hs->SetMinimum(1000);
	if(var.Contains("Cbb") || var.Contains("Cgg")) gPad->SetLogy(1);	
	hs->Draw("hist");
	hs->GetXaxis()->SetLabelSize(0);
	hs->GetYaxis()->SetTitle("Events");
	hs->GetYaxis()->SetTitleOffset(1.2);
	hs->GetYaxis()->SetTitleSize(0.05);
	hs->GetYaxis()->SetLabelSize(0.04);
	hdata->Draw("PESAME");
	hmc_sig->Draw("histSAME");
	CMS_lumi(pad1);
	// TLatex *texSymbol = new TLatex ();
	// texSymbol -> SetNDC();
	// texSymbol -> SetTextFont(42);
	// texSymbol -> SetTextSize(0.04);
	// texSymbol -> DrawLatex(0.12, 0.915, "#bf{CMS} #it{work-in-progress}");

	pad1->Modified();

	TLegend *le1 = new TLegend(0.55,0.52,0.92,0.87);
	// le1->SetNColumns(2);
	le1->SetFillStyle(0);
	le1->SetBorderSize(0);
	le1->AddEntry(hdata,"Data","PE");	
	// le1->AddEntry(hmc_sig,"signal#times10^{5}","L");
	le1->AddEntry(hmc_sig,"signal#times50","L");

	// if(!era.Contains("UL")) le1->AddEntry(httbar,"TTbar","f");
	le1->AddEntry(hHGG,"H#rightarrow#gamma#gamma","f");
	le1->AddEntry(hGjetPt20to40,"#gamma+jets (40 > P_{T} > 20)","f");
	le1->AddEntry(hGjetPt40,"#gamma+jets (P_{T} > 40)","f");
	le1->AddEntry(hGGjets,"#gamma#gamma+jets","f");
	
	// if(Hlabel&&channel=="ele") le1->AddEntry((TObject*)0,"H#rightarrow e^{+}e^{-}#gamma"," ");
	// if(Hlabel&&channel=="mu") le1->AddEntry((TObject*)0,"H#rightarrow #mu^{+}#mu^{-}#gamma"," ");
	le1->Draw();
	pl->cd(); 

	/*--the ratio plot--*/
	//hmc->Sumw2();
	//hdata->Sumw2();
	TH1 *hrMC = (TH1F*)hmc->Clone();
	TH1 *hrdata = (TH1F*)hdata->Clone();
	hrdata->Divide(hrMC);

	TPad *pad1_1 = new TPad("pad1_1","pad1_1", 0, 0, 1, 0.25);

	pad1_1->SetTopMargin(0.03);
	pad1_1->SetBottomMargin(0.35);
	pad1_1->SetRightMargin(0.05);
	pad1_1->SetLeftMargin(0.125);

	pad1_1->Draw("A");
	pad1_1->cd();
	hrdata->SetMarkerSize(1);
	hrdata->SetMarkerStyle(20);
	hrdata->SetLineColor(1);
	hrdata->GetYaxis()->SetTitle("Data/MC");
	hrdata->GetYaxis()->SetNdivisions(505);
	hrdata->GetYaxis()->CenterTitle();
	hrdata->GetYaxis()->SetTitleOffset(0.4);
	hrdata->GetYaxis()->SetTitleSize(0.135);
	hrdata->GetYaxis()->SetLabelSize(0.12);
	
	hrdata->GetXaxis()->SetTitle(unit);
	//hrdata->GetXaxis()->SetNdivisions(505);
	hrdata->GetXaxis()->SetTitleOffset(0.9);
	hrdata->GetXaxis()->SetTitleSize(0.15);
	hrdata->GetXaxis()->SetLabelSize(0.12);
	hrdata->SetMinimum(-0.5+(hrdata->GetBinContent(hrdata->GetMinimumBin())));
	if(hrdata->GetMinimum()<0) hrdata->SetMinimum(0);

	hrdata->SetMaximum(0.3+(hrdata->GetBinContent(hrdata->GetMaximumBin())));
	// hrdata->SetMinimum(0.5);
	
	if(hrdata->GetMaximum()>4) hrdata->SetMaximum(4);
	
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
	// if(hrdata->GetMinimum()<0.5) line2->Draw("SAME");
	// if(hrdata->GetMaximum()>1.5) line3->Draw("SAME");
	if(hrdata->GetMaximum()>2) line4->Draw("SAME");
	if(hrdata->GetMaximum()>3) line5->Draw("SAME");
	TLegend *le1_1 = new TLegend(0.2,0.75,0.4,0.95);
	le1_1->AddEntry(hrdata,"Data/MC","PE");  
	le1_1->SetBorderSize(0);
	//le1_1->Draw();

	pl->cd();
	TString output;
	system(Form("mkdir -p %s", dir.Data()));
	output=Form("./%s/%s.pdf", dir.Data(), var.ReplaceAll('/', '_').Data());
	pl->Print(output);
	pl->Clear();
	delete hggF; delete hVBF; delete hZH; delete hWmH; delete hWpH; delete httH;
	delete hZg; delete hGjetPt40; delete hGjetPt20to40; delete hGGjets; delete hHGG; delete hdata; delete hrdata; delete hrMC; 
	delete hmc_sig; delete hmc;
	delete pl;
	
}

void draw_DataMC(){
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
        {40, 0, 400},
        {40, 110, 150},
        {25, -5, 5},
        {10, -3.2, 3.2},

        {40, 35, 200},
        {40, 25, 120},
        {20, -2.6, 2.6},
        {20, -2.6, 2.6},
        {30, -3.2, 3.2},
        {30, -3.2, 3.2},

		
		// bb part
        {40, 0, 300},
        {24, 70, 190},
        {25, -5, 5},
        {20, -3.2, 3.2},

        {50, 25, 250},
        {50, 25, 120},
        {20, -2.7, 2.7},
        {20, -2.7, 2.7},
        {30, -3.2, 3.2},
        {30, -3.2, 3.2},

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
    for(int j=0; j<var.size(); j++){
        // drawHist("tree", var[j][0], var[j][1], range[j][0], range[j][1], range[j][2]);
        drawDataMC(var[j][0], var[j][1], range[j][0], range[j][1], range[j][2], "DataMC_22postEE_latest_with");
    	// drawDataMC("UL18", "TH", var[j][0], var[j][1], range[j][0], range[j][1], range[j][2], "HBID");
    	// drawDataMC("UL18", "TH", var[j][0], var[j][1], range[j][0], range[j][1], range[j][2], "prasantID");
    	// drawDataMC("UL18", "TH", var[j][0], var[j][1], range[j][0], range[j][1], range[j][2], "OfficialID");

	}

}
	//MC signal
	// setTDRStyle();
	
	// TCanvas *pl = new TCanvas(); 
	// pl->cd();
	// TH1F *h1 = new TH1F("h1","",10,0,10);
	// h1->Fill(1,100);
	// h1->Fill(3,80);
	// h1->Fill(4,50);
	// h1->GetYaxis()->SetTitle("Events");
	// h1->GetXaxis()->SetTitle("Var(unit)");
	// h1->Draw("");
	// CMS_lumi(pl);
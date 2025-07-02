#include "TPad.h"
#include "TBox.h"
#include "TLine.h"
#include "TLatex.h"
#include "TASImage.h"

//
// Global variables
//

TString cmsText     = "CMS";
float cmsTextFont   = 61;  // default is helvetic-bold

bool writeExtraText = true;
TString extraText   = " work-in-progress";//"Private work"; //"Preliminary";
float extraTextFont = 52;  // default is helvetica-italics

// text sizes and text offsets with respect to the top frame
// in unit of the top margin size
float lumiTextSize   = 0.45;
float lumiTextOffset = 0.19;
float cmsTextSize    = 0.6;
float cmsTextOffset  = 0.1;  // only used in outOfFrame version

float relPosX    = 0.045;
float relPosY    = 0.035;
float relExtraDY = 1.2;

// ratio of "CMS" and extra text size
float extraOverCmsTextSize  = 0.76;

TString lumi_13TeV = "20.1 fb^{-1}";
TString lumi_8TeV  = "19.7 fb^{-1}";
TString lumi_7TeV  = "5.1 fb^{-1}";
//TString lumi_sqrtS = "2.6 fb^{-1}"; //2015 data
//TString lumi_sqrtS = "11.4-12.9 fb^{-1}"; //ICHEP data
//TString lumi_sqrtS = "36.811 fb^{-1}"; //Moriond 17 data (previous)
TString lumi_sqrtS[5] = {"35.92 fb^{-1}", "41.53 fb^{-1}", "59.73 fb^{-1}", "137 fb^{-1}", "34.65 fb^{-1}"}; //Moriond 17 data 
bool drawLogo = false;

void CMS_lumi(TPad *pad, int iPeriod = 0, int iPosX = 0, int iLumi_sqrtS = 4);
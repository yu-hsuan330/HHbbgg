import numpy as np
import pandas as pd
import math
from xgboost import plot_importance
# from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.inspection import permutation_importance
import seaborn as sns
import shap
# import pandas as pd
# import dask.dataframe as dd
# from scipy.interpolate import interp1d
# from sklearn import metrics, preprocessing
# from itertools import combinations
# from CateUtils import *
# from .dfUtils import *

def prGreen(prt): 
    print("\033[92m {}\033[00m" .format(prt))

def prRed(prt): 
    print("\033[91m {}\033[00m" .format(prt))

def pltSty(ax, xName = "x-axis", yName = "y-axis", TitleSize = 17, LabelSize = 16, TickSize = 13, MajTickLength = 7, MinTickLength = 4, yAuto = True):
    ax.set_xlabel(xName, fontsize = LabelSize, loc = "right")
    ax.set_ylabel(yName, fontsize = LabelSize, loc = "top")
    # ax.text(1, 1, "(13 TeV)", horizontalalignment = "right", verticalalignment = "bottom", transform = ax.transAxes, fontsize = TitleSize)
    ax.text(0, 1.01, "CMS", horizontalalignment = "left", verticalalignment = "bottom", transform = ax.transAxes, fontsize = TitleSize * 1.3, fontweight = "bold")
    ax.text(TitleSize * 0.01, 1.015, "work-in-progress", horizontalalignment = "left", verticalalignment = "bottom", transform = ax.transAxes, fontsize = TitleSize, style='italic')

    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if yAuto :
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction = "in", length = MajTickLength, labelsize = TickSize, top = True, right = True)
    ax.tick_params(direction = "in", length = MinTickLength, which = "minor", labelsize = TickSize, top = True, right = True)

def plot_Features(MVA, df, weight, Conf):
    # Input the feature units and the weight
    features_unit, feature_bins = MVA["features_unit"], MVA["feature_bins"]
    nFeature = len(features_unit)
    
    row = math.ceil(nFeature/4)
    fig, axes = plt.subplots(row, 4, figsize=(7*4, 6*row))
    for idx, feature in enumerate(MVA["features"]):
        ax = axes[(idx)//4, idx-(idx//4)*4]

        # Read the features by class
        for j, Class in enumerate(Conf.Classes):
            weight_Class = np.asarray(df[weight][df['Category'] == j])
            feature_Class = np.asarray(df[feature][df['Category'] == j])
            ax.hist(feature_Class, label=Class, weights=weight_Class, bins=feature_bins[idx], density=True,
                    histtype='step', alpha=0.7, linewidth=4, color=Conf.ClassColors[j])

        pltSty(ax, xName=features_unit[idx], yName="Events", yAuto=False)
        ax.legend(title=MVA["Label"], loc="best", title_fontsize=12, fontsize=12, frameon=False)

    fig.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"Feature_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
    fig.savefig(Conf.OutputDirName+"/Plots/"+"Feature_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')

def plot_Features_Scaled(MVA, x_train, y_train, w_train, Conf):

    x_train = pd.DataFrame(x_train, columns=MVA["features"])
    # input the feature units and the weight
    features_unit, feature_bins = MVA["features_unit"], MVA["feature_bins"]
    nFeature = len(features_unit) # due to photon ID MVA->EE/EB
    
    row = math.ceil(nFeature/4)
    fig, axes = plt.subplots(row, 4, figsize=(7*4, 6*row))
 
    for idx, feature in enumerate(MVA["features"]):
        ax = axes[(idx)//4, idx-(idx//4)*4]

        # input the feature in the signal/background
        for j, Class in enumerate(Conf.Classes):
            # weight_Class = np.asarray(w_train)
            feature_Class = np.asarray(x_train[feature][np.argmax(y_train, axis=1) == j])
            ax.hist(feature_Class, label=Class, bins=30, density=True, #, weights=weight_Class
                    histtype='step', alpha=0.7, linewidth=4, color=Conf.ClassColors[j])

        pltSty(ax, xName=features_unit[idx], yName="Events", yAuto=False)
        ax.legend(title=MVA["Label"], loc="best", title_fontsize=12, fontsize=12, frameon=False)

    fig.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"Feature_"+MVA["MVAtype"]+"_Norm.pdf", bbox_inches='tight')
    fig.savefig(Conf.OutputDirName+"/Plots/"+"Feature_"+MVA["MVAtype"]+"_Norm.pdf", bbox_inches='tight')

def plot_VarImportance(MVA, model, Conf, X_train, X_test, Y_test):
    # Plot the feature importance
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    
    if MVA["Algorithm"] == "XGB":
        bst = model
        bst.feature_names = MVA["features_unit"]
        plot_importance(bst, ax = ax, importance_type = "gain", height = 0.7, grid = False, title = None, xlabel = None, ylabel = None, show_values=0, color = "#A2D5AB")
    
    elif MVA["Algorithm"] == "DNN":
        # Use SHAP to explain the model's predictions
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_test[:1000]) 
        for i, class_names in enumerate(Conf.Classes):
            fig, ax = plt.subplots(1, 1, figsize=(6,6))
            shap.summary_plot(shap_values[..., i], X_test[:1000], feature_names=MVA["features_unit"], show=False, color=Conf.ClassColors[i], plot_type="bar", max_display=30) #, class_names= ['a']
        
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['top'].set_visible(True)
            plt.gca().spines['left'].set_linewidth(0.5)
            plt.gca().spines['bottom'].set_linewidth(0.5)
            
            pltSty(ax, xName = "Importance", yName = "Features", yAuto = False)
            plt.draw()
            fig.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"VI_"+MVA["MVAtype"]+"_{}_all.pdf".format(i), bbox_inches='tight')
            fig.savefig(Conf.OutputDirName+"/Plots/"+"VI_"+MVA["MVAtype"]+"_{}_all.pdf".format(i), bbox_inches='tight')
            plt.close("all")
        for i, class_names in enumerate(Conf.Classes):
            fig, ax = plt.subplots(1, 1, figsize=(6,6))
            shap.summary_plot(shap_values[..., i], X_test[:2000], feature_names=MVA["features_unit"], show=False, color=Conf.ClassColors[i], plot_type="bar", max_display=10) #, class_names= ['a']
        
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['top'].set_visible(True)
            plt.gca().spines['left'].set_linewidth(0.5)
            plt.gca().spines['bottom'].set_linewidth(0.5)
            
            pltSty(ax, xName = "Importance", yName = "Features", yAuto = False)
            plt.draw()
            fig.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"VI_"+MVA["MVAtype"]+"_{}.pdf".format(i), bbox_inches='tight')
            fig.savefig(Conf.OutputDirName+"/Plots/"+"VI_"+MVA["MVAtype"]+"_{}.pdf".format(i), bbox_inches='tight')
            plt.close("all")
    else:
        prRed("Model not supported for feature importance plot!")
    
    
def plot_ROC(MVA, y_train_categorical, y_test_categorical, y_train_pred, y_test_pred, w_train, w_test, Conf):
    # y_train_categorical = ToCategorical(y_train, num_classes=len(Conf.Classes))
    # y_test_categorical = ToCategorical(y_test, num_classes=len(Conf.Classes))

    fpr, tpr, th, roc_auc = {}, {}, {}, {}
    fpr_tr, tpr_tr, th_tr, roc_auc_tr = {}, {}, {}, {}

    for i in range(len(Conf.Classes)):
        fpr[i], tpr[i], th = roc_curve(y_test_categorical[:, i], y_test_pred[:, i], sample_weight=w_test)
        fpr_tr[i], tpr_tr[i], th_tr = roc_curve(y_train_categorical[:, i], y_train_pred[:, i], sample_weight=w_train)
        roc_auc[i] = auc(fpr[i], tpr[i])
        roc_auc_tr[i] = auc(fpr_tr[i], tpr_tr[i])

    # fpr, tpr, th = roc_curve(y_test_categorical[:,0], y_test_pred[:,0], sample_weight=w_test)
    # fpr_tr, tpr_tr, th_tr = roc_curve(y_train_categorical[:,0], y_train_pred[:,0], sample_weight=w_train)
    # roc_auc = auc(fpr, tpr)
    # roc_auc_tr = auc(fpr_tr, tpr_tr)

    bkgrej = {key: 1 - value for key, value in fpr.items()}
    bkgrej_tr = {key: 1 - value for key, value in fpr_tr.items()}

    # bkgrej = (1 - fpr)
    # bkgrej_tr = (1 - fpr_tr)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    pltSty(ax, xName = "Signal efficiency", yName = "Background rejection")
    for i, color in zip(range(len(Conf.Classes)), Conf.ClassColors):
        # ax.plot(tpr_tr[i], bkgrej_tr[i], label='Training  AUC=%2.1f%%' % (roc_auc_tr[i]*100), linewidth=4, color=color, linestyle="dashed")
        ax.plot(tpr[i], bkgrej[i], label=Conf.Classes[i]+'(AUC=%2.1f%%)' % (roc_auc[i]*100), linewidth=4, color=color)
        ax.legend(title=MVA["Label"], loc="lower left", title_fontsize=15, fontsize=15, frameon=False)
    fig.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"ROC_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
    fig.savefig(Conf.OutputDirName+"/Plots/"+"ROC_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')

def plot_MVA(MVA, y_train_categorical, y_test_categorical, y_train_pred, y_test_pred, w_train, w_test, Conf, transform=False):
    
    n_classes = len(Conf.Classes)

    # if MVA["Algorithm"] == "XGB":
    #     y_train_categorical = to_categorical(y_train, n_classes)
    #     y_test_categorical = to_categorical(y_test, n_classes)

    # elif MVA["Algorithm"] == "DNN":
    #     y_train_categorical = y_train
    #     y_test_categorical = y_test

    figMVA, axMVA = plt.subplots(1, 1, figsize=(6, 6))
    xmin = 0
    if transform:
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), clip=True)
        min_max_scaler.fit(y_train_pred)
        y_train_pred = min_max_scaler.transform(y_train_pred)
        y_test_pred = min_max_scaler.transform(y_test_pred)
        xmin = -1
    for k in range(n_classes):
        axMVA.hist(y_test_pred[:, k][y_test_categorical[:, k]==1], bins=np.linspace(xmin, 1, 41), label=Conf.Classes[k]+'_test',
                    weights=w_test[y_test_categorical[:, k]==1]/np.sum(w_test[y_test_categorical[:, k]==1]),
                    histtype='step',linewidth=2,color=Conf.ClassColors[k])
        axMVA.hist(y_train_pred[:, k][y_train_categorical[:, k]==1],bins=np.linspace(xmin, 1, 41),label=Conf.Classes[k]+'_train',
                    weights=w_train[y_train_categorical[:, k]==1]/np.sum(w_train[y_train_categorical[:, k]==1]),
                    histtype='stepfilled',alpha=0.3,linewidth=2,color=Conf.ClassColors[k])
    pltSty(axMVA, xName="Score", yName="Events")
    # axMVA.set_ylim([0,0.09])
    # if Conf.channel == "ele":
    #     axMVA.legend(title=MVA["Label"], loc="upper center", title_fontsize=12, fontsize=12, frameon=False)
    # else:    
    axMVA.legend(title=MVA["Label"], loc="best", title_fontsize=12, fontsize=12, frameon=False)
    if Conf.MVAlogplot:
        axMVA.set_yscale('log')

    if transform :
        figMVA.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"MVAt_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
        figMVA.savefig(Conf.OutputDirName+"/Plots/"+"MVAt_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
    else:
        figMVA.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"MVA_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
        figMVA.savefig(Conf.OutputDirName+"/Plots/"+"MVA_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')

def plot_confusion_matrix(MVA, y_test, y_test_pred, Conf, Norm=False):
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_test_pred, axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    cm_type = 'd'
    if Norm == True:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_type = '.1%'
    
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
    ax = sns.heatmap(cm, annot=True, fmt=cm_type, cmap='Oranges', cbar=False, xticklabels=Conf.Classes, yticklabels=Conf.Classes, annot_kws={"fontsize":14})
    # Oranges Blues
    pltSty(ax, xName = "Predicted Class", yName = "Actual Class", yAuto = False, MajTickLength = 0, MinTickLength = 0)

    if(len(Conf.Classes) > 3):
        ax.set_xticks(np.arange(len(Conf.Classes))+0.5, labels=Conf.Classes, rotation=45, ha="right", va="center", rotation_mode="anchor")
        ax.set_yticks(np.arange(len(Conf.Classes))+0.5, labels=Conf.Classes, rotation=0, ha="right", va="center", rotation_mode="anchor")

    fig.tight_layout()
   
    plt.draw()

    if Norm == True:
        fig.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"ConfMat_"+MVA["MVAtype"]+"_Normalized.pdf", bbox_inches='tight')
        fig.savefig(Conf.OutputDirName+"/Plots/"+"ConfMat_"+MVA["MVAtype"]+"_Normalized.pdf", bbox_inches='tight')
    else: 
        fig.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"ConfMat_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
        fig.savefig(Conf.OutputDirName+"/Plots/"+"ConfMat_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
    
    plt.close("all")
    

def plot_correlation_matrix(MVA, X_train, Conf):
    # Compute the correlation matrix
    X_train = X_train[X_train["vbfjet_lead_ptOverM"] < 0]
    X_train = X_train[MVA["features"][0:-18]]
    print(X_train.shape)
    units = MVA["features_unit"][0:-18]
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns=units)
    
    corr = X_train.corr()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(13, 11))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap='coolwarm', ax=ax, xticklabels=units, yticklabels=units)
    pltSty(ax, xName = "Input features", yName = "Input features", yAuto = False, MajTickLength = 0, MinTickLength = 0)
    
    ax.set_xticks(np.arange(len(units))+0.5, labels=units, rotation=90, ha="right", va="center", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(units))+0.5, labels=units, rotation=0, ha="right", va="center", rotation_mode="anchor")
    fig.tight_layout()

    # Set titles and labels
    # ax.set_title(title)
    fig.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"CorrMat_"+MVA["MVAtype"]+"1.pdf", bbox_inches='tight')
    fig.savefig(Conf.OutputDirName+"/Plots/"+"CorrMat_"+MVA["MVAtype"]+"1.pdf", bbox_inches='tight')

def loss_history(MVA, training_loss, testing_loss, Conf):
    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    # Visualize loss history
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(epoch_count, training_loss)
    ax.plot(epoch_count, testing_loss)
    ax.legend(['Training Loss', 'Testing Loss'], fontsize=12, frameon=False)
    pltSty(ax, xName = "Epoch", yName = MVA["Label"]+': Loss')
    fig.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/Loss_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
    fig.savefig(Conf.OutputDirName+"/Plots/"+"Loss_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')

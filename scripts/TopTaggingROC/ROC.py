import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os, sys, glob
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.interpolate import interp1d
import uproot
import torch
# Function that takes the labels and score of the positive class
# (top class) and returns a ROC curve, as well as the signal efficiency
# and background rejection at a given targe signal efficiency, defaults
# to 0.3
def buildROC(labels, score, targetEff=[0.3,0.5]):
    if not isinstance(targetEff, list):
        targetEff = [targetEff]
    fpr, tpr, threshold = roc_curve(labels, score)
    idx = [np.argmin(np.abs(tpr - Eff)) for Eff in targetEff]
    eB, eS = fpr[idx], tpr[idx]
    return fpr, tpr, threshold, eB, eS

# Averaging (x,y) points for repeated x values
# See https://stackoverflow.com/questions/7790611/average-duplicate-values-from-two-paired-lists-in-python-using-numpy
def f_numpy(x_vals, y_vals):
    result_x = np.unique(x_vals)
    result_y = np.empty(result_x.shape)
    for i, x in enumerate(result_x):
        result_y[i] = np.mean(y_vals[x_vals == x])
    return result_x, result_y

def main(args):
    fig = plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    
    if(len(sys.argv) > 1): mode = str(sys.argv[1])
    else: mode = 'light'
    
    # Data directory for reference networks
    datadir_ref = "./scores/"
    
    # Output directory
    outdir = "./figures/"
    os.makedirs(outdir, exist_ok=True)

    # Output filename
    outfilename = "TopTaggingROC"
    outfilename += '_' + mode

    # LorentzNet score files
    my_ROC_filenames = glob.glob('./scores/LorentzNet/*.npy', recursive=True)
            
    # Prep the axis colors (silly matplotlib way of doing things...)
    theme_color = ['xkcd:black', '0.65','xkcd:white'] # text, grid line (grey scale)
    if(mode == 'dark'): theme_color = ['xkcd:white', '0.65','xkcd:black']
    mpl.rc('axes',edgecolor=theme_color[0],labelcolor = theme_color[0], facecolor = theme_color[0])
    mpl.rc('xtick',color=theme_color[0])
    mpl.rc('ytick',color=theme_color[0])

    # Test dataset and associated labels
    testlabelsfilename = "testlabels.npy"
    labels = np.load(datadir_ref + testlabelsfilename)

    # List of results to load. These come from the Top Tagging Reference Dataset
    # which is described here: https://docs.google.com/document/d/1Hcuc6LBxZNX16zjEGeq16DAzspkDC4nDTyjMp1bWHRo/edit
    # and hosted here: https://desycloud.desy.de/index.php/s/4DDFkfYRGaPo2WJ
    results =	{
    # #  "NSub(6)": "6_body_test_sets.npy",
    # #  "DGCNN": "DGCNN_v1.npy",
    # "EFN": "EFN_v0.npy",
    # "EFP": "EFP_v0.npy",
    # #  "LDA": "LDA_testScores.npy",
    "LGN": "LGN_run3_predictions.pt", #"LGN_top.best.test.pt",
    "P-CNN": "P-CNN_v2.npy",
    "PFN": "PFN_v0.npy",
    "ParticleNet": "particlenet.root",
    # "ParticleNet": "ParticleNet_v2.npy",
    "ResNeXt": "ResNeXt50_v2.npy",
    "EGNN": "EGNN.npy",
    # #  "R-CNN": "RutgersCNN_9runs.npy",
    # #  "TreeNiN": "TreeNiN_hd50.npy",
    # #  "LBN": "lbn_ensemble9.npz",
    # #  "NSub(8)": "nsub_8body_v2.npy",
    # "TopoDNN": "topodnn_v2.npy",
    }

    # Loop over the results and extract the score for the signal class
    # then build a ROC curve

    lines=iter(['dotted', 'dashed', 'dashdot', (0, (1, 1)),  (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5))])
    for network in results:
        print("Building ROC curve for network: %s" % network)
        filename = results[network]  # Get file name
        if filename.endswith(".root"): ## weaver output
            df = uproot.open(datadir_ref + filename)['Events'].arrays(library='pd')
            bkg_name = 'jet_isQCD'
            sig_name = 'jet_isTop'
            y_true = df[sig_name].values
            y_pred = df[f'score_{sig_name}'].values
            auc = roc_auc_score(y_true,y_pred)
            acc = ((y_pred > 0.5) == y_true).sum()/y_true.shape[0]
            fpr, tpr, threshold, eB, eS = buildROC(y_true, y_pred)
        elif filename.endswith(".pt"): ## LGN output
            pred = torch.load(datadir_ref + filename, map_location = 'cpu')
            y_true = pred['targets'].numpy()
            score = pred['predict'].numpy()
            auc = roc_auc_score(y_true,score[:,1])
            y_pred = np.argmax(score,axis = 1)
            acc = (y_pred == y_true).sum()/y_true.shape[0]
            fpr, tpr, threshold, eB, eS = buildROC(y_true, score[:,1])
        else:
            score    = np.load(datadir_ref + filename)   # Load file and get scores
            if score.shape[-1] == 3 : # EGNN.py
                my_score = score[...,2]
                my_label = score[...,0]
                acc = ((my_score > 0.5) == my_label).sum()/my_label.shape[0]                   
                fpr, tpr, threshold, eB, eS  = buildROC(my_label,my_score)
                auc = roc_auc_score(my_label,my_score)
            else:
                auc = roc_auc_score(labels, score)
                acc = ((score > 0.5).reshape(-1,1) == labels).sum()/labels.shape[0]
                fpr, tpr, threshold, eB, eS = buildROC(labels, score)
        rejection = 1./eB
        print(f"auc:{auc:.4f} 0.3 1/eB: {1./eB[0]:.2f}, 0.5 1/eB: {1./eB[1]:.2f} Acc: {acc:.4f}")
        #plt.plot(tpr, 1/fpr, label = r'%s ($1/\epsilon_{B}$ = %0.2f)' % (network, rejection) )
        plt.plot(tpr, 1/fpr, label = '%s' % (network), linestyle = next(lines), lw = 2.5)

    # Our results.
    # We will average the curves and include an error band.
    # This is complicated by the fact that the x-values for the curves' points
    # are not always identical (nor do they seem to have the same number of them).
    # So we must do some interpolation.

    x_all = np.linspace(0.,1., num = 50000, endpoint = True) # common carrier. density is somewhat arbitrary density, determined empirically
    fits = [] # for the interpolated curves, applied to the common carrier
    rej = []
    accuracy = []
    for ROC_file in my_ROC_filenames:
        path = Path(ROC_file)
        my_res = np.load(ROC_file)
        my_score = my_res[...,2]
        my_label = my_res[...,0]
        acc = ((my_score > 0.5) == my_label).sum()/my_label.shape[0]
        my_fpr, my_tpr, my_thresh, eB, eS  = buildROC(my_label,my_score)
        auc = roc_auc_score(my_label,my_score)
        rej.append(1./eB)
        accuracy.append(acc)
        print(f"{ROC_file} auc:{auc:.4f}; 0.3 1/eB: {1./eB[0]:.2f}; 0.5 1/eB: {1./eB[1]:.2f}; Acc: {acc:.5f}")
        x_vals = my_tpr
        y_vals = my_fpr
        
        # Get rid of infinities in y-values of the ROC curve.
        # This can seemingly be avoided by handling 1/y
        # and then just inverting it inside the matplotlib plot command,
        # but we want to average y-values so we can't do that.
        
        n_zero = 0
        for i in range(len(y_vals)):
            if(y_vals[i] == 0.):
                n_zero += 1
            else: break
            
        # quick hack - instead of reaching inf,
        # y-values will plateau at the highest
        # finite value (this will be off the plot)
        y_vals[:n_zero] = y_vals[n_zero]
        y_vals = 1./y_vals
        
        # Sort the x-values, apply sorting to y-values.
        indices = np.argsort(x_vals)
        x_vals = x_vals[indices]
        y_vals = y_vals[indices]

        # Interpolation has issues if we have repeated x-values,
        # which in practice we sometimes have.
        # As a fix, we will average y-values for repeated x-values.
        # E.g. (x,y1),(x,y2) -> (x,avg(y1,y2))
        x_vals, y_vals = f_numpy(x_vals, y_vals)
        f = interp1d(x_vals, y_vals, 'linear')
        fits.append(f(x_all))
    
    rej = np.array(rej)
    accuracy = np.array(accuracy)
    print(f"0.3 1/eB mean: {rej[:,0].mean():.2f}, std: {rej[:,0].std():.2f} ")
    print(f"0.5 1/eB mean: {rej[:,1].mean():.2f}, std: {rej[:,1].std():.2f} ")
    print(f"ACC mean:{accuracy.mean():.5f}, std: {accuracy.std():.5f}")
    data_collection = np.vstack(tuple(fits)) # collect all the interpolated curves in one matrix
    f_avg = np.average(data_collection, axis=0) # average curve at each point
    f_stdev = np.std(data_collection, axis = 0) # stdev of curves at each point
    x = plt.plot(x_all, f_avg, label = '%s' % ('LorentzNet'), linewidth=2.5, color='#4bade9')
    plt.fill_between(x_all, f_avg -  f_stdev, f_avg + f_stdev, linewidth=0., color = x[0].get_color(), alpha = 0.4)

    #plt.title('Receiver Operating Characteristic')
        
    legend = plt.legend(loc = 'upper right',prop={'size':15})
    plt.setp(legend.get_texts(), color = theme_color[0])
    legend.get_frame().set_facecolor(theme_color[2])
    plt.plot([0, 1], [0, 1],'r--')
    
    plt.xlim([0, 1])
    plt.ylim([4, 40000])
    plt.yscale('log')
    
    plt.grid(True)
    plt.grid(b=True, which='both', color=theme_color[1], linestyle='--')
    
    plt.xticks(np.arange(0, 1, step=0.1))
#    plt.yticks(color = theme_color[0])
    
    plt.ylabel(r'Background rejection $\frac{1}{\epsilon_{B}}$', color = theme_color[0])
    plt.xlabel(r'Signal efficiency $\epsilon_{S}$', color = theme_color[0])
    
    plt.savefig(outdir + outfilename + ".jpg", transparent=True, dpi=400, bbox_inches = 'tight')
    plt.savefig(outdir + outfilename + ".pdf", transparent=True, dpi=400, format='pdf', bbox_inches = 'tight')
    plt.show()

if __name__ == '__main__':
    main(sys.argv)

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, plot_roc_curve, roc_auc_score, roc_curve, auc, accuracy_score, precision_recall_fscore_support, jaccard_score
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import clone

import pickle
import time
import os
from sys import platform
from IPython.display import clear_output
from itertools import combinations 
import networkx as nx

# PICKLE FILES
from six.moves import cPickle as pickle 
def pickle_file(file_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(file_, f)
    f.close()
def load_pickle(filename_):
    with open(filename_, 'rb') as f:
        ret_file = pickle.load(f)
    return ret_file

num_subs     = 323
num_parcels  = 90
vector_size  = 4005 #np.int((num_parcels * num_parcels)/2 - (num_parcels/2))

##########################################################################
# matrix manipulation
##########################################################################

def digitize_rdm(rdm_raw, n_bins=10): 
    """
        Digitize an input matrix to n bins (10 bins by default)
        rdm_raw: a square matrix 
    """
    rdm_bins = [np.percentile(np.ravel(rdm_raw), 100/n_bins * i) for i in range(n_bins)] # compute the bins 
    rdm_vec_digitized = np.digitize(np.ravel(rdm_raw), bins = rdm_bins) * (100 // n_bins) # Compute the vectorized digitized value 
    rdm_digitized = np.reshape(rdm_vec_digitized, np.shape(rdm_raw)) # Reshape to matrix
    rdm_digitized = (rdm_digitized + rdm_digitized.T) / 2     # Force symmetry in the plot
    return rdm_digitized

# change shape 
def symm_mat_to_ut_vec(mat):
    """
        go from symmetrical matrix to vectorized/flattened upper triangle
    """
    vec_ut = mat[np.triu_indices(len(mat), k=1)]
    return vec_ut
def ut_mat_to_symm_mat(mat):
    '''
        go from upper tri matrix to symmetrical matrix
    '''
    for i in range(0, np.shape(mat)[0]):
        for j in range(i, np.shape(mat)[1]):
            mat[j][i] = mat[i][j]
    return mat
def ut_vec_to_symm_mat(vec):
    '''
        go from vectorized/flattened upper tri (to upper tri matrix) to symmetrical matrix
    '''
    ut_mat = ut_vec_to_ut_mat(vec)
    symm_mat = ut_mat_to_symm_mat(ut_mat)
    return symm_mat
def ut_vec_to_ut_mat(vec):
    '''
        go from vectorized/flattened upper tri to a upper tri matrix
            1. solve get matrix size: matrix_len**2 - matrix_len - 2*vector_len = 0
            2. then populate upper tri of a m x m matrix with the vector elements 
    '''
    
    # solve quadratic equation to find size of matrix
    from math import sqrt
    a = 1; b = -1; c = -(2*len(vec))   
    d = (b**2) - (4*a*c) # discriminant
    roots = (-b-sqrt(d))/(2*a), (-b+sqrt(d))/(2*a) # find roots   
    if False in np.isreal(roots): # make sure roots are not complex
        raise Exception('Roots are complex') # dont know if this can even happen if not using cmath...
    else: 
        m = int([root for root in roots if root > 0][0]) # get positive root as matrix size
        
    # fill in the matrix 
    mat = np.zeros((m,m))
    vec = vec.tolist() # so can use vec.pop()
    c = 0  # excluding the diagonal...
    while c < m-1:
        r = c + 1
        while r < m: 
            mat[c,r] = vec[0]
            vec.pop(0)
            r += 1
        c += 1
    return mat

##########################################################################
# different kinds of sublevel matrices: correlations, distances 
##########################################################################

from sklearn.metrics import pairwise_distances
def crossrun_fc(run1_tseries, run2_tseries):
    ''' (date: 01-2021)
        inputs:
        
        outputs:
        refs, links etc...
    '''
    fc_mat = np.zeros((len(run1_tseries), len(run1_tseries)))
    
    # vectorize to speed up....
    for r1, r1_tseries in enumerate(run1_tseries):
        if np.any(r1_tseries==0) or np.any(r1_tseries==np.nan):
            fc_mat[r1,r2] = np.nan
            continue
        else:
            for r2, r2_tseries in enumerate(run2_tseries):
                if np.any(r2_tseries==0) or np.any(r2_tseries==np.nan):
                    fc_mat[r1,r2] = np.nan
                    continue
                else:
                    fc_mat[r1,r2], _ = scipy.stats.pearsonr(r1_tseries, r2_tseries)
    return fc_mat
def make_mean_distance_matrix_tseries(tseries):
    mean_tseries   = np.mean(tseries, axis=1)
    mean_distances = pairwise_distances(mean_tseries, metric='euclidean')
    return mean_distances
def make_mean_distance_ut_vec(means):
    mean_distances = pairwise_distances(means.reshape(-1,1), metric='euclidean')
    return symm_mat_to_ut_vec(mean_distances)

##########################################################################
# decoder training, testing
##########################################################################

max_iter = 100000
random_state = 1

class ClassifierGridSearch(object):

    def __init__(self, classifier_dict, params_dict):
        """
        Accepts a dictionary of classifiers and parameter grids and performs a cross validated grid search
        """
        self.classifier_dict = classifier_dict
        self.params_dict = params_dict
    
    def grid_search(self, X, y, **grid_kwargs):
        """
            search hyperparameter space
                kwargs: sklearn args - scoring metric, cv, n_folds, etc
                ?specify what kind of scoring exists?
        """
        self.X = X
        self.y = y
        self.grid_searches = {} # to output

        for c, (self.name, classifier) in enumerate(self.classifier_dict.items()): 
            start_time = time.time()
            print('Running GridSearchCV for %s.' % self.name)
            search_params = self.params_dict[self.name] # get the parameters grid
            grid_search = GridSearchCV(classifier, search_params, **grid_kwargs) # create object
            grid_search.fit(self.X, self.y) # fit
            self.grid_searches[self.name] = grid_search # create a grid search dict
            print("Time to run %s: %s seconds " % (self.name, (time.time() - start_time)), '\n')
        print('Done.')

    def grid_search_summary(self, sort_by='mean_test_score'):
        """
            Summarize grid search results
        """
        # get results
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)

        # turn results into dataframe    
        df = pd.concat(frames)
        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)
        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator']+columns
        df = df[columns]

        return df

clf_grid    = {'SGDClassifier': SGDClassifier(random_state=random_state, max_iter=max_iter)}
param_grid = {'SGDClassifier': {
        'loss': ['log', 'hinge'],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'l1_ratio': np.linspace(0.1,1,10),
        'alpha': [0.0001, 0.1, 1, 5, 25, 100]}}

class ClassifierPipeline(object): 

    def __init__(self, classifier_dict):
        """
            Preprocessing should be done 
            Args: 
                dictionary of classifier, w its parameters
                number of cross-validation folds
        """
        self.classifier_dict = classifier_dict
    
    def fit(self, X, y, numfolds):
        """
            Fit k-fold cross-validated classifier, store model metrics & create ROC plot
            X: observation x features matrix
            y: list of labels
            folds: how many folds of cross-validation
        """
        self.conf_matrix = {}
        fig, ax = plt.subplots(figsize=(28,5), nrows=1, ncols=len(self.classifier_dict.keys()))

        for c, (classifier_name, classifier) in enumerate(self.classifier_dict.items()):   
            
            print(f'Running {classifier_name} classification')
        
            # store this fold's outputs
            k_accuracy = np.zeros((numfolds))
            conf_mat = np.zeros((2,2)) 

            # for roc plot
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            cv = StratifiedKFold(n_splits=numfolds)  
            for k, (train, test) in enumerate(cv.split(X, y)):
                
                # Create a clone of the classifier for each cross-validation
                classifier_clone = clone(classifier)
                if "SVC" in classifier_name:
                    classifier_clone = CalibratedClassifierCV(classifier_clone)

                # fit model & predict
                classifier_clone.fit(X[train], y[train])
                predictions = classifier_clone.predict(X[test])

                # classifier metrics 
                k_accuracy[k] = classifier_clone.score(X[test], y[test])
                conf_mat = conf_mat + confusion_matrix(y[test], predictions) # sum across folds
                
                # ROC plot for this fold
                viz = plot_roc_curve(classifier_clone, X[test], y[test], name=f'ROC fold {k+1}', alpha=0.3, lw=1, ax=ax[c])
                
                # Store tpr and fpr values to calculate mean ROC at the end
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            # Calculate mean ROC and plt  
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0 
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

            ax[c].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
            ax[c].plot(mean_fpr, mean_tpr, color='b',
                    label=f'Mean ROC (AUC={np.round(mean_auc, 2)} +/- {np.round(std_auc, 2)})',
                    lw=2, alpha=.8)
            ax[c].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=f'+/- 1 std. dev.')
            ax[c].set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f'{classifier_name} ROC')
            ax[c].legend(loc="lower right")
    
            self.conf_matrix[classifier_name] = conf_mat

            # Print c.v. results
            print(f"{numfolds}-fold cross-validated accuracy: {np.round(np.mean(k_accuracy), 5)*100}%")
            print(k_accuracy)
            print()
            
            # train each classifier with full data
            classifier.fit(X, y)
    
    def model_metrics(self):
        """
            print out some model metrics
        """
        for c, (classifier_name, conf_matrix) in enumerate(self.conf_matrix.items()):
            print(classifier_name)
            print('Confusion matrix:')
            print('[ TP:', conf_matrix[0][0], ', FP:', conf_matrix[0][1], ']')
            print('[ FN:', conf_matrix[1][0], ', TN:', conf_matrix[1][1], ']')
            accuracy = (conf_matrix[0][0] + conf_matrix[1][1])/np.sum(conf_matrix)
            print(f'Accuracy: {np.round(accuracy*100, 2)}%')
            precision = conf_matrix[0][0]/(conf_matrix[0][0] + conf_matrix[0][1])
            print(f"Precision: {np.round(precision*100, 2)}%")
            recall = conf_matrix[0][0]/(conf_matrix[0][0] + conf_matrix[1][0])
            print(f"Recall: {np.round(recall*100, 2)}%")
            print()
        
    def ensemble_prediction(self):
        """
            take mean class probabilities from each classifier
            threshold on .5 
            
            plot labels & predicted labels
        """
        
        y_tick_labels = []
        self.probas = np.zeros((len(self.X),len(self.classifier_dict)+1)) 
        self.predictions_kk = np.zeros((len(self.X),len(self.classifier_dict)+1))
        for c, (classifier_name, proba) in enumerate(self.proba.items()):
            self.probas[:,c] = self.proba[classifier_name][:,1] # get the probability of class 1
            self.predictions_kk = self.classifier_dict[classifier_name].predict(self.X)
            y_tick_labels.append(classifier_name)
            
        self.probas[:,c+1] = np.mean(self.probas[:,0:len(self.classifier_dict)], axis=1) # average all probabilities togehter
        predictions = (self.probas > .5).astype(int).astype(int) # class 1 predictions
        
        labels_ordered = self.y[self.test_ixs[classifier_name]] # training set indices should be the same across classifiers
        ensemble_accuracy = np.mean(predictions[:,c+1] == labels_ordered) 
        print("Ensemble prediction accuracy: %0.1f%% " % (ensemble_accuracy * 100))
        
        # plot it
        y_tick_labels.insert(0, 'True labels')
        y_tick_labels.append('Ensemble prediction')
        self.prediction_matrix = np.vstack([labels_ordered, predictions.T]).T
        self.prediction_matrix2 = np.vstack([labels_ordered, self.predictions_kk.T]).T
        self.predictions_plot(self.prediction_matrix, y_tick_labels, names)
        
    def predictions_plot(self, prediction_matrix, y_tick_labels, names):
        """
            heatmaps of predictions etc...
        """
        plt.figure(figsize=(15*len(prediction_matrix[1,:]), 10))
        sorted_ind = np.argsort(prediction_matrix[:,0])
        sns.heatmap(prediction_matrix.T[:,sorted_ind], yticklabels=y_tick_labels, xticklabels=np.array(names)[training==1][sorted_ind], cmap='Blues', linewidths=.25, linecolor='black', cbar=False)
        plt.yticks(rotation=60, fontsize = 12) 
        plt.show()

##########################################################################
# permutation utils
##########################################################################

def permutation_clf_cv(clf, X, y, num_folds=10, num_models=1000):
    '''
        clf: classifier object
        X, y: data
        num_folds: how many folds in cv
        num_models: how many shuffled models to run
    '''
    import random
    perm_accs = np.zeros(num_models)
    for n in range(num_models): 
        acc = np.zeros(num_folds)
        cv  = StratifiedKFold(n_splits=num_folds)  
        for k, (train, test) in enumerate(cv.split(X, y)):
            y_train_ = y[train].copy()
            random.shuffle(y_train_) # break connection between data & labels in training
            clf_clone = clone(clf)
            clf_clone.fit(X[train], y_train_)
            acc[k] = clf_clone.score(X[test], y[test])
        perm_accs[n] = np.mean(acc) # mean acc across folds
    return perm_accs

def plot_permutation_dist(perm_acc_dist, obs_acc):
    '''
    '''
    dp = sns.distplot(perm_acc_dist, kde=True) 
    dp.axvline(obs_acc, 0, .95, color='black') # for p-value
    plt.title('Permuted Distribution', fontsize=15)
    plt.xlabel('Accuracy')
    plt.show()
    pval = (len(np.where(np.abs(perm_acc_dist) >= np.abs(obs_acc))[0]) / len(perm_acc_dist))
    if obs_acc > np.median(perm_acc_dist) and pval <.05: # directional hypothesis...
        pval /= 2
    print('Mean accuracy =', np.round(np.mean(perm_acc_dist), 4))
    print('P-value =', np.round(pval, 4))

##########################################################################
# interpretation utils
##########################################################################    
    
def create_network_inds(load_path):
    parcel_names = np.load(load_path)    
    network_inds = {}
    for i, p in enumerate(parcel_names):
        network = '_'.join(p.split('_')[:-1])
        if network in network_inds:
            network_inds[network].append(i)
        else:
            network_inds[network] = [i]
    return parcel_names, network_inds

def convert_upper_mat(arr):
    import math
    approx_dim = (1/2+math.sqrt(1+2*len(arr)))
    if approx_dim - int(approx_dim) > 0.01:
        print('Invalid array shape')
        return
    else:
        dim = int(approx_dim)
    if len(arr.shape) == 1:
        matrix = np.zeros((dim, dim))
        matrix[np.triu_indices(matrix.shape[0], k = 1)] = arr
        matrix = matrix + matrix.T
        return matrix
    elif len(arr.shape) == 2:
        return arr[np.triu_indices(dim, k=1)]
    else:
        print('Invalid array')
        
def get_network_colors(network_inds):
    network_colors = []
    for color in np.arange(90):
        if color in network_inds['ant_sal']:
            network_colors.append(0)
        elif color in network_inds['auditory']:
            network_colors.append(1)
        elif color in network_inds['bg']:
            network_colors.append(2)
        elif color in network_inds['dors_DMN']:
            network_colors.append(3)
        elif color in network_inds['high_visual']:
            network_colors.append(4)
        elif color in network_inds['lang']:
            network_colors.append(5)
        elif color in network_inds['lecn']:
            network_colors.append(6)
        elif color in network_inds['post_sal']:
            network_colors.append(7)
        elif color in network_inds['precuneus']:
            network_colors.append(8)
        elif color in network_inds['prim_visual']:
            network_colors.append(9)
        elif color in network_inds['recn']:
            network_colors.append(10)
        elif color in network_inds['sensorimotor']:
            network_colors.append(11)
        elif color in network_inds['vent_DMN']:
            network_colors.append(12)
        elif color in network_inds['visiospatial']:
            network_colors.append(13)    
    return np.array(network_colors)

class GraphFromCorrv2():
    def __init__(self, matrix, names, network_inds):
        
        # Convert upper matrix to 2D matrix if upper matrix given
        if matrix.ndim == 1:
            matrix = convert_upper_mat(matrix)
        
        # Generate graph and relabel nodes with parcel names
        self.graph = nx.from_numpy_matrix(matrix)
        index_mapping = dict(zip(np.arange(len(names)).astype(int), names))
        self.graph = nx.relabel.relabel_nodes(self.graph, index_mapping)
        self.matrix = matrix

        # Save network assignment of each parcel as a node attribute
        network_list = list(network_inds.keys())
        for parcel_name in names:
            network = '_'.join(parcel_name.split('_')[:-1])
            network_ind = network_list.index(network)
            nx.set_node_attributes(self.graph, {parcel_name: network_ind}, 'network')

        # Save degree centrality as a node attribute
        nx.set_node_attributes(self.graph, nx.degree_centrality(self.graph), 'degree_centrality')

    
    def find_communities(self, levels=1, assign=False):
        communities_generator = nx.algorithms.community.girvan_newman(self.graph)
        for _ in np.arange(levels):
            level_communities = next(communities_generator)
        if assign:
            self.communities = level_communities
            counter = 1
            for comm in level_communities:
                for node_name in comm:
                    nx.set_node_attributes(self.graph, {node_name: counter}, 'community')
                counter +=1
        return level_communities
    
# This function converts a list of stanford parcels to their indices
def parcel_names_to_ind(comm_names, parcel_names):
    inds = np.zeros(len(comm_names))
    for i, n in enumerate(comm_names):
        inds[i] = np.where(parcel_names==n)[0][0]
    return inds.astype(int)

# Define a helper function to get the coordinates and inds
def get_coord(df, name):
    from ast import literal_eval
    return literal_eval(df[df['stanford_name']==name]['coordinates'].to_list()[0])
def get_ind(df, name):
    return df[df['stanford_name']==name]['parcel_ind'].to_list()[0]
def get_description(df, name):
    return df[df['stanford_name']==name]['description'].to_list()[0]

##########################################################################
# meta-analysis utils
##########################################################################  

def stanford_parcellate(img, root):
#     from nilearn import datasets
    from nilearn import image
#     from nilearn import surface
#     from nilearn import plotting
#     from nilearn import input_data
    from nilearn import masking
    
    directory = f'{root}/data/3mm_Stanford_ROIs'
    parcellation = []
    proportion = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".nii"):
            parcel_img = image.load_img(os.path.join(directory, filename))
            res = image.resample_to_img(img, parcel_img)
            masked = np.round(masking.apply_mask(res, parcel_img), 3)
            nonzero_masked = masked[masked!=0]
            if nonzero_masked.size > 0:
                parcellation.append(np.mean(nonzero_masked))
            else:
                parcellation.append(0)
            proportion.append(nonzero_masked.size/len(masked))
        else:
            continue
    return {
        'mean_value': np.array(parcellation),
        'proportion': np.array(proportion)
    }
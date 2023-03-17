import numpy as np
import pandas as pd
import random
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, accuracy_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer

def PT(data, cols):
    PT = PowerTransformer()
    data[cols] = PT.fit_transform(data[cols])
    return data

def set_seed(seed):
    """
    Sets a global random seed of your choice
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def clf_plot_distributions(data, features, hue='target', ncols=3, method='hist'):
    nrows = int(len(features) / ncols) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, round(nrows*16/ncols)))
    for ax,feature in zip(axes.ravel()[:len(features)],features):
        if method == 'hist':
            sns.kdeplot(data=data, x=feature, ax=ax)
        elif method == 'cdf':
            sns.ecdfplot(data=data, x=feature, ax=ax)
        elif method == 'box':
            sns.boxplot(data=data, x=feature, ax=ax)
        elif method == 'bar':
            temp = data.copy()
            temp['counts'] = 1
            temp = temp.groupby([feature], as_index=False).agg({'counts':'sum'})
            sns.barplot(data=temp, x=feature, y='counts', ax=ax)
        elif method == 'hbar':
            temp = data.copy()
            temp['counts'] = 1
            temp = temp.groupby([feature], as_index=False).agg({'counts':'sum'})
            sns.barplot(data=temp, y=feature, x='counts', ax=ax)
    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.show()

def score_clusters(X, predictions, silhouette = False, verbose=False):
    """
    Evaluate how good our cluster label predictions are.
    the silhouette score is the slowest to compute ~90 secs, silhuotte = False to speed up computation
    """
    if silhouette:
        s_score = silhouette_score(X=X, labels=predictions, metric='euclidean')
        ch_score = calinski_harabasz_score(X=X, labels=predictions)
        db_score = davies_bouldin_score(X=X, labels=predictions)

        if verbose:
            print("David Bouldin score: {0:0.4f}".format(db_score))
            print("Calinski Harabasz score: {0:0.3f}".format(ch_score))
            print("Silhouette score: {0:0.4f}".format(s_score))

        return db_score, ch_score ,s_score

    else:
        db_score = davies_bouldin_score(X=X, labels=predictions)
        ch_score = calinski_harabasz_score(X=X, labels=predictions)

        if verbose:
            print("David Bouldin score: {0:0.4f}".format(db_score))
            print("Calinski Harabasz score: {0:0.3f}".format(ch_score))

        return db_score, ch_score

def soft_voting(predict_number, data, best_cols, sampling_size=40000, with_replace=True, max_iter=300, n_init=3, tol=0.001):
    """
    Arguments: Number of predictions rounds, dataframe, columns, sample size of dataframe for each round with replace T/F.
    Returns an array of prediction probabilities
    """
    #initialise dataframe with 0's
    predicted_probabilities = pd.DataFrame(np.zeros((len(data),7)), columns=range(1,8))
    # loop with a different random seeds
    for i in range(predict_number):
        print("=========", i, "==========")
        df_scaled_sample = data.sample(sampling_size, replace=with_replace)
        gmm = BayesianGaussianMixture(n_components=7, covariance_type = 'full', max_iter=max_iter, init_params="kmeans", n_init=n_init, random_state=i, tol=tol)
        gmm.fit(df_scaled_sample[best_cols])
        pred_probs = gmm.predict_proba(data[best_cols])
        pred_probs = pd.DataFrame(pred_probs, columns=range(1,8))
        
        # ensuring clusters are labeled the same value at each fit
        if i == 0:
            initial_centers = gmm.means_
        new_classes = []
        for mean2 in gmm.means_:
            #for the current center of the current gmm, find the distances to every center in the initial gmm
            distances = [np.linalg.norm(mean1-mean2) for mean1 in initial_centers]
            # select the class with the minimum distance
            new_class = np.argmin(distances) + 1 #add 1 as our labels are 1-7 but index is 0-6
            new_classes.append(new_class)
        # if the mapping from old cluster labels to new cluster labels isn't 1 to 1
        if len(new_classes) != len(set(new_classes)):
            print("iteration", i, "could not determine the cluster label mapping, skipping")
            continue
        #apply the mapping by renaming the dataframe columns representing the original labels to the new labels    
        pred_probs = pred_probs.rename(columns=dict(zip(range(1,8),new_classes)))
        
        #add the current prediction probabilities to the overall prediction probabilities
        predicted_probabilities = predicted_probabilities + pred_probs
        # lets score the cluster labels each iteration to see if soft voting is helpful
        score_clusters(data[best_cols], predicted_probabilities.idxmax(axis=1), verbose=True)
    
    #normalise dataframe so each row sums to 1
    predicted_probabilities = predicted_probabilities.div(predicted_probabilities.sum(axis=1), axis=0)
    return predicted_probabilities

def best_class(df):
    """
    Takes the soft voting predictions probabilities and returns a dataframe
    """
    new_df = df.copy()
    new_df["highest_prob"] = df.max(axis=1)
    new_df["best_class"] = df.idxmax(axis=1)
    new_df["second_highest_prob"] = df.apply(lambda x: x.nlargest(2).values[-1], axis=1)
    new_df["second_best_class"] = df.apply(lambda x: np.where(x == x.nlargest(2).values[-1])[0][0]+1, axis=1)
    return new_df

def k_fold_cv(model, X, y, n_splits, model_name, verbose=True):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    feat_imp, y_pred_list, y_true_list, acc_list = [], [], [], []
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(X, y)):
        if verbose: print('=========='*10+f'\n Fold number: {fold}')
        X_train = X.loc[trn_idx]
        X_val = X.loc[val_idx]

        y_train = y.loc[trn_idx]
        y_val = y.loc[val_idx]

        if model_name == 'XGB':
            xy_train, xy_val = y_train.astype(int)-1, y_val.astype(int)-1
            model.fit(X_train, xy_train)
            y_pred = model.predict(X_val)
            y_pred = y_pred.astype(float)+1
            y_val = xy_val.astype(float)+1.

        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

        y_pred_list = np.append(y_pred_list, y_pred)
        y_true_list =- np.append(y_true_list, y_val)

        acc_list.append(accuracy_score(y_pred, y_val))

        if verbose: print(f'Accuracy Score: {acc_list[-1]}')
        try:
            feat_imp.append(model.feature_importances_)
        except AttributeError:
            pass

    return feat_imp, y_pred_list, y_true_list, acc_list, X_val, y_val

def plot_feature_importances(model_importances, feature_names, model_name, n_folds=5, ax=None, boxplot=False):
    importances_df = pd.DataFrame({'feature_cols': feature_names, 'importances_fold_0': model_importances[0]})
    for i in range(1, n_folds):
        importances_df[f'importances_fold_{i}'] = model_importances[i]
    importances_df['importances_fold_median'] = importances_df.drop('feature_cols', axis=1).median(axis=1)
    importances_df = importances_df.sort_values(by='importances_fold_median', ascending=False)
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 25))
    if boxplot == False:
        ax = sns.barplot(data=importances_df, x='importances_fold_median', y='feature_cols', color='blue')
        ax.set_xlabel('Median Feature Importance Across Folds')
    elif boxplot == True:
        importances_df = importances_df.drop('importances_fold_median', axis=1)
        importances_df = importances_df.set_index('feature_cols').stack().reset_index().rename(columns={0: 'feature_importance'})
        ax = sns.boxplot(data=importances_df, y='feature_cols', x='feature_importance', color='blue', orient='h')
        ax.set_xlabel('Feature Importance Across Folds')
    plt.title(model_name)
    ax.set_ylabel('Feature Names')
    return ax

def second_highest(cluster_class_probs, title):
    second_highest_probs_sum = cluster_class_probs.groupby(['best_class', 'second_best_class'])['second_highest_prob'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(25, 13))
    tmp_df = pd.DataFrame({'second_best_class': range(1, 8)})

    for i in range(1, 8):
        second_best_match = second_highest_probs_sum.loc[second_highest_probs_sum['best_class']==i, ['second_best_class', 'second_highest_prob']]
        plot_df = pd.merge(left=tmp_df, right=second_best_match, on='second_best_class', how='left')
        ax = plt.subplot(2, 4, i)
        sns.barplot(data=plot_df, x='second_best_class', y='second_highest_prob', palette=sns.color_palette('hls', 7))
        ax.set_ylabel('Sum of Probability')
        ax.set_title(f'Assigned Class: {str(i)}')
    plt.suptitle(title)

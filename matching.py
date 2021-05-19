import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import NearestNeighbors

from scipy.stats import binom, gaussian_kde, ttest_ind, ranksums
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import os.path


# groups as 1/0 array
def recode_groups(groups, propensity=None):
    treat = propensity[groups == 1]
    control = propensity[groups == 0]
    print('treatment count:', treat.shape)
    print('control count:', control.shape)
    if len(control) > len(treat):
        print("One to many could be an option")
    return treat, control


def savefig(fname, verbose=True):
    path = os.path.join('..', 'figs', fname)
    plt.savefig(path)
    if verbose:
        print("Figure saved as '{}'".format(path))


def rank_test(covariates, groups):
    """
    Wilcoxon rank sum test for the distribution of treatment and control covariates.

    Parameters
    ----------
    covariates : DataFrame
        Dataframe with one covariate per column.
        If matches are with replacement, then duplicates should be
        included as additional rows.
    groups : array-like
        treatment assignments, must be 2 groups

    Returns
    -------
    A list of p-values, one for each column in covariates
    """
    colnames = list(covariates.columns)
    J = len(colnames)
    pvalues = np.zeros(J)
    for j in range(J):
        var = covariates[colnames[j]]
        res = ranksums(var[groups == 1], var[groups == 0])
        pvalues[j] = res.pvalue

    pvalues = pd.Series(pvalues)
    pvalues = round(pvalues, 3)
    return pd.DataFrame({'features': colnames, 'p-value': pvalues})


def t_test(covariates, groups):
    """
    Two sample t test for the distribution of treatment and control covariates

    Parameters
    ----------
    covariates : DataFrame
        Dataframe with one covariate per column.
        If matches are with replacement, then duplicates should be
        included as additional rows.
    groups : array-like
        treatment assignments, must be 2 groups

    Returns
    -------
    A list of p-values, one for each column in covariates
    """
    colnames = list(covariates.columns)
    J = len(colnames)
    pvalues = np.zeros(J)
    for j in range(J):
        var = covariates[colnames[j]]
        res = ttest_ind(var[groups == 1], var[groups == 0])
        pvalues[j] = res.pvalue

    pvalues = pd.Series(pvalues)
    pvalues = round(pvalues, 3)
    return pd.DataFrame({'features': colnames, 'p-value': pvalues})


# use old_df: tab(dataframe with propensity score) , new_df:
def standardized_diff(df):
    '''
    Computes absolute standardized mean differences for covariates by group
    the formula is smd = (Xbar_tret - Xbar_control)/ sqrt((s_tret^2+s_cont^2)/2)
    smd <0.2 good balance
    '''

    features = df.columns.tolist()
    features.remove('treatment')
    features.remove('propensity')

    table = df.groupby('treatment').agg({feature: ['mean', 'std'] for feature in features})
    print(table.head())

    feature_smds = []
    for feature in features:
        feature_table= table[feature].values
        cont_mean = feature_table[0, 0]
        cont_std = feature_table[0, 1]
        tret_mean = feature_table[1, 0]
        tret_std = feature_table[1, 1]

        smd = (tret_mean - cont_mean) / np.sqrt((tret_std ** 2 + cont_std ** 2) / 2)
        smd = round(abs(smd), 4)
        feature_smds.append(smd)

    return pd.DataFrame({'features': features, 'smd': feature_smds})



# def plotScores(groups, propensity, many=True):
#     '''
#     Plot density of propensity scores for each group before and after matching
#
#     Inputs: groups = treatment assignment, pre-matching
#             propensity = propensity scores, pre-matching
#             matches = output of Match or MatchMany
#             many = indicator - True if one-many matching was done (default is True), otherwise False
#     '''
#
#     # 1 Density distribution of propensity score (logic) broken down by treatment status
#
#

class Match:
        """
        Parameters
        ----------
        groups : array-like
            treatment assignments, must be 2 groups
        propensity : array-like
            object containing propensity scores for each observation.
            Propensity and groups should be in the same order (matching indices)
        """

        # groups like treatments
        def __init__(self, groups, propensity):
            self.groups = groups
            self.propensity = propensity
            assert self.groups.shape == self.propensity.shape, "Input dimensions dont match"
            assert len(np.unique(self.groups) == 2), "Wrong number of groups"
            assert all(self.propensity >= 0) and all(self.propensity <= 1), "Propensity scores must be between 0 and 1"

        # pass in new df from propensity score class
        def match_method_knn(self, df, k=1):
            """
            Implements greedy one-to-many matching on propensity scores.
            Parameters
            ----------
            df: dataframe that has propensity scores generated by PropensityScore class
            k : int
             (default is 1). If method is "knn", this specifies the k in k nearest neighbors
            """
            treat, control = recode_groups(self.groups, self.propensity)

            knn = NearestNeighbors(n_neighbors=k)
            control = control.to_numpy()
            treat = treat.to_numpy()
            knn.fit(control.reshape(-1, 1))
            distances, indices = knn.kneighbors(treat.reshape(-1, 1))

            # get treatment and control match
            df_treatment = df[self.groups == 1]
            df_control = df[self.groups == 0]

            if k == 1:
                print("this is a one-to-one matching")
                indices = indices.reshape(indices.shape[0])
                df_control_mat = df_control.iloc[indices]
                df_matched = pd.concat([df_treatment, df_control_mat])
                return df_matched

            elif k > 1:
                matched_list = list()
                attribute = df_treatment.columns
                print("since k>1, this is a one-to-many matching.")
                df_control = df_control.to_numpy()

                for k in range(k):
                    matches = []
                    for j in indices[:, k]:
                        matches.append(df_control[j].tolist())

                    matches = pd.DataFrame(matches, columns=attribute)
                    df_matched = pd.concat([df_treatment, matches])
                    matched_list.append(df_matched)

                return matched_list

            else:
                raise ValueError('Invalid k value')

        def test_balance(self, matched_df=None, old_df=None, test=None, many=True):
            if test is None:
                test = ['smd', 't', 'rank', 'plot']

            if many:
                smd_b = list()
                smd_a = list()
                t = list()
                rank = list()
                for i in range(len(matched_df)):
                    if 'smd' in test:
                        smd_before = standardized_diff(old_df[i])
                        smd_after = standardized_diff(matched_df[i])
                        smd_b.append(smd_before)
                        smd_a.append(smd_after)

                        print("standardized mean differences %.4f" % smd_b)
                        print("standardized mean differences %.4f" % smd_a)

                    if 't' in test:
                        features = matched_df[i].columns.tolist()
                        features.remove('treatment')
                        features.remove('propensity')
                        ttest= t_test(features, matched_df[i]['treatment'])
                        t.append(ttest)
                        print("The test result for two sample t-test %d" % t)

                    if 'rank' in test:
                        features = matched_df[i].columns.tolist()
                        features.remove('treatment')
                        features.remove('propensity')
                        rtest = rank_test(features, matched_df[i]['treatment'])
                        rank.append(rtest)
                        print("The test result for Wilcoxon-test %d" % rank)

                    if 'plot' in test:
                        fig, ax = plt.subplots(1, 2)
                        fig.suptitle('Density distribution plots for propensity score and logit(propensity score).')
                        predictions_logit = np.array([np.logit(xi) for xi in matched_df[i]['propensity']])
                        sns.kdeplot(x= matched_df[i]['propensity'], hue="treatment", ax=ax[0])
                        ax[0].set_title('Propensity Score')

                        sns.kdeplot(x=predictions_logit , hue="treatment", ax=ax[1])
                        ax[1].axvline(-0.4, ls='--')
                        ax[1].set_title('Logit of Propensity Score')
                        plt.savefig('propensity score.png')
                        plt.show()

                        savefig('propensity_{:03d}.jpg'.format(i))

            else:
                if 'smd' in test:
                    smd_before = standardized_diff(old_df)
                    smd_after = standardized_diff(matched_df)

                    print("standardized mean differences %.4f" % smd_before)
                    print("standardized mean differences %.4f" % smd_after)

                if 't' in test:
                    features = matched_df.columns.tolist()
                    features.remove('treatment')
                    features.remove('propensity')
                    ttest = t_test(features, matched_df['treatment'])
                    print("The test result for two sample t-test %d" % ttest)

                if 'rank' in test:
                    features = matched_df.columns.tolist()
                    features.remove('treatment')
                    features.remove('propensity')
                    rtest = rank_test(features, matched_df['treatment'])
                    print("The test result for Wilcoxon-test %d" % rtest)

                if 'plot' in test:
                    fig, ax = plt.subplots(1, 2)
                    fig.suptitle('Density distribution plots for propensity score and logit(propensity score).')
                    predictions_logit = np.array([np.logit(xi) for xi in matched_df['propensity']])
                    sns.kdeplot(x=matched_df['propensity'], hue="treatment", ax=ax[0])
                    ax[0].set_title('Propensity Score')

                    sns.kdeplot(x=predictions_logit, hue="treatment", ax=ax[1])
                    ax[1].axvline(-0.4, ls='--')
                    ax[1].set_title('Logit of Propensity Score')
                    plt.savefig('propensity score.png')
                    plt.show()

                    savefig('propensity_{:03d}.jpg'.format(i))





























































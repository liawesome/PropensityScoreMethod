import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import NearestNeighbors
import scipy
from scipy.stats import binom, gaussian_kde, ttest_ind, ranksums
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import os.path
from itertools import chain


# groups as 1/0 array
def recode_groups(groups, propensity=None):
    treat = propensity[groups == 1]
    control = propensity[groups == 0]
    print('treatment count:', treat.shape)
    print('control count:', control.shape)
    

    if ((len(control)> 0) and (len(treat)>0)):
        if len(control) > len(treat):
            print("One to many could be an option")
    else:
        print("invalid control/treatment size")
        
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


def t_test(df, covariates, groups):
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

    J = len(covariates)
    pvalues = np.zeros(J)
    for j in range(J):
        var = df[covariates[j]]
        res = ttest_ind(var[groups == 1], var[groups == 0])
        pvalues[j] = res.pvalue

    pvalues = pd.Series(pvalues)
    pvalues = round(pvalues, 3)
    return pd.DataFrame({'features': covariates, 'p-value': pvalues})

def transform_data(df):
    features = list(df.columns)
    features.remove('treatment')
    features.remove('propensity')
    return features



# use old_df: tab(dataframe with propensity score) , new_df:
def standardized_diff(df):
    '''
    Computes absolute standardized mean differences for covariates by group
    the formula is smd = (Xbar_tret - Xbar_control)/ sqrt((s_tret^2+s_cont^2)/2)
    smd <0.2 good balance
    '''
    features = transform_data(df)
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


def prop_test(col):
    """
    Performs a Chi-Square test of independence on <col>
    See stats.chi2_contingency()
    Parameters
    ----------
    col : str
        Name of column on which the test should be performed
    Returns
    ______
    dict
        {'var': <col>,
         'before': <pvalue before matching>,
         'after': <pvalue after matching>}
    """
    
    pval_before = round(scipy.stats.chi2_contingency(prep_prop_test(data,col))[1], 6)
    pval_after = round(scipy.stats.chi2_contingency(prep_prop_test(matched_data,col))[1], 6)
    return {'var':col, 'before':pval_before, 'after':pval_after}



def prep_prop_test(data, var):
    """
    Helper method for running chi-square contingency tests
    Balances the counts of discrete variables with our groups
    so that missing levels are replaced with 0.
    i.e. if the test group has no records with x as a field
    for a given column, make sure the count for x is 0
    and not missing.
    Parameters
    ----------
    data : pd.DataFrame()
        Data to use for counting
    var : str
        Column to use within data
    Returns
    -------
    list
        A table (list of lists) of counts for all enumerated field within <var>
        for test and control groups.
    """
    counts = data.groupby([var, yvar]).count().reset_index()
    table = []
    for t in (0, 1):
        os_counts = counts[counts[yvar] ==t]\
                                 .sort_values(var)
        cdict = {}
        for row in os_counts.iterrows():
            row = row[1]
            cdict[row[var]] = row[2]
        table.append(cdict)
    # fill empty keys as 0
    all_keys = set(chain.from_iterable(table))
    for d in table:
        d.update((k, 0) for k in all_keys if k not in d)
    ctable = [[i[k] for k in sorted(all_keys)] for i in table]
    return ctable

def chi_square(old_df,matched_df, col):
    
    title_str = '''
    Proportional Difference (test-control) for {} Before and After Matching
    Chi-Square Test for Independence p-value before | after:
    {} | {}
    '''
    test_results = []

    for col in cat_column:
        dbefore = prep_plot(old_df, col, colname="before")
        dafter = prep_plot(matched_df, col, colname="after")
        df = dbefore.join(dafter)
        test_results_i = prop_test(col)
        test_results.append(test_results_i)

    # plotting
    df.plot.bar(alpha=.8)
    plt.title(title_str.format(col, test_results_i["before"],
               test_results_i["after"]))
    lim = max(.09, abs(df).max().max()) + .01
    plt.ylim((-lim, lim))
    return pd.DataFrame(test_results)[['var', 'before', 'after']]


def prep_plot(data, var,colname):
    # treat, control 
    c = data[data[yvar] == 0]
    t = data[data[yvar] == 1]
    # dummy var for counting
    dummy = [i for i in t.columns if i not in \
              (var, "match_id", "record_id", "weight")][0]
    countt = t[[var, dummy]].groupby(var).count() / len(t)
    countc = c[[var, dummy]].groupby(var).count() / len(c)
    ret = (countt-countc).dropna()
    ret.columns = [colname]
    return ret   



def plot_scores(old_df, match_df,args):
    fig, ax =plt.subplots(1,2)
    sns.distplot(old_df.propensity[old_df['treatment'] == 0], label='Control', ax=ax[0])
    sns.distplot(old_df.propensity[old_df['treatment'] == 1], label='Treat', ax=ax[0])
    ax[0].set_title('Before Matching')
    ax[0].set(xlabel='Scores')
    ax[0].set(ylabel='Density')
    
    sns.distplot(match_df.propensity[match_df['treatment'] == 0], label='Control',ax=ax[1])
    sns.distplot(match_df.propensity[match_df['treatment'] == 1], label='Treat', ax=ax[1])
    plt.legend(loc='upper right')
    ax[1].set_title("After Matching")
    ax[1].set(xlabel='Scores')
    ax[1].legend(loc='center left',bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.show()

    
    for arg in args:
        fig, ax = plt.subplots(1,2)
        fig.suptitle('Comparison of {} split by treatment status.'.format(arg))
        sns.distplot(old_df[old_df['treatment'] == 0][arg], label="Control", ax=ax[0])
        sns.distplot(old_df[old_df['treatment'] == 1][arg], label="treated", ax=ax[0])
        ax[0].set_title('Before Matching')
        ax[0].set(xlabel='Scores')
        
        sns.distplot(match_df[match_df['treatment'] == 0][arg], label="Control", ax=ax[1])
        sns.distplot(match_df[match_df['treatment'] == 1][arg], label="treated", ax=ax[1])
        ax[1].set_title("After Matching")
        ax[1].set(xlabel='Scores')
        ax[1].legend(loc='center left',bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.show()
        fig.tight_layout()
        fig.savefig("{}_comparison.png".format(arg))      

# plot categorical variables       
def plot_catg(old_df, matched_df, args):
    
    for arg in args:
        
        res= old_df.groupby([arg,'treatment']).size()
        tab = res / res.groupby(level=0).sum()
        d = tab.reset_index(name='count')
        
        fig, (ax1, ax2) =plt.subplots(1,2)
        d.pivot(arg, "treatment", "count").plot(kind='bar', ax =ax1)
        plt.xlabel(arg) 
        plt.ylabel('proportion')
        ax1.set_title('Before Matching')
        
        result= matched_df.groupby([arg,'treatment']).size()
        tab = result / result.groupby(level=0).sum()
        d = tab.reset_index(name='count')
        
        d.pivot(arg, "treatment", "count").plot(kind='bar',ax=ax2)
        plt.xlabel(arg) 
        plt.ylabel('proportion')
        ax2.set_title('After Matching')
        fig.tight_layout()
       
        
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
            assert len(np.unique(self.groups) != 2), "Wrong number of groups"
            assert all(self.propensity >= 0) and all(self.propensity <= 1), "Propensity scores must be between 0 and 1"

        # pass in new df from propensity score class
        def match_method_knn(self, df, k=1):
            """
            Implements greedy one-to-many matching on propensity scores.
            Parameters
            ----------
            df: dataframe that has propensity scores generated by PropensityScore class
            k : int
             (default is 1). If method is "knn", this specifies the k in k nearest neighbor
            """
            try:
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


                    for k in range(k):
                        matches = []
                        for j in indices[:, k]:
                            matches.append(df_control.iloc[j])

                        matches = pd.DataFrame(matches, columns=attribute)
                        df_matched = pd.concat([df_treatment, matches])
                        matched_list.append(df_matched)

                    return matched_list

                else:
                    raise ValueError('Invalid k value')
                    
            except ValueError:
                return None
                
        

        def test_balance(self, num_columns, catg_columns, matched_df=None, old_df=None, test=None, many=True):
            ###########new#########
            columns = num_columns + cat_columns
            columns.append('treatment')
            columns.append('propensity')
            
            matched_df= matched_df[columns]
            old_df= old_df[columns]
            
            if test is None:
                test = ['smd', 't', 'rank','chi-sq', 'plot']

            if many:
                smd_b = list()
                smd_a = list()
                t = list()
                rank = list()
                for i in range(len(matched_df)):
                    if 'smd' in test:
                        old_df = pd.get_dummies(old_df, columns=catg_columns)
                        matched_df = pd.get_dummies(matched_df, columns=catg_columns)
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
                        print(t)

                    if 'rank' in test:
                        features = matched_df[i].columns.tolist()
                        features.remove('treatment')
                        features.remove('propensity')
                        rtest = rank_test(features, matched_df[i]['treatment'])
                        rank.append(rtest)
                        print("The test result for Wilcoxon-test:")
                        print(rank)

                    if 'plot' in test:
                        print('For each matched data'.format(i))
                        plot_scores(old_df, matched_df, args)
                        plot_catg(old_df, matched_df,catg_columns)
                        

            else:
                if 'smd' in test:
                    smd_before = standardized_diff(old_df)
                    smd_after = standardized_diff(matched_df)

                    print(smd_before)
                    print(smd_after)

                if 't' in test:
                    features = transform_data(matched_df)
                    ttest = t_test(matched_df, features, matched_df['treatment'])
                    print("The test result for t-test:")
                    print(ttest)

                # for continuous data 
                if 'rank' in test:
                    features = list(matched_df.columns)
                    features.remove('treatment')
                    features.remove('propensity')
                    rtest = rank_test(features, matched_df['treatment'])
                    print("The test result for Wilcoxon-test:")
                    print(rtest)
                    
                if 'plot' in test:
                    args = num_column
                    plot_scores(old_df, matched_df, args)
                    plot_catg(old_df, matched_df,catg_columns)
                    
                    
                    

    


def logit(x):
    propensity = np.log(x / (1 - x))
    return x


























































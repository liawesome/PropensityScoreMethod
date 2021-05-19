import os
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import propsensity
import pandas as pd
from propsensity import PropensityScore
from matching import Match
from matching import standardized_diff,t_test,rank_test

if __name__ == "__main__":
    dataset = pd.read_csv('TNL_Universe_Final.csv')

    num_column = ['AGE', 'TENURE']
    # print(num_column)
    cat_column = ['education', 'housing', 'Marital']
    p = PropensityScore(dataset, num_column, cat_column)
    df = p.compute_score(method='logistic', penalty='l2')
    #print(df)
    match = Match(df['treatment'], df['propensity'])
    result = match.match_method_knn(df, k=2)

    print(len(result))


    res = standardized_diff(result)
    #print(res)
    features = result.columns.tolist()
    features.remove('treatment')
    features.remove('propensity')
    a = t_test(result[features], result['treatment'])
    b = rank_test(result[features], result['treatment'])
    print(b)

   # r = match.test_balance(matched_df=result, old_df=df, test='t', many=False)



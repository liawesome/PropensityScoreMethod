import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.compose import ColumnTransformer
import pandas as pd
import statsmodels.api as sm


class PropensityScore:

    def __init__(self, df, numer_columns, catg_columns, target_col='groups', target_var='T'):
        """
        Parameters
        -----------
        df : original dataset
        numer_columns: numeric covariates
        cat_columns: categorical covariates
        """
        self.df = df
        self.numer_columns = numer_columns
        self.catg_columns = catg_columns
        self.target_col = target_col
        self.target_var = target_var

    def compute_score(self, method ='logistic', C=1e6, penalty=None, solver='lbfgs'):

        # transform the group into binary 0/1
        self.df['treatment'] = self.df.apply(lambda x: 1 if x[self.target_col] == self.target_var else 0, axis=1)

        if self.target_col in self.catg_columns:
            self.catg_columns.remove(self.target_col)

        t = self.df['treatment'].values # pass as 0/1 value
        columns = self.numer_columns + self.catg_columns
        # transform categorical variable to dummy variable
        X = self.df[columns]
        X_encoded = pd.get_dummies(X, columns=self.catg_columns)

        # standardise the numeric columns if necessary
        # column_transformer = ColumnTransformer(
        #     [('numerical', StandardScaler(), self.numer_columns)],
        #     sparse_threshold=0,
        #     remainder='passthrough'
        # )
        # X_encoded = column_transformer.fit_transform(X_encoded)

        # get propensity score
        if method == 'logistic':
            logistic = LogisticRegression(C=C, penalty=penalty, solver=solver)
            propensity = logistic.fit(X_encoded,t).predict_proba(X_encoded)[:, 1]
            predictions_binary = logistic.fit(X_encoded, t).predict(X_encoded)
            # question!!!
            X_encoded['propensity'] = propensity

        elif method == 'probit':
            covariates = sm.add_constant(X_encoded, prepend=False)
            probit = sm.Probit(t, covariates).fit(disp=False, warn_convergence=True)
            ps = probit.predict()
            X_encoded['propensity'] = ps
            predictions_binary = [0 if x < 0.5 else 1 for x in ps]
        else:
            raise ValueError('Invalid method')

        print('Accuracy: {:.4f}\n'.format(metrics.accuracy_score(t, predictions_binary)))
        print('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(t, predictions_binary)))
        print('F1 score is: {:.4f}'.format(metrics.f1_score(t, predictions_binary)))

        X_encoded['treatment'] = t

        return X_encoded
        # return df containing propensity scores, treatment for each observation







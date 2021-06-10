import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.compose import ColumnTransformer
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV

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

    def compute_score(self, method ='logistic', penalty=None, solver='lbfgs',max_iter =10000):

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
            lr = LogisticRegressionCV()
            propensity = lr.fit(X_encoded,t).predict_proba(X_encoded)[:, 1]
            predictions_binary = lr.fit(X_encoded, t).predict(X_encoded)
            X['propensity']= propensity
            X['treatment']= self.df['treatment']

        elif method == 'rf':
            rf = RandomForestClassifier(max_depth=10,class_weight={0:1,1:1})
            propensity = rf.fit(X_encoded,t).predict_proba(X_encoded)[:, 1]
            predictions_binary = rf.fit(X_encoded, t).predict(X_encoded)
            
            param_range = np.arange(1, 20, 1)

            # Calculate accuracy on training and test set using range of parameter values
            train_scores, test_scores = validation_curve(RandomForestClassifier(class_weight={0:1,1:1}), 
                                                         X_encoded, 
                                                         t, 
                                                         param_name="n_estimators", 
                                                         param_range=param_range, cv=5,
                                                         scoring="accuracy")


            # Calculate mean and standard deviation for training set scores
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)

            # Calculate mean and standard deviation for test set scores
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            # Plot mean accuracy scores for training and test sets
            plt.plot(param_range, train_mean, label="Training score", color="blue")
            plt.plot(param_range, test_mean, label="Cross-validation score", color="orange")

            # Plot accurancy bands for training and test sets

            # Create plot
            plt.title("Validation Curve With Random Forest")
            plt.xlabel("Number Of Trees")
            plt.ylabel("Accuracy Score")
            plt.tight_layout()
            plt.legend(loc="best")
            plt.show()
            X['propensity']= propensity
            X['treatment']= self.df['treatment']
        
        else:
            raise ValueError('Invalid method')

        print('Accuracy: {:.4f}\n'.format(metrics.accuracy_score(t, predictions_binary)))
        print('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(t, predictions_binary)))
        print('F1 score is: {:.4f}'.format(metrics.f1_score(t, predictions_binary)))


        return X
        # return df containing propensity scores, treatment for each observation







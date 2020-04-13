# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn import metrics
import xgboost as xgb

from imblearn.ensemble import BalancedBaggingClassifier

import graphviz
import plotly.graph_objects as go
import plotly.offline as py

from plot_learning_curve import plot_learning_curve

import warnings

def feature_importance_scatter_plot(model_name,model_feature_importance,feature_data):
    # Scatter plot 
    trace = go.Scatter(
        y = model_feature_importance,
        x = feature_data.columns,
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 25,
    #       size= feature_dataframe['AdaBoost feature importances'].values,
            #color = np.random.randn(500), #set color equal to a variable
            color = model_feature_importance,
            colorscale='Portland',
            showscale=True
        ),
        text = feature_data.columns
    )
    data = [trace]

    layout= go.Layout(
        autosize= True,
        title= model_name,
        hovermode= 'closest',
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
        yaxis=dict(
            title= 'Feature Importance',
            ticklen= 5,
            gridwidth= 2
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig,filename='scatter2010')
c_flag_c = 2100
def model_describe(y_test,y_pred,y_pred_prob):
    print('Model Summary Report')
    print(metrics.classification_report(y_test,y_pred))
    print('\nAUC score',metrics.roc_auc_score(y_test,y_pred_prob))
    fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred_prob)
    plt.figure()
    plt.plot(fpr,tpr)
    plt.title('ROC Curve')
    plt.show()
c_flag = 370

def model_train(x_train, x_test, y_train, y_test, model_func, parameters_grid=None):

    if parameters_grid:
        model = model_func()

        gridSearch = GridSearchCV(model,parameters_grid,cv = 10, scoring=roc_auc_weighted)
        gridSearch.fit(x_train,y_train)
    #     print(gridSearch.best_score_)
        best_params = gridSearch.best_params_
        print(best_params)

        model = model_func()
        model.set_params(**gridSearch.best_params_)
    else:
        model = model_func()

    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    try:
        y_pred_prob = model.predict_proba(x_test)[:,1]
    except:
        y_pred_prob = y_pred
    print('In-Sample Accuracy: ', round(model.score(x_train, y_train) * 100, 2), '%')
    print('Out-of-Sample Accuracy: ', round(model.score(x_test, y_test) * 100, 2), '%', '\n')
    model_describe(y_test,y_pred,y_pred_prob)
    
    return model

def my_preprocess(a, b, c, d):
    
    global my_count
    
    if my_count == 0:
        a = pd.concat([a, b.iloc[:c_flag]])
        c = pd.concat([c, d.iloc[:c_flag]])
        my_count += 1
        
    elif my_count == 1:
        a = pd.concat([a, b.iloc[:c_flag_c]])
        c = pd.concat([c, d.iloc[:c_flag_c]])

    return a, b, c, d

def get_pred_prob(model, x_test):
    y_pred = model.predict(x_test)
    try:
        y_pred_prob = model.predict_proba(x_test)[:,1]
    except:
        y_pred_prob = y_pred
    return y_pred_prob
my_count = 0
def comb_result(nc_model, c_model):
        
    global c_X_test
    global nc_X_test
    global c_Y_test
    global nc_Y_test
    
    c_y_pred = c_model.predict(c_X_test)
    c_y_pred_prob = get_pred_prob(c_model, c_X_test)
    
    nc_y_pred = nc_model.predict(nc_X_test)
    nc_y_pred_prob = get_pred_prob(nc_model, nc_X_test)
    
    Y_test = pd.concat([c_Y_test, nc_Y_test]).values
    global a
    global b
    a = c_y_pred
    b = nc_y_pred
    y_pred = list(c_y_pred) + list(nc_y_pred)
    y_pred_prob = list(c_y_pred_prob) + list(nc_y_pred_prob)
    
    model_describe(Y_test,y_pred,y_pred_prob)
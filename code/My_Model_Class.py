import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error, roc_auc_score, make_scorer, roc_curve, auc

class My_Model():
    
    def __init__(self, X, y, model_func, lambda_func, cv_num=10, lambda_set=np.logspace(-5, 5, 20), test_size=0.1):
        self.__X = X
        self.__y = y
        self.__model_func = model_func
        self.__lambda_func = lambda_func
        self.__cv_num = cv_num
        self.__lambda_set = lambda_set
        self.__test_size = test_size
        
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(X, y, shuffle=True, test_size=self.__test_size)

        self.__run_cross_validation()
        self.__train_best_model()
        
    def __run_cross_validation(self):

        def my_custom_loss_func(y_true, y_pred):

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            y_sigmoid_pred = sigmoid(y_pred)

            return mean_squared_error(y_true, y_sigmoid_pred)

        score = make_scorer(my_custom_loss_func, greater_is_better=False)

        loss_list = []

        for my_lambda in self.__lambda_set:

            temp_model = self.__model_func(alpha=my_lambda)
            loss = -cross_val_score(temp_model, self.__X, self.__y, cv=self.__cv_num, scoring=score)
            loss_list.append(loss)

        self.__loss_list = loss_list
    
    def __train_best_model(self):
        self.__best_lambda = self.__lambda_func(self.__lambda_set, self.__loss_list)
        self.__best_model = self.__model_func(self.__best_lambda)

        self.__y_pred = self.__best_model.fit(self.__X_train, self.__y_train).predict(self.__X_test)

        self.__roc_score=roc_auc_score(self.__y_test, self.__y_pred)
        
    def get_model_func(self):
        return self.__model_func
    
    def get_lambda_func(self):
        return self.__lambda_func
    
    def get_lambda_set(self):
        return self.__lambda_set
    
    def get_loss_list(self):
        return self.__loss_list
    
    def get_best_lambda(self):
        return self.__best_lambda
    
    def get_best_model(self):
        return self.__best_model
    
    def get_y_test(self):
        return self.__y_test

    def get_y_pred(self):
        return self.__y_pred
    
    def get_roc_score(self):
        return self.__roc_score
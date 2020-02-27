from functools import reduce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

class My_Table_Plotter():
    # def __init__(self, result_model_dict, model_name_list, connect_name_list):
    #     self.__model_dict = result_model_dict
    #     self.__model_name_list = model_name_list
    #     self.__connect_name_list = connect_name_list
        
        # if len(model_name_list) > 1:
        #     self.__multi_index = pd.MultiIndex.from_tuples(list(reduce(lambda x,y:x+y,list(map(lambda z:list(zip([z]*len(connect_name_list), connect_name_list)), model_name_list)))))
        # else:
        #     self.__multi_index = pd.MultiIndex.from_tuples(list(list(zip(model_name_list*len(connect_name_list), connect_name_list))))
    
    def __init__(self, result_model_dict):
        self.__model_dict = result_model_dict

        self.__keys = list(result_model_dict.keys())
        self.__multi_index = pd.MultiIndex.from_tuples(self.__keys)

        self.__model_name_list = list(self.__multi_index.levels[0])
        self.__connect_name_list = list(self.__multi_index.levels[1])
        self.__threshold_list = list(self.__multi_index.levels[2])

    def get_coef_df(self):
        m = self.__model_dict[self.__keys[0]].get_best_model()

        col_name_list = list()
        _ = list(map(lambda x:col_name_list.append("V"+str(x+1)), range(len(m.coef_))))

        coef_df = pd.DataFrame(index=self.__keys, data=0, columns=col_name_list)

        for key, model in self.__model_dict.items():
            coef_df.loc[key, :] = model.get_best_model().coef_

        coef_df[coef_df==0] = np.nan

        coef_df.index = self.__multi_index

        return coef_df

    def get_num_of_selected_var_df(self, threshold=0.1):
        num_of_selected_var_df = pd.DataFrame(index=self.__connect_name_list, columns=self.__model_name_list, data=np.nan)

        for model_name in self.__model_name_list:
            for connect_name in self.__connect_name_list:
                model = self.__model_dict[(model_name, connect_name, threshold)]
                temp = model.get_best_model().coef_
                temp[temp != 0] = 1
                num_of_selected_var_df.loc[connect_name, model_name] = np.sum(temp)
        
        return num_of_selected_var_df

    def get_roc_cruve(self):
        for model_name in self.__model_name_list:
            plt.figure()
            legend_list = ["CSM"]

            for connect_name in self.__connect_name_list[:1]:
                cur_model = self.__model_dict[(model_name, connect_name, 0.1)]
                
                false_positive_rate, true_positive_rate, thresholds = roc_curve(cur_model.get_y_test(), cur_model.get_y_pred())
                roc_auc=auc(false_positive_rate, true_positive_rate)

                plt.plot(false_positive_rate, true_positive_rate,label='%s = %0.4f' % (connect_name, roc_auc))
            
            for threshold in self.__threshold_list:
                legend_list.append("NS-CSM(threshold=%.2f)" % threshold)

                y_test_list = []
                y_pred_list = []

                for connect_name in self.__connect_name_list[1:]:
                    cur_model = self.__model_dict[(model_name, connect_name, threshold)]
                
                    y_test_list.append(cur_model.get_y_test())
                    y_pred_list.append(cur_model.get_y_pred())
                
                y_test = np.hstack(y_test_list)
                y_pred = np.hstack(y_pred_list)

                false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
                roc_auc=auc(false_positive_rate, true_positive_rate)

                plt.plot(false_positive_rate, true_positive_rate,label='%s = %0.4f' % (connect_name, roc_auc))

            plt.plot([0,1],[0,1],'r--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title("ROC Curve - %s" % model_name)
            plt.legend(legend_list, loc="lower right")
            plt.show()

    def get_roc_df(self):
        roc_df = pd.DataFrame()

        for model_name in self.__model_name_list:
            col_values = []

            for connect_name in self.__connect_name_list[:1]:
                cur_model = self.__model_dict[(model_name, connect_name, 0.1)]
                
                false_positive_rate, true_positive_rate, thresholds = roc_curve(cur_model.get_y_test(), cur_model.get_y_pred())
                roc_auc=auc(false_positive_rate, true_positive_rate)

                col_values.append(roc_auc)

            for threshold in self.__threshold_list:
                y_test_list = []
                y_pred_list = []

                for connect_name in self.__connect_name_list[1:]:
                    cur_model = self.__model_dict[(model_name, connect_name, threshold)]
                
                    y_test_list.append(cur_model.get_y_test())
                    y_pred_list.append(cur_model.get_y_pred())
                
                y_test = np.hstack(y_test_list)
                y_pred = np.hstack(y_pred_list)

                false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
                roc_auc=auc(false_positive_rate, true_positive_rate)

                col_values.append(roc_auc)
            
            roc_df = pd.concat([roc_df, pd.DataFrame(columns=[model_name], data=col_values)], axis=1)
            roc_df.index = ["CSM", "NS-CSM(γ = 0.05)", "NS-CSM(γ = 0.1)"]
        return roc_df

    def get_auc_df(self):
        pass
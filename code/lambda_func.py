import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def lambda_min_func(lambda_set, loss_list, plot_flag=False):

    temp_result = list(map(np.mean, loss_list))

    if plot_flag:
        plt.figure()
        plt.plot(list(map(np.mean, loss_list)))

        plt.title("lambda.min")
        plt.xlabel("lambda index")
        plt.ylabel("mean square error")
        plt.show()

    return lambda_set[np.argmin(temp_result)]

def lambda_1se_func(lambda_set, loss_list, plot_flag=False):

    def one_se_minimum_mean_func(my_list):
        return np.mean((np.array(my_list) - np.min(my_list))**2)

    temp_result = list(map(one_se_minimum_mean_func, loss_list))

    if plot_flag:
        plt.figure()
        plt.plot(temp_result)

        plt.title("lambda.1se")
        plt.xlabel("lambda index")
        plt.ylabel("minimum mean square error")
        plt.show()

    return lambda_set[np.argmin(temp_result)]
#!/usr/bin/env python
# coding: utf-8

# In[18]:


#importing important libraries
import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_squared_error
import Functions


# In[19]:


def main(classes, size, time_series, name, seasonal):
    global nb_classes
    global test_size
    global serie
    nb_classes = classes
    test_size = size
    serie = time_series


    logging.info("***Inicia auto-ARIMA***" )

    #divide into train and validation set


    x_train = time_series[:len(time_series)-size]
    test = time_series[len(time_series)-size:]

    mse = 0
    for i in range ((len(test)- nb_classes)):

        train = np.concatenate((x_train, test[:i]))
        model = auto_arima(train, trace=True,start_p=0, start_q=0, start_P=0, start_Q=0,
                          max_p=10, max_q=10, max_P=10, max_Q=10, seasonal=seasonal,
                          stepwise=False, suppress_warnings=True, D=1, max_D=10,
                          error_action='ignore',approximation = False)

        model.fit(train)
        y_pred = model.predict(n_periods= nb_classes)
        y_pred = y_pred.reshape(nb_classes,1)
        mse_partial =  mean_squared_error(test[i:nb_classes+i], y_pred)
        mse = mse + mse_partial
        logging.info("ARIMA MSE step prediction: %.10f%%" % (mse_partial))

    mse = mse / len(test)
    logging.info("ARIMA Total MSE: %.10f%%" % (mse))

    return mse


# In[ ]:

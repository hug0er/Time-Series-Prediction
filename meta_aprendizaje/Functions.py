#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np


class TimeSeriesNN3():
    def __init__(self,STEPS_AHEAD ,NUMBER_TESTING , SERIE, WINDOW):
        self.STEPS_AHEAD = STEPS_AHEAD
        self.WINDOW = WINDOW
        self.NUMBER_TESTING = NUMBER_TESTING
        self.SERIE = SERIE
        self.SIZE = SERIE.shape[0]
        self.NUMBER_TRAINING = self.SIZE - self.NUMBER_TESTING

    def divide_testing (self):
        return (self.SERIE[:self.NUMBER_TRAINING],self.SERIE[self.NUMBER_TRAINING:])

    def divide_validation (self,serie):
        experimentx = np.zeros((len(serie) - self.WINDOW -self.STEPS_AHEAD + 1, self.WINDOW))
        experimenty = np.zeros((len(serie) - self.WINDOW -self.STEPS_AHEAD + 1,self.STEPS_AHEAD))
        cnt =  0
        cnt2 = 0
        #serie = serie.reset_index(drop = True)
        for i in range (experimentx.shape[0]):
            cnt += cnt2
            for j in range (self.WINDOW):
                experimentx[i][j] = serie[cnt]
                cnt += 1
            #print(serie[cnt:cnt + self.STEPS_AHEAD], " \n")
            valores = serie[cnt:cnt + self.STEPS_AHEAD]
            for h in range(len(valores)):
                experimenty[i][h] = valores[h]
            cnt = 0
            cnt2 += 1
        return (experimentx, experimenty)

    def batch (self):
        training,testing = self.divide_testing()
        tex,tey = self.divide_validation(testing)
        tx,ty = self.divide_validation(training)
        return (tx,ty) , (tex,tey)

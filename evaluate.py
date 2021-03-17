import torch
import torch.nn as nn 
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error



class evaluate_result():
    def __init__(self):
        super(evaluate_result,self).__init__()


    def CRA(self, predict, label):
        cra = 0
        total_len = 0
        for i in range(len(predict)):
            total_len += len(predict[i])
            for j in range(len(predict[i])):
                cra += 1-(abs(label[i][j]-predict[i][j])/label[i][j]) 
        return cra/total_len
    


    def RMSE(self, predict, label):
        total_len = 0
        rmse = 0
        for i in range(len(predict)):
            total_len += len(predict[i])
            for j in range(len(predict[i])):
                rmse += (label[i][j]-predict[i][j])*(label[i][j]-predict[i][j])

        return math.sqrt(rmse/total_len)

    def MAE(self, predict,label):
        total_len = 0
        mae = 0
        for i in range(len(predict)):
            total_len += len(predict[i])
            for j in range(len(predict[i])):
                mae += abs(label[i][j]-predict[i][j])
        return mae/total_len



    def MAPE(self, predict, label):
        mape = 0
        total_len=0
        for i in range(len(predict)):
            total_len += len(predict[i])
            for j in range(len(predict[i])):
                mape +=abs((label[i][j]-predict[i][j])/label[i][j]) 
        return mape/total_len



    def rul_score(self, predict, label):
        score = 0
        for i in range(len(predict)):
            for j in range(len(predict[i])):
                h = predict[i][j] - label[i][j]
                if h < 0:
                    score+=(math.exp(-h/13)-1)
                else:
                    score+=(math.exp(h/10)-1)
        return score

    def CRA_smoothed(self, predict, label):
        cra = 0
        total_len = len(predict)
        for i in range(len(predict)):
            cra += 1-(abs(label[i]-predict[i])/label[i]) 
        return cra/total_len
    


    def RMSE_smoothed(self, predict, label):
        total_len = len(predict)
        rmse = 0
        for i in range(len(predict)):
            rmse += (label[i]-predict[i])*(label[i]-predict[i])

        return math.sqrt(rmse/total_len)*100

    def MAE_smoothed(self, predict,label):
        mae = 0
        total_len = len(predict)
        for i in range(len(predict)):
            mae += abs(label[i]-predict[i])
        return mae/total_len*100



    def MAPE_smoothed(self, predict, label):
        mape = 0
        total_len=len(predict)
        for i in range(len(predict)):
            mape +=abs((label[i]-predict[i])/label[i]) 
        return mape/total_len*100



    def rul_score_smoothed(self, predict, label):
        score = 0
        for i in range(len(predict)):
            h = predict[i] - label[i]
            if h < 0:
                score+=(math.exp(-h/13)-1)
            else:
                score+=(math.exp(h/10)-1)
        return score

    




from trainingModel import trainModel,trainValidation
from predictFromModel import prediction_validation,prediction 
import os
import pandas as pd
from application_logger import logging

class model:
    def __init__(self):
        self.log = logging.App_Logger()
        self.file_object = open('app.txt','a+')

    def predict(self,file='Recived_file\day.csv'):
        try:
            self.log.log(self.file_object,"-----------Prediction validation entered-------")
            p = prediction_validation(file=file)
            p.prediction_val()
            self.log.log(self.file_object,"-----------Prediction validation exited-------")
            self.log.log(self.file_object,"-----------Prediction entered-------")
            pp = prediction(file=file)
            pp.predictFromModel()
            self.log.log(self.file_object,"-----------Prediction exited-------")
            
        except Exception as e:
            self.log.log(self.file_object,f"<<<<Error occured :: {e}>>>>")
            raise Exception
    
    def train(self,file='day.csv'):
        try:
            self.log.log(self.file_object,"-----------Training started----------")
            t_val = trainValidation(file=file)
            t_val.train_validation()
            self.log.log(self.file_object,"-----------Training validation exited-------")
            t_train = trainModel()
            t_train.trainingModel()
            self.log.log(self.file_object,"-----------Training model exited-------")
        except Exception as e:
            self.log.log(self.file_object,f"Error in train :: {e}")
            raise Exception
        

model = model()
model.train()
model.predict()

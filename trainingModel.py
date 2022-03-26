"""
This is the Entry point for Training the Machine Learning Model.
Written By: Biswajit Mohapatra
Version: 1.0
Revisions: None
"""

from os import listdir
from sklearn.model_selection import train_test_split
from application_logger.logging import App_Logger
import pickle

import pandas as pd
import numpy as np
import json
import re
import os
import shutil
import calendar
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score

from math import sqrt

class trainModel(object):
    def __init__(self):
        self.log = App_Logger()
        self.file_object = open("Training_logs/trainmodel.txt",'a+')
        self.file = 'Training_Files/Good_files/day.csv'

    def trainingModel(self):
        self.log.log(self.file_object, "<<<<Training Started>>>>")
        try:
            data = pd.read_csv(self.file)
            self.log.log(self.file_object,"<<<<<<<  Doing Preprocessing  >>>>>>")

            # Preprocessing

            #removing columns 
            data = data.drop(['instant','dteday','casual','registered'],axis=1)
            self.log.log(self.file_object, "<<<< Columns removed from the data... >>>>")
            
            # Converting numerical to categorical
            self.log.log(self.file_object, "<<<<Converting numerical values to ctegorical...>>>>")
            if (data.mnth.dtypes != 'O')==True: #check for dtype of the column is object type or not
                data['mnth'] = data['mnth'].apply(lambda x: calendar.month_abbr[x])
            if (data.season.dtypes != 'O') == True:
                data.season = data.season.map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
            if (data.weathersit.dtypes != 'O') == True:
                data.weathersit = data.weathersit.map({1: 'Clear', 2: 'Mist & Cloudy',
                                               3: 'Light Snow & Rain', 4: 'Heavy Snow & Rain'})
            if (data.weekday.dtypes != 'O') == True:
                data.weekday = data.weekday.map({0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thrusday", 5: "Friday", 6: "Saturday"})

            self.log.log(self.file_object, "<<<<Converted numerical values to ctegorical...>>>>")

            # Creating Dummies
            self.log.log(self.file_object, "<<<<Creating dummy variables...>>>>")
            col = data[['season','mnth','weekday','weathersit']]
            dummy = pd.get_dummies(col,drop_first=True)
            data_new = pd.concat([dummy,data],axis=1)

            self.log.log(self.file_object, "<<<<Dummy data created successfully...>>>>")

            # Dropping columns for which dummy variables were created
            new_col = ['season', 'mnth', 'weekday','weathersit']
            self.log.log(self.file_object, "<<<<Droping columns for which dummys are created...>>>>")
            data_new = data_new.drop(new_col,axis=1)
            self.log.log(self.file_object, "<<<<Dropped columns for which dummy variables were created...>>>>")

            # create separate features and labels
            self.log.log(self.file_object, "<<<<Creating train and test datasets...>>>>")
            train,test = train_test_split(data_new,train_size=0.7, test_size=0.3, random_state=100)
            self.log.log(self.file_object, "<<<<Train Test Data Created....>>>>")

            # Apply scaler() to all the columns except the 'dummy' variables.
            self.log.log(self.file_object, "<<<<Scaling the data started...>>>>")
            num_vars = ['cnt','hum','windspeed','temp','atemp'] 
            scaler = MinMaxScaler()
            data_new[num_vars] = scaler.fit_transform(data_new[num_vars])
            self.log.log(self.file_object, "<<<<Data Scaled Successfully...>>>>")

            # Creating x_train and y_train 
            self.log.log(self.file_object, "<<<<Crating X and Y training data...>>>>")
            y_train = data_new.pop('cnt')
            x_train = data_new 
            self.log.log(self.file_object, "<<<<x_train and y_train created...>>>>")

            # Running RFE with the output number of the variable equal to 10
            self.log.log(self.file_object, "<<<<Linear Regression function initiated...>>>>")
            lm = LinearRegression()
            lm.fit(x_train,y_train)
            self.log.log(self.file_object, "<<<<LinerRegression fit done...>>>>")
            rfe = RFE(lm)
            rfe = rfe.fit(x_train,y_train)
            self.log.log(self.file_object, "<<<<<<RFE fit done...>>>>>>")
            rfe_cols = x_train.columns[~rfe.support_]
            rfe_cols = list(rfe_cols)
            x_train_rfe = x_train[rfe_cols]
            x_train_rfe = sm.add_constant(x_train_rfe)
            self.log.log(self.file_object, "<<<<Constant added...>>>>")

            # Dropping holiday column for having p>|t| value more that 0.005

            x_train_new = x_train_rfe
            # Dropping const and hum columns for having VIF score more than 5
            x_train_new = x_train_new.drop(['const'], axis=1)

            self.log.log(self.file_object, "<<<<Creating first fitted model...>>>>")
            x_train_lm = sm.add_constant(x_train_new)
            lm_first_model = sm.OLS(y_train,x_train_lm).fit()
            self.log.log(self.file_object, "<<<<Created First Model....>>>>")

            self.log.log(self.file_object, "<<<<Saving the model...>>>>")
            path = 'models/model.pkl'
            with open(path,'wb') as f:
                pickle.dump(lm_first_model,f)
                f.close()
            self.log.log(self.file_object, f"<<<<Model Saved successfully at :: {path}>>>>")
            self.file_object.close()

        except Exception as e:
            self.log.log(self.file_object, f"<<<<Error occured at trainModel :: {e}>>>>")
            self.file_object.close()
            raise Exception


class trainValidation(object):
    """

    """
    def __init__(self,file):
        self.log = App_Logger()
        self.file_object = open("Training_logs/trainingmodel.txt",'a+')
        self.file = file
        self.schema = 'schema_training.json'

    def train_validation(self):
        try:
            self.log.log(self.file_object,"<<<<<< Train Validation Started >>>>>>")
            with open(self.schema, 'r') as f:
                dic = json.load(f)
                f.close()
            pattern = dic["SampleFileName"]
            NumberOfColumns = dic["NumberOfColumns"]
            column_names = dic['ColName']

            self.log.log(self.file_object,"<<<<Schema File loaded and data extracted...>>>>")

            # Regex cration pattern
            self.log.log(self.file_object,"<<<<Creating manual regex...>>>>")
            regex = "['day']+\.csv"
            self.log.log(self.file_object, "<<<<Created manual regex...>>>>")

            # validating filename of training files
            try:
                self.log.log(self.file_object, "<<<<<< Filename Validation Started...>>>>>>")
                if (re.match(regex,self.file)):
                    if self.file.endswith(".csv"):
                        l = self.file.split(".csv")
                        filename = l[0]
                        if filename == 'day' or filename == 'hour':
                            shutil.move(self.file,f'Training_Files/Good_files')
                        else:
                            shutil.move(self.file,f'Training_Files/Bad_files')
                    else:
                        shutil.move(self.file,f'Training_Files/Bad_files')
                else:
                    shutil.move(self.file,f'Training_Files/Bad_files')
                self.log.log(self.file_object, "<<<<<< Filename Validation Completed...>>>>>>")
            except Exception as e:
                self.log.log(self.file_object,f"Error occured while validating... :: {e}")

            # Column validation
            try:
                self.log.log(self.file_object, "<<<<<<Column validation started...>>>>>>")
                for file in listdir('Training_Files/Good_files'):
                    csv = pd.read_csv(f"Training_Files/Good_files/{file}")
                    if csv.shape[1] == NumberOfColumns:
                        pass
                    else:
                        shutil.move(file,'Training_Files/Bad_files')

                self.log.log(self.file_object, "<<<<<<Column validation finished...>>>>>>")
            except Exception as e:
                self.log.log(self.file_object, f"<<<<<<Error occured while column validation :: {e}>>>>>>")
            
            # Check MIsiing Values
            try:
                self.log.log(self.file_object, "<<<<<<Missing value check started...>>>>>>")
                for file in listdir('Training_Files/Good_files'):
                    csv = pd.read_csv(f"Training_Files/Good_files/{file}")
                    count = 0
                    for columns in csv:
                        if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                            count+=1
                            shutil.move(f"Training_Files/Good_files/{file}",'Training_Files/Bad_files')
                            self.log.log(self.file_object,f"Bad files moved :: {file}")
                            break
            except Exception as e:
                self.log.log(self.file_object,f"Error while checking missing values :: {e}")
            # Replace missing value with null
            try:
                self.log.log(self.file_object,"Missing value replacing with NULL")
                files = [f for f in listdir('Training_Files/Good_files')]
                for file in files:
                    csv = pd.read_csv(f"Training_Files/Good_files/{file}")
                    csv.fillna("NULL", inplace=True)
                    csv.to_csv(f"Training_Files/Good_files/{file}", index=None,header=True)
                    self.log.log(self.file_object, "<<<<Missing value replaced...>>>>")
            except Exception as e:
                self.log.log(self.file_object, "<<<<Error occured in replacing missing values... :: {e}>>>>")
                raise Exception

        except Exception as e:
            self.log.log(self.file_object, "<<<<Error while validating :: {e}>>>>")
            

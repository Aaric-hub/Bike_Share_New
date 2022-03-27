import pandas as pd
import statsmodels.api as sm
from application_logger import logging
import pickle
import shutil
import calendar
import re
import os
import numpy as np
import json
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class prediction(object):
    def __init__(self,file):
        self.log = logging.App_Logger()
        self.file_object = open("Prediction_Logs/prediction.txt","a+")
        self.file = f"Recived_file/{file}"
        self.model = pickle.load(open('models/model.pkl','rb'))

    def predictFromModel(self):
        try:
            self.log.log(self.file_object,"<<<<<<Start of Prediction>>>>>>>")
            data = pd.read_csv(self.file)
            self.log.log(self.file_object,"<<<<<<Data Recived>>>>>>")

            # Doing Preprocessing 
            
            try:
                self.log.log(self.file_object,"<<<<<<Preprocessing Started>>>>>>>")
                columns = ['instant', 'dteday', 'casual', 'registered']
                data_new = data.drop(labels=columns,axis=1)
                self.log.log(self.file_object,"<<<<<<Columns Dropped>>>>>>")
            except Exception as e:
                self.log.log(self.file_object,f"<<<<<<Error occured :: {e}>>>>>>")
                raise Exception

            # Converting numerical values to ctegorical

            try:
                self.log.log(self.file_object,"<<<<<<Converting numerical values to ctegorical>>>>>>")
                if (data_new.mnth.dtypes != 'O') == True:
                    data_new['mnth'] = data_new['mnth'].apply(lambda x: calendar.month_abbr[x])
                
                if (data_new.season.dtypes != 'O') == True:
                    data_new.season = data_new.season.map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
                if (data_new.weathersit.dtypes != 'O') == True:
                    data_new.weathersit = data_new.weathersit.map({1: 'Clear', 2: 'Mist & Cloudy',
                                               3: 'Light Snow & Rain', 4: 'Heavy Snow & Rain'})
                if (data_new.weekday.dtypes != 'O') == True:
                    data_new.weekday = data_new.weekday.map({0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thrusday", 5: "Friday", 6: "Saturday"})
                
                self.log.log(self.file_object,f"<<<<<<Converting numerical values to ctegorical completed>>>>>>")

            except Exception as e:
                self.log.log(self.file_object, "<<<<Error occured while converting numerical values to ctegorical>>>>")

            # Create Dummies  
            try:
                self.log.log(self.file_object, "<<<<<<<<<<Creating dummy>>>>>>>>>>")
                col = ['season','mnth','weekday','weathersit']
                d = data_new[col]
                dummy = pd.get_dummies(d,drop_first=True)
                data_new_dummy = pd.concat([dummy,data_new],axis=1)
                self.log.log(self.file_object, "<<<<Dummy data created...>>>>")
                data_new_1 = data_new_dummy.drop(col,axis=1)
                self.log.log(self.file_object, "<<<<Columns removed for which dummy were created...>>>>")
            except Exception as e:
                self.log.log(self.file_object,f"Error occured while creating dummy data..:: {e}")
                raise Exception

            # Creating train test
            try:
                self.log.log(self.file_object,"<<<<<<Creating train and test data>>>>>>")
                train,test = train_test_split(data_new_1,train_size=0.7, test_size=0.3,random_state=100)
                self.log.log(self.file_object,"<<<<Train and test data created successfully>>>>")
            except Exception as e:
                self.log.log(self.file_object,f"Error occured while creating train and test data :: {e}")
                raise Exception
            
            # Scaling the test data
            try:
                self.log.log(self.file_object,"<<<<<<Scaling of the data started...>>>>>>")
                num_vars = ['cnt','hum','windspeed','temp','atemp']
                scaler = MinMaxScaler()
                test[num_vars] = scaler.fit_transform(test[num_vars])
                self.log.log(self.file_object,"<<<<Sclaing done Successfully...>>>>")
            except Exception as e:
                self.log.log(self.file_object,f"Error occured while scaling :: {e}")
                raise Exception
            
            # creating x and y datasets
            try:
                self.log.log(self.file_object,"<<<<Creating x and y datasets...>>>>")
                y_test = test.pop('cnt')
                x_test = test
                self.log.log(self.file_object,"<<<<x and y datasets created successfully...>>>>")
            except Exception as e:
                self.log.log(self.file_object,f"Error occured while creating x and y datasets...>>>>")
                raise Exception

            # loading the model
            x_test = x_test[['season_Summer', 'mnth_Aug', 'mnth_Jul', 'mnth_Jun', 'mnth_Mar',
                            'mnth_May', 'mnth_Oct', 'weekday_Monday', 'weekday_Saturday',
                            'weekday_Sunday', 'weekday_Thrusday', 'weekday_Tuesday',
                            'weekday_Wednesday', 'workingday', 'atemp']]
            model = self.model
            self.log.log(self.file_object,"<<<<<Model loaded successfully...>>>>")
            x_test = sm.add_constant(x_test)
            test_cols = x_test.columns
            #x_test = x_test[test_cols[1:]]
            #print(x_test.columns)
            result = list(model.predict(x_test))
            result_ = pd.DataFrame(list(zip(result)),columns=['Predictions'])
            path = 'Prediction_output/Predictions.csv'
            result_.to_csv(path,header=True,mode='a+')
            self.log.log(self.file_object,f"<<<<<< Prediction file saved at {path}>>>>>>>")

        except Exception as e:
            self.log.log(self.file_object,f"Error while doing prediction :: {e}")
            raise Exception

class prediction_validation(object):
    def __init__(self,file):
        self.file = file
        self.log = logging.App_Logger()
        self.file_object = open("Prediction_Logs/prediction_validation.txt","a+")
        self.schema = 'schema_prediction.json'

    def prediction_val(self):
        try:
            self.log.log(self.file_object,"<<<<<<Prediction validation started...>>>>>>")
            with open(self.schema,'r') as f:
                dic = json.load(f)
                f.close()
            
            pattern = dic['SampleFileName']
            no_of_col = dic['NumberOfColumns']
            col_names = dic['ColName']
            self.log.log(self.file_object,"Data from schema loaded")

            # Validating Columns
            try:
                self.log.log(self.file_object,"validating column names")
                for file in os.listdir('Recived_file'):
                    if file != 'Bad_file':
                        csv = pd.read_csv(f'Recived_file/{file}')
                        if csv.shape[1] == no_of_col:
                            pass
                        else:
                            self.log.log(self.file_object,f"{file} moved due to invalid number of columns to Recived_file/Bad_file")
                            shutil.move(file,'Recived_file/Bad_file')
                self.log.log(self.file_object,"validating file names completed successfully")
            except Exception as e:
                self.log.log(self.file_object,f"Validating file names failed :: {e}")
                raise Exception

            # validating if any column has all values missing
            try:
                self.log.log(self.file_object,"validating if any column has all values missing")
                for file in os.listdir('Recived_file'):
                    if file != 'Bad_file':
                        csv = pd.read_csv(f'Recived_file/{file}')
                        count = 0
                        for columns in csv:
                            if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                                count+=1 
                                shutil.move(file,'Recived_file/Bad_file')
                                self.log.log(self.file_object,f"{file} moved to Bad_file, missing all values in a column...")
                self.log.log(self.file_object,"Missing values in column validation completed successfully..")
            except Exception as e:
                self.log.log(self.file_object,f"Failed to validate column's missing values:: {e}")
                raise Exception
            
            self.log.log(self.file_object,"prediction validation complted successfully")

        except Exception as e:
            self.log.log(self.file_object,f"Failed to perform file validation:: {e}")
            raise Exception
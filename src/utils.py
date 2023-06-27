
import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as f:
            dill.dump(obj,f)
    except Exception as e:

        raise CustomException(e,sys)
    
def load_object(file_path):
    try:

        with open(file_path,'rb') as f:
            return dill.load(f)

    except Exception as e:

        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    
    report = {}
    try:
        for model_name,model in models.items():

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            

            train_model_score =round(r2_score(y_train,y_train_pred),2)
            test_model_score =round(r2_score(y_test,y_test_pred),2)
            report[model_name] = test_model_score 

        return report
    
    except Exception as e:

        raise CustomException(e,sys)
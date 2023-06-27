
import os
import sys

from dataclasses import dataclass

from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:

    train_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_arr,test_arr):

        try:
            
            logging.info('Splitting train and test data')
            X_train,y_train,X_test,y_test = train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]

            models = {
                "Linear Regression": LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                "AdaBoost Regression": AdaBoostRegressor(),
                "Random Forest": RandomForestRegressor(),
                "CatBoost Regression": CatBoostRegressor(),
                "XGBoost Regression":  XGBRegressor(),
                'GradientBossting Regression': GradientBoostingRegressor()
            }

            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]


            if best_model_score <0.6:
                raise CustomException('No best model found')
            
            logging.info(f'Best found model on both training and testing dataset is {best_model_name}')
            
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square 

        except Exception as e:
            raise CustomException(e,sys)

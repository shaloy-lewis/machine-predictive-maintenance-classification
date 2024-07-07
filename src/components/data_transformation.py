import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.utils.utils import save_object

from src.utils.constants import (CAT_FEATURES,
                                 NUM_FEATURES)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation(self):
        try:
            logging.info('Data transformation initiated')
            logging.info('Pipeline Initiated')
            
            numeric_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer()),
                    ("scaler",StandardScaler())
                ]
            )
            
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder",OneHotEncoder(sparse_output=False,drop='if_binary'))
                ]
            )
            
            preprocessor=ColumnTransformer(
                [
                    ("numeric_pipeline",numeric_pipeline,NUM_FEATURES),
                    ("catategorical_pipeline",categorical_pipeline,CAT_FEATURES),
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured in get_data_transformation")
            raise customexception(e,sys)
    
    def initiate_data_transformation(self,X_train_path,X_val_path,X_test_path,
                                     y_train_path,y_val_path,y_test_path,
                                     y_train_enc_path,y_val_enc_path,y_test_enc_path):
        try:
            X_train=pd.read_csv(X_train_path)
            X_val=pd.read_csv(X_val_path)
            X_test=pd.read_csv(X_test_path)
            y_train=pd.read_csv(y_train_path)
            y_val=pd.read_csv(y_val_path)
            y_test=pd.read_csv(y_test_path)
            y_train_enc=pd.read_csv(y_train_enc_path)
            y_val_enc=pd.read_csv(y_val_enc_path)
            y_test_enc=pd.read_csv(y_test_enc_path)
            
            logging.info("Read train and test data for data transformation")
            logging.info(f'Train Dataframe shape : \n{X_train.shape}')
            logging.info(f'Validation Dataframe shape : \n{X_val.shape}')
            logging.info(f'Test Dataframe shape : \n{X_test.shape}')
            
            preprocessor = self.get_data_transformation()
            
            logging.info('Applying preprocessing object on training and testing datasets')
            X_train=pd.DataFrame(preprocessor.fit_transform(X_train),columns=preprocessor.get_feature_names_out())
            X_val=pd.DataFrame(preprocessor.fit_transform(X_val),columns=preprocessor.get_feature_names_out())
            X_test=pd.DataFrame(preprocessor.transform(X_test),columns=preprocessor.get_feature_names_out())
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Preprocessing file saved in pickle format")
            
            return (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                y_train_enc,
                y_val_enc,
                y_test_enc
            )
            
        except Exception as e:
            logging.info("Exception occured in initiate_data_transformation")
            raise customexception(e,sys)

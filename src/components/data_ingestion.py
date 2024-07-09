import pandas as pd
import os
import sys

from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.constants import (TARGET, CAT_FEATURES, NUM_FEATURES,
                                 VAL_SIZE, TEST_SIZE)

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts","X_train.csv")
    val_data_path:str=os.path.join("artifacts","X_val.csv")
    test_data_path:str=os.path.join("artifacts","X_test.csv")
    train_target_path:str=os.path.join("artifacts","y_train.csv")
    val_target_path:str=os.path.join("artifacts","y_val.csv")
    test_target_path:str=os.path.join("artifacts","y_test.csv")
    train_enc_path:str=os.path.join("artifacts","y_train_enc.csv")
    val_enc_path:str=os.path.join("artifacts","y_val_enc.csv")
    test_enc_path:str=os.path.join("artifacts","y_test_enc.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("initiating data ingestion")
        try:
            df=pd.read_csv(os.path.join('data','predictive_maintenance.csv'),index_col='UDI')
            df.drop('Product ID', axis=1, inplace=True)
            logging.info("data read successfully")
            
            logging.info("raw dataset ingested successfully")
            
            logging.info("initiating train test split")
            X=df[CAT_FEATURES+NUM_FEATURES]
            y=df[TARGET]
            
            X_train,X_val,y_train,y_val=train_test_split(X
                                               , y
                                               , test_size=VAL_SIZE
                                               , stratify=y
                                               , shuffle=True
                                               , random_state=42)
            X_train,X_test,y_train,y_test=train_test_split(X_train
                                               , y_train
                                               , test_size=TEST_SIZE
                                               , stratify=y_train
                                               , shuffle=True
                                               , random_state=42)
            
            logging.info("train test split completed")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            X_train.to_csv(self.ingestion_config.train_data_path,index=False)
            X_val.to_csv(self.ingestion_config.val_data_path,index=False)
            X_test.to_csv(self.ingestion_config.test_data_path,index=False)
            y_train.to_csv(self.ingestion_config.train_target_path,index=False)
            y_val.to_csv(self.ingestion_config.val_target_path,index=False)
            y_test.to_csv(self.ingestion_config.test_target_path,index=False)
            
            logging.info("Encoding multi-class target variable")
            ohe=OneHotEncoder(sparse_output=False)

            y_train_enc = pd.DataFrame(ohe.fit_transform(y_train[['Failure Type']]), columns=ohe.get_feature_names_out())
            y_val_enc = pd.DataFrame(ohe.transform(y_val[['Failure Type']]), columns=ohe.get_feature_names_out())
            y_test_enc = pd.DataFrame(ohe.transform(y_test[['Failure Type']]), columns=ohe.get_feature_names_out())
            
            y_train_enc.to_csv(self.ingestion_config.train_enc_path,index=False)
            y_val_enc.to_csv(self.ingestion_config.val_enc_path,index=False)
            y_test_enc.to_csv(self.ingestion_config.test_enc_path,index=False)
            
            logging.info("data ingestion completed")
            
            return (             
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_target_path,
                self.ingestion_config.val_target_path,
                self.ingestion_config.test_target_path,
                self.ingestion_config.train_enc_path,
                self.ingestion_config.val_enc_path,
                self.ingestion_config.test_enc_path
            )

        except Exception as e:
            logging.info('Exception occured in data_ingestion')
            raise customexception(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
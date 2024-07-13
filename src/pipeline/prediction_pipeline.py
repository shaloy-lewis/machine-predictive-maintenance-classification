import os
import sys
import pandas as pd
from src.exception.exception import customexception
from src.logger.logging import logging
from src.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        logging.info("Initializing the prediction pipeline")
        self.preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
        self.model_path=os.path.join("artifacts","model.pkl")

        self.preprocessor=load_object(self.preprocessor_path)
        self.model=load_object(self.model_path)

    def predict(self,features):
        try:
            scaled_features=self.preprocessor.transform(features)
            pred=self.model.predict(scaled_features)
            logging.info('Predictions obtained successfully')

            return pred

        except Exception as e:
            raise customexception(e,sys)

class CustomData:
    def __init__(self,
                 air_temperature_k: float,
                 process_temperature_k: float,
                 rotational_speed_rpm: float,
                 torque_nm: float,
                 tool_wear_min: float,
                 Type: str):
        self.air_temperature_k = air_temperature_k
        self.process_temperature_k = process_temperature_k
        self.rotational_speed_rpm = rotational_speed_rpm
        self.torque_nm = torque_nm
        self.tool_wear_min = tool_wear_min
        self.Type = Type 

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Air temperature [K]': [self.air_temperature_k],
                'Process temperature [K]': [self.process_temperature_k],
                'Rotational speed [rpm]': [self.rotational_speed_rpm],
                'Torque [Nm]': [self.torque_nm],
                'Tool wear [min]': [self.tool_wear_min],
                'Type': [self.Type]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Data received for inference prediction')
            return df
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise customexception(e,sys)
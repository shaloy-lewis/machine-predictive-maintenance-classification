from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from src.utils.utils import save_object
from src.utils.constants import (LEARNING_RATE,PATIENCE,MIN_DELTA,
                                 START_FROM_EPOCH,EPOCS,BATCH_SIZE)

import keras
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,X_train,X_val,y_train,y_val,y_train_enc,y_val_enc):
        try:
            logging.info("Initiating model training")
            inp_dim=X_train.shape[1]
            logging.debug(f"Input dimension: {inp_dim}")
            
            logging.debug("Building model architecture.")
            inputs = Input(shape=(inp_dim,))
            x = inputs
            
            for i in range(10,2,-1):
                x = Dense(2**i, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(0.5)(x)
            
            failure_prob=Dense(1, activation='sigmoid', name='failure_prob')(x)
            failure_type=Dense(6, activation='softmax', name='failure_type')(x)

            model=Model(inputs=inputs,outputs=[failure_prob, failure_type])

            logging.debug(model.summary())
            
            opt=Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt,
                        loss={'failure_prob': 'binary_crossentropy', 'failure_type': 'categorical_crossentropy'},
                        metrics={'failure_prob': ['recall','precision','AUC'], 'failure_type': ['recall','precision','AUC']}
                        )
            es=EarlyStopping(monitor='val_loss',
                            patience=PATIENCE,
                            min_delta=MIN_DELTA,
                            mode='min',
                            start_from_epoch=START_FROM_EPOCH,
                            verbose=0,
                            restore_best_weights=False)
            
            logging.info("Model training in progress")
            history = model.fit(X_train,
                    {'failure_prob': y_train['Target'],
                     'failure_type': y_train_enc},
                    validation_data=(X_val,
                    {'failure_prob': y_val['Target'],
                     'failure_type': y_val_enc}),
                    epochs=EPOCS,
                    batch_size=BATCH_SIZE,
                    callbacks=[es])
                                    
            logging.info("Model training done")
            
            logging.info("Saving model")
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=model
            )
            logging.info("Model saved successfully")

        except Exception as e:
            logging.info('Exception occured in model training')
            raise customexception(e,sys)
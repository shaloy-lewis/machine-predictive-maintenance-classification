import os
import sys
import numpy as np
import pandas as pd
from src.utils.utils import load_object
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    top_k_accuracy_score, cohen_kappa_score, matthews_corrcoef
)
from src.logger.logging import logging
from src.exception.exception import customexception

class ModelEvaluation:
    def __init__(self):
        logging.info("Model evaluation initiated")
        
    def evaluate_model(self,true,pred,true_enc):
        ## failure probability
        roc_auc = roc_auc_score(true['Target'].values, pred[0])
        recall = recall_score(true['Target'].values, np.round_(pred[0]))
        precision = precision_score(true['Target'].values, np.round_(pred[0]))
        f1 = f1_score(true['Target'].values, np.round_(pred[0]))

        ## failute type
        top2_acc = top_k_accuracy_score(true['Failure Type'].values, pred[1], k=2)
        top3_acc = top_k_accuracy_score(true['Failure Type'].values, pred[1], k=3)
        ck_score = cohen_kappa_score(np.argmax(true_enc,axis=1), pd.DataFrame(pred[1]).idxmax(axis=1))
        mcc = matthews_corrcoef(np.argmax(true_enc,axis=1), pd.DataFrame(pred[1]).idxmax(axis=1))
        
        return {
            'failure_prob':{
                'roc_auc':roc_auc,
                'recall':recall,
                'precision':precision,
                'f1_score':f1
            },
            'failure_type':{
                'top2_accuracy':top2_acc,
                'top3_accuracy':top3_acc,
                'Cohen-Kappa_score':ck_score,
                'MCC_score':mcc 
            }
        }

    def initiate_model_evaluation(self,X_train,X_val,X_test,y_train,y_val,y_test,y_train_enc,y_val_enc,y_test_enc):
        try:
            logging.info("Model loading in progress")
            model_path=os.path.join("artifacts","model.pkl")
            model=load_object(model_path)

            logging.info("Calculating model predictions")
            y_train_pred=model.predict(X_train)
            y_val_pred=model.predict(X_val)
            y_pred=model.predict(X_test)

            logging.info("Evaluating model")
            train_performance = self.evaluate_model(y_train, y_train_pred, y_train_enc)
            val_performance = self.evaluate_model(y_val, y_val_pred, y_val_enc)
            test_performance = self.evaluate_model(y_test, y_pred, y_test_enc)
            
            logging.info("Model performance on training set:")
            logging.info(train_performance)
            logging.info("Model performance on validation set:")
            logging.info(val_performance)
            logging.info("Model performance on test set:")
            logging.info(test_performance)
            
        except Exception as e:
            logging.info("Exception occured in model evaluation")
            raise customexception(e,sys)
    
    
            
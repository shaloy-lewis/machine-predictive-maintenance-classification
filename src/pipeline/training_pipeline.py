from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

data_ingestion=DataIngestion()
(train_data_path,val_data_path,test_data_path,
 train_target_path,val_target_path,test_target_path,
 train_enc_path,val_enc_path,test_enc_path)=data_ingestion.initiate_data_ingestion()

data_transformation=DataTransformation()
(X_train,X_val,X_test,
 y_train,y_val,y_test,
 y_train_enc,y_val_enc,y_test_enc)=data_transformation.initiate_data_transformation(train_data_path,val_data_path,test_data_path,
                                                                                    train_target_path,val_target_path,test_target_path,
                                                                                    train_enc_path,val_enc_path,test_enc_path)

model_trainer=ModelTrainer()
model_trainer.initate_model_training(X_train,X_val,
                                     y_train,y_val,
                                     y_train_enc,y_val_enc)

model_eval=ModelEvaluation()
model_eval.initiate_model_evaluation(X_train,X_val,X_test,
                                     y_train,y_val,y_test,
                                     y_train_enc,y_val_enc,y_test_enc)
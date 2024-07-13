from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData
from src.logger.logging import logging

app = FastAPI(
    title='Machine Predictive Maintenance Classification',
    description="This model Predicts the tool's probability of failure and classifies the type of failure using the tool environment data"
)

@app.get("/")
async def root():
    return {"message": "Machine failure prediction api"}

@app.get("/ping", summary='Health check')
def health_check():
    return {"message": "Health check successful!"}

class MachineData(BaseModel):
    air_temperature_k: float = 298.9
    process_temperature_k: float = 309.1
    rotational_speed_rpm: int = 2861
    torque_nm: float = 4.6
    tool_wear_min: int = 143
    Type: str = 'L'
    
@app.post("/predict")
def predict_failure(data: MachineData):
    try:
        custom_data = CustomData(
            air_temperature_k=data.air_temperature_k,
            process_temperature_k=data.process_temperature_k,
            rotational_speed_rpm=data.rotational_speed_rpm,
            torque_nm=data.torque_nm,
            tool_wear_min=data.tool_wear_min,
            Type=data.Type 
        )

        # Convert to DataFrame
        features_df = custom_data.get_data_as_dataframe()
        logging.info("Converted to DF successfully")

        # Initialize the prediction pipeline and make predictions
        pipeline = PredictPipeline()
        failure_prob,failure_type = pipeline.predict(features_df)
        logging.info("Predictions obtained successfully")
        
        # Return the prediction result
        output= {
            "probability_of_failure": round(failure_prob[0].tolist()[0],6),
            "failure_type": {
                "Heat Dissipation Failure":{"probability":round(failure_type[0].tolist()[0],6)}
                , "Overstrain Failure":{"probability":round(failure_type[0].tolist()[2],6)}
                , "Power Failure":{"probability":round(failure_type[0].tolist()[3],6)}
                , "Random Failures":{"probability":round(failure_type[0].tolist()[4],6)}
                , "Tool Wear Failure":{"probability":round(failure_type[0].tolist()[5],6)}
                }
        }
        
        return output
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Failure ,Failure Type_Random Failures,Failure Type_Tool Wear Failure
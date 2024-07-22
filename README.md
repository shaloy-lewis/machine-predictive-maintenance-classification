<h1 align="center"> Machine Predictive Maintenance Classification</h1>

![--](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Overview
- This repository hosts an **Artificial Neural Network** model, served via **FastAPI**, that predicts whether a tool is on the verge of failure or not based on it's working environment. It also predicts the most probable type of failue the tool may experience using the same information.
- The app is deployed on streamlit. Try it out <a href="https://machine-predictive-maintenance-classification.streamlit.app/"> here </a>
- Dataset obtained from <a href="https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification"> Kaggle </a>

![--](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### A. Run with Docker
1. Clone the repository
```
git clone https://github.com/shaloy-lewis/machine-predictive-maintenance-classification.git
cd machine-predictive-maintenance-classification
```
2. Build and run the Docker container
```
docker-compose build
docker-compose up
```
3. Access the application
```
http://localhost:8080
```

### B. Run Locally Without Docker
1. Clone the repository
```
git clone https://github.com/shaloy-lewis/machine-predictive-maintenance-classification.git
cd machine-predictive-maintenance-classification
```
2. Create and activate virtualenv
```
pip install virtualenv
python3.12 -m venv venv
```
*For windows*
```
venv/Scripts/activate.bat
```
*For linux*
```
source venv/bin/activate
```
3. Install all the required packages and dependencies
```
pip install -r requirements.txt
```
5. Run the server
```
uvicorn api:app --reload --port 8080 --host 0.0.0.0
```
6. Access the application
```
http://localhost:8080
```
![--](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Getting Predictions
```
curl -X 'POST' \
  'http://localhost:[hostname]/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "air_temperature_k": 298.9,
  "process_temperature_k": 309.1,
  "rotational_speed_rpm": 2861,
  "torque_nm": 4.6,
  "tool_wear_min": 143,
  "Type": "L"
}'
```
Change the hostname with the hostname given on your environment

![--](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Sample Response
```
{
    "probability_of_failure": 0.999088,
    "failure_type": {
        "Heat Dissipation Failure": {
            "probability": 0.235062
        },
        "Overstrain Failure": {
            "probability": 0.143143
        },
        "Power Failure": {
            "probability": 0.618992
        },
        "Random Failures": {
            "probability": 0.0009
        },
        "Tool Wear Failure": {
            "probability": 0.001512
        }
    }
}
```

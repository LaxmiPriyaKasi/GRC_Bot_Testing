from fastapi import FastAPI, HTTPException
import os
import requests
import numpy as np
import pandas as pd
import json
from typing import List, Dict
from pydantic import BaseModel

app = FastAPI(
    title="Model Inference API",
    description="API for model inferencing using Databricks serving endpoint",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Welcome to the GRC Chatbot API!"}

class DataFrameSplit(BaseModel):
    columns: List[str]
    data: List[List[str]]


class ModelInput(BaseModel):
    dataframe_split: DataFrameSplit


def create_tf_serving_json(data):
    return {
        'inputs': {name: data[name].tolist() for name in data.keys()}
        if isinstance(data, dict)
        else data.tolist()
    }


@app.post("/predict")
async def score_model(input_data: ModelInput):
    url = 'https://dbc-08667beb-0669.cloud.databricks.com/serving-endpoints/chatbot_testing/invocations'
    headers = {
        'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
        'Content-Type': 'application/json'
    }

    # Convert input data to dictionary format
    data_dict = input_data.dict()

    # Convert to DataFrame if needed
    df = pd.DataFrame(
        data_dict['dataframe_split']['data'],
        columns=data_dict['dataframe_split']['columns']
    )

    # Prepare data for inference
    ds_dict = (
        {'dataframe_split': df.to_dict(orient='split')}
        if isinstance(df, pd.DataFrame)
        else create_tf_serving_json(df)
    )

    try:
        data_json = json.dumps(ds_dict, allow_nan=True)
        response = requests.request(
            method='POST',
            headers=headers,
            url=url,
            data=data_json
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f'Request failed with status {response.status_code}, {response.text}'
            )

        return response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Example usage:
"""
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "dataframe_split": {
        "columns": ["query"],
        "data": [["What are the key requirements for maintaining quality management documentation?"]]
    }
}'
"""
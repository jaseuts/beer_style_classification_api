from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import torch
from src.models.pytorch import PytorchDataset 
from src.models.pytorch import test_classification

app = FastAPI()
scale = load('../models/scaler.joblib')
ord_enc_brew = load('../models/ord_encoder_brew.joblib')
ord_enc_type = load('../models/ord_encoder_type.joblib')
model = torch.load('../models/pytorch_beer_model.pt')


@app.get("/")
def read_root():
    return {"Description": '''This project uses a Neural Network built with
                              Pytorch architechture to predict a beer type
                              based on the data input by users''',
            "Endpoints": ["'/'(GET)", "'/health/(GET)", "/beer/type/'(GET)",
                          "'/beers/type'(GET)"],
            "Expected input": ['brewery name', 'ovreall rating: from 0.0 to 5.0', 
                               'aroma rating: from 1.0 to to 5.0', 
                               'appearance rating: from 0.0 to 5.0', 
                               'palate rating: from 1.0 to 5.0', 
                               'taste rating: from 1.0 to 5.0', 
                               'alcohol by volume: from 0.0 to 58.0"],
            "Expected output": 'beer type'
           }
    
   
@app.get('/health', status_code=200)
def healthcheck():
    return 'Pytorch Neural Network is ready predict a beer type based on your input data!'
    
    
def format_features(brewery: str, overall: float, aroma: float, appearance: float,
                    palate: float, taste: float, abv: float):
    return {
        'brewery': [brewery],
        'overall': [overall],
        'aroma': [aroma],
        'appearance': [appearance],
        'palate': [palate],
        'taste': [taste],
        'abv': [abv]
    }
    

def text_to_float(text: pd.Series):
    return ord_enc_brew.transform(text)
    
'''
def scale_features(features: pd.DataFrame):
    features = scale.transform(features)
    return features
'''

def numeric_to_text(number: pd.Series):
    return ord_enc_type.inverse_transform(number)


@app.get("/beer/type")
def predict(brewery: str, overall: float, aroma: float, appearance: float,
            palate: float, taste: float, abv: float):
    features = format_features(brewery, overall, aroma, appearance,
                    palate, taste, abv)                   
    df = pd.DataFrame(features)
    df['brewery'] = ord_enc_brew.transform(df['brewery'])
    df = scale.transform(df)
    data = torch.Tensor(df)
    pred = model(data).argmax(1)
    pred = ord_enc_type.inverse_transform(output.reshape(-1, 1))
    return JSONResponse(pred.tolist())
    
    
    
    
    
    
    
'''    
    df_copy = df.copy()
    df_copy['brewery'] = ord_encoder(df_copy['brewery'])
    cols = ['overall', 'aroma', appearance, palate, taste, abv
    pred = gmm_pipe.predict(obs)
    return JSONResponse(pred.tolist())
'''  
    
    
    
    

# # import fastapi
# # model creation for fastapi
# #  load pickle file
# # making prediction endpoint

# from fastapi import FastAPI
# import pickle
# import pydantic import BaseModel
# from typing import Literal
# import pandas as pd
# from fastapi.responses import JSONResponse

# app = FastAPI(title='this is first ML api project') 

# # validation class
# class PricePredict(BaseModel) 





# with open('model.pkl','rb') as fs:
#     model=pickle.load(fs)

# @app.get('/')
# def greet():
#     return {'this is home page'}  

# # predict post endpoint

# @app.post('/'predict)
# def predict(data:PricePreict) 
#`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````` 
#import fastapi
#model creation for fastapi
#loading pickle file
#making predict endpoint
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import pickle
from fastapi.responses import JSONResponse

app = FastAPI(title="Car Price Prediction API")

# Load model ONCE
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class PricePredict(BaseModel):
    year: int
    km_driven: int
    fuel: Literal[0, 1, 2, 3]
    seller_type: Literal[0, 1, 2]
    transmission: Literal[0, 1]
    owner: Literal[0, 1, 2, 3, 4]
    engine: float
    seats: float

@app.get("/")
def home():
    return {"message": "Go to /docs for prediction"}

@app.post("/predict")
def predict(data: PricePredict):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return JSONResponse(content={"prediction": float(prediction)})

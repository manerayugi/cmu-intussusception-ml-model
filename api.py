import pandas as pd
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel
import json

import joblib
# # Load the model
# model = joblib.load('./models/dt_kmsmote.joblib')

data_fetures = ['Type of intussusception_appendicocolic', 'Type of intussusception_colocolic',
       'Type of intussusception_ileocolic', 'Type of intussusception_ileoileal',
       'Type of intussusception_ileoileocolic', 
       'Gender_Female', 'Gender_Male',
       'Vomiting_absent', 'Vomiting_present',
       'Bloody stool/Rectal bleeding_absent', 'Bloody stool/Rectal bleeding_present', 
       'Abdominal distension_absent', 'Abdominal distension_present', 
       'Palpable abdominal mass_absent', 'Palpable abdominal mass_present', 
       'Location of intussusception_LLQ',
       'Location of intussusception_LUQ', 'Location of intussusception_RLQ',
       'Location of intussusception_RUQ', 'Location of intussusception_rectum',
       'Method of reduction_Hydrostatic reduction', 'Method of reduction_Pneumatic reduction',
       'Small bowel obstruction from Plain film Abdomen _No', 'Small bowel obstruction from Plain film Abdomen _Yes',
       'a thick peripheral hypoechoic rim_absent', 'a thick peripheral hypoechoic rim_present',
       'free intraperitoneum fluid_absent', 'free intraperitoneum fluid_present',
       'fluid trapped within intussusceptum_absent', 'fluid trapped within intussusceptum_present',
       'enlarged lymph node in intussusception_absent', 'enlarged lymph node in intussusception_present',
       'pathologic lead point_absent', 'pathologic lead point_present',
       'absence of blood flow in the intussusception_absent', 'absence of blood flow in the intussusception_present', 
       'Age', 'Body weight', 'Duration of symptoms', 'Body Temperature']
cat_cols = ['Type of intussusception','Gender','Vomiting', 'Bloody stool/Rectal bleeding', 'Abdominal distension','Palpable abdominal mass','Location of intussusception','Method of reduction','Small bowel obstruction from Plain film Abdomen ',
       'a thick peripheral hypoechoic rim', 'free intraperitoneum fluid',
       'fluid trapped within intussusceptum',
       'enlarged lymph node in intussusception', 'pathologic lead point',
       'absence of blood flow in the intussusception']
num_cols = ['Age', 'Body weight', 'Duration of symptoms', 'Body Temperature']

def convertinput2data(data_input_dict):
    # set empty data
    data = {}
    data = {key: 0 for key in data_fetures}
    data.update({key: data_input_dict[key] for key in data if key in data_input_dict})
    #merge data
    # data.update(data_input_dict)
    print('data', data)
    
    for feature_in in cat_cols:
        _feature_name = f"{feature_in}_{data_input_dict[feature_in]}"
        data[_feature_name] = 1
        
    return(data)

def trasform_data_dict2df(data):
    if type(data) == dict:
        x = pd.DataFrame(data, index=[0])
    else:
        x = pd.DataFrame(pd.Series(data)).T
    
    return x

# the type of input data
class intussusception_input_data(BaseModel):
    data: dict
    
description = """ """

app = FastAPI(title='ML Intussusception API by MEE', description=description, version='1.0')

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.on_event("startup")
def load_model():
    # assigned the loaded model to app.model
    app.model_dt_kmsmote = joblib.load('./models/dt_kmsmote.joblib')
    app.model_xgb_adasyn = joblib.load('./models/xgb_adasyn.joblib')
    app.model_xgb_kmsmote = joblib.load('./models/xgb_kmsmote.joblib')
    
@app.post("/ml-predict")
async def ml_predict_intus(*, user_input: intussusception_input_data):
    print(user_input)
    data_input = dict(user_input)['data']
    # data = {key: 0 for key in data_fetures}
    # data = pd.DataFrame(columns=data_fetures)
    print(data_input)
    
    data = convertinput2data(data_input)
    x = trasform_data_dict2df(data)
    result_dt_kmsmote = np.round(app.model_dt_kmsmote.predict_proba(x), decimals=6)[-1]
    result_xgb_adasyn = np.round(app.model_xgb_adasyn.predict_proba(x), decimals=6)[-1]
    result_xgb_kmsmote = np.round(app.model_xgb_kmsmote.predict_proba(x), decimals=6)[-1]
    print('result_dt_kmsmote', result_dt_kmsmote)
    print('result_xgb_adasyn', result_xgb_adasyn)
    print('result_xgb_kmsmote', result_xgb_kmsmote)
    # results = {}
    results = {
        "dt_kmsmote": {
            'Failed': float(result_dt_kmsmote[0]), 'Success': float(result_dt_kmsmote[1])
        },
        "xgb_adasyn": {
            'Failed': float(result_xgb_adasyn[0]), 'Success': float(result_xgb_adasyn[1])
        },
        "xgb_kmsmote": {
            'Failed': float(result_xgb_kmsmote[0]), 'Success': float(result_xgb_kmsmote[1])
        }
    }
    print('results', results)
    return results

if __name__ == "__main__":
    # for local testing
    import os
    os.system("uvicorn api:app --host 0.0.0.0 --port 3000 --reload")
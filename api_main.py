import os 
from fastapi import FastAPI 
from fastapi.encoders import jsonable_encoder


from pydantic import BaseModel, Field 

from starter.train_model import api_output, get_cat_features


app = FastAPI()
cat_features = get_cat_features() 


class InputData(BaseModel):
    age: int = Field(..., example=32)
    workclass: str = Field(..., example="Private")
    fnlwgt: int = Field(..., example=205019)
    education: str = Field(..., example="Assoc-acdm")
    education_num: int = Field(..., example=12)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Sales")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=50)
    native_country: str = Field(..., example="United-States")



if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system('rm -rf .dvc/cache')
    os.system('rm -rf .dvc/tmp/lock')
    os.system('dvc config core.hardlink_lock true')
    if os.system("dvc pull -q") != 0:
        exit("dvc pull failed")
    os.system("rm -rf .dvc .apt/usr/lib/dvc")



@app.get("/")
def home():
    return {"Hello": "Rushikesh This is your home page"}


@app.post('/inference')
async def predict_income(inputrow: InputData):
    row_dict = jsonable_encoder(inputrow)
    model_path = 'model/RF_with_encoder_lb.pkl'
    prediction = api_output(row_dict, model_path, cat_features)

    return {"Salary class": prediction}

#import uvicorn
# commenting out 
#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)
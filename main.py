from pandas import DataFrame
from joblib import load
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException

model = load("pipeline_reg.joblib")

app = FastAPI()

class DataPredict(BaseModel):
    data_to_predict: list[list] = [[17, 1, 0, 2, 19.833723, 7, 1, 2, 0, 0, 1, 0], 
                                   [18, 0, 0, 1, 15.408756, 0, 0, 1, 0, 0, 0, 0]]

@app.post("/predict")
def predict(request: DataPredict):
    try:
        list_data = request.data_to_predict
        df_data = DataFrame(list_data, columns=['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering'])   
        prediction = model.predict(df_data)
        return {"prediction": prediction.tolist()}

    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {'Universidad EIA': 'MLOps'}
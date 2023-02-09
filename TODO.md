# Testing
Add a test to determine is the sklearn model has been fitted see https://stackoverflow.com/questions/39884009/whats-the-best-way-to-test-whether-an-sklearn-model-has-been-fitted





# Handle variable names containing  hyphens inside Pydantic rather than in the Python package

The recommended way to handle this is to defined an `alias` for the relevant fields or to define an `alias_generator` for your Model. 

As well as accessing Pydantic model attributes directly via their names (e.g. `model.foobar`), models can be converted and exported in a number of ways: See [Exporting Models](https://pydantic-docs.helpmanual.io/usage/exporting_models/)

If data source field names do not match your code style (e. g. CamelCase fields), you can automatically generate aliases using `alias_generator`: See [Pydantic alias generator](https://docs.pydantic.dev/usage/model_config/#alias-generator)

```
from pydantic import BaseModel

def to_camel(string: str) -> str:
    return ''.join(word.capitalize() for word in string.split('_'))

class Voice(BaseModel):
    name: str
    language_code: str

    class Config:
        alias_generator = to_camel

voice = Voice(Name='Filiz', LanguageCode='tr-TR')
print(voice.language_code)
#> tr-TR
print(voice.dict(by_alias=True))
#> {'Name': 'Filiz', 'LanguageCode': 'tr-TR'}
```

```
def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")


class Input(BaseModel):
    age: int = Field(..., example=45)
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    fnlgt: int = Field(..., example=2334)
    hours_per_week: int = Field(..., example=60)
    marital_status: str = Field(..., example="Never-married")
    native_country: str = Field(..., example="Cuba")
    occupation: str = Field(..., example="Prof-specialty")
    race: str = Field(..., example="Black")
    relationship: str = Field(..., example="Wife")
    sex: str = Field(..., example="Female")
    workclass: str = Field(..., example="State-gov")

    class Config:
        alias_generator = hyphen_to_underscore
        allow_population_by_field_name = True
```



# Load model attributes at startup rather than when the FastAPI inference endpoint is called

Currently the model and other artifacts are loaded when the `predict` endpoint is called. This introduces some latency into the predictions. You can configure these artifacts to be loaded on start up of the  application.

See https://fastapi.tiangolo.com/advanced/events/

```
@app.on_event("startup")
async def startup_event(): 
	# Write the artifacts to global object so that can be accessed elsewhere 
    global model, encoder, binarizer
    model = pickle.load(open("./model/model.pkl", "rb"))
    encoder = pickle.load(open("./model/encoder.pkl", "rb"))
    binarizer = pickle.load(open("./model/lb.pkl", "rb"))
```



As well as a `startup` `on_event`, there is also a  `shutdown` `on_event`

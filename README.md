# XXX


## Overview


## Repo location



## Project structure
```
<content_root>
├── ml
│	├── __init__.py
│	├── data.py
│   └── model.py
├── model_artifacts
│	├── encoder.pk
│	├── lb.pk
│	├── model.pk
│   └── slice_output.txt
├── raw_data
│   └── census.csv
├── screenshots
│	├── continuous_deployment.PNG
│	├── continuous_integration.PNG
│	├── example.PNG
│	├── example_cont.PNG
│	├── live_get.PNG
│   └── live_post.PNG
├── tests
│	├── test_app.py
│   └── test_model_build.py
├── aequitas_bias_audit.ipynb
├── live_post.py
├── main.py
├── model_card.md
├── Procfile
├── README.md
├── requirements.txt
└── train_model.py
```

where
| Name |Description |
| :-- | :-- |
| /ml | Machine learning functions package |
| /ml/data.py | Functions to support data pre-processing |
| /ml/model.py | Functions to support the model build |
| /model_artifacts/encoder.pk | Saved encoder transform |
| /model_artifacts/lb.pk | Saved binariser encoder transform |
| /model_artifacts/model.pk | Saved decision tree model |
| /model_artifacts/slice_output.txt | Results of slice performance |
| /raw_data/census.csv | Development data |
| /raw_data/census.csv | Development data |
| /screenshots/continuous_deployment.PNG | Screenshot of continuous deployment |
| /screenshots/continuous_integration.PNG | Screenshot of continuous integration |
| /screenshots/example.PNG | Screenshot of FastAPI docs of POST request with default request body |
| /screenshots/example_cont.PNG | Screenshot of FastAPI docs of POST request with response body |
| /screenshots/live_get.PNG | Screenshot of of Heroku GET request |
| /screenshots/live_post.PNG | Screenshot of of Heroku POST request |



|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |



|
| :-------------------------------- | :-----------: | :-------------------------------- |
| data                              |   directory   | development data                  |
| images\eda                        |   directory   | EDA output                        |
| images\results                    |   directory   | model performance analysis output |
| logs                              |   directory   | test logs                         |
| models                            |   directory   | saved models                      |
| churn_library.py                  | Python module | model build module                |
| churn_script_logging_and_tests.py | Python module | module build module test script   |
| LICENSE                           | licence file  | licence T&Cs                      |
| README.me                         | markdown file | this project readme               |
| requirements.txt                  |   text file   | project dependencies              |

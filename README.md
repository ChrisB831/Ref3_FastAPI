# Overview
Here a random forest classifier predicting higher salary is developed on publicly available Census Bureau data. 

* Model performance is assessed acorss the entire test dataset along with individual slices
* A bias audit is conducted using Aequitas
* A series of units test are applied, which along with Flake8 are used the test the validity of the code before deployment
* The model is deployed using the FastAPI package. A further series of test asses the robustness of the AP
* The API is deployed using Heroku
* Finally GitHub Actions are utilised, which along with Heroku ensure full CI?CD capability



# Repo location
The Githun repo is located at https://github.com/ChrisB831/p3.git



# Development data
The data is based upon an extract of 32, 561 records from the 1994 US census (see [Link](https://archive.ics.uci.edu/ml/datasets/census+income



# Project structure
```
<content_root>
├── ml
│	├── __init__.py
│	├── data.py
│ └── model.py
├── model_artifacts
│	├── encoder.pk
│	├── lb.pk
│	├── model.pk
│ └── slice_output.txt
├── raw_data
│   └── census.csv
├── screenshots
│	├── continuous_deployment.PNG
│	├── continuous_integration.PNG
│	├── example.PNG
│	├── example_cont.PNG
│	├── live_get.PNG
│ └── live_post.PNG
├── tests
│	├── test_app.py
│ └── test_model_build.py
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
| /tests/test_app.py | Test script for the API defined in ./main.py |
| /tests/test_model_build.py | Test script for the model build defined in ./train_model.py |
| aequitas_bias_audit.ipynb | Jupyter notebook containing bias study code using Aequitas |
| live_post.py| Test a POST request to the live app hosted by Heroku |
| main.py | FastAPI app functionality |
| model_card.md | Model card detailing creation, use, and the shortcomings of the model |
| Procfile | Heroku Procfile |
| README.md | This readme |
| requirements.txt | Project dependencies |
| train_model.py | Core functionality to train, assess and save model |

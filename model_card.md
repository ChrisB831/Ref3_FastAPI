# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf



## Model Details
* Developed by Christopher Bonham in the "Deploying a Scalable ML Pipeline in Production" project as part of the "Machine Learning DevOps Engineer" Udacity nanodegree
* Release date is 9th February 2023
* The classifier was based on a Random Forest Classifier (see [Link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)). To avoid over training, the following parameters we changed from their default value
  *  `max_depth` = 5
  * `n_estimators`=  50



## Intended Use
The model is used to predict whether income exceeds $50K/yr based upon individual level data




## Training Data
The data is based upon an extract of 32, 561 records from the 1994 US census (see [Link](https://archive.ics.uci.edu/ml/datasets/census+income

The dataset columns were
| Column | Type | Description |
| :-: | :-: | :-- |
| age | continuous | |
| workclass | categorical | Working stats |
| fnlwgt | continuous |number of people the census believes the entry represents |
| education | categorical | highest education grade attained |
| education-num | continuous | number of years of education |
| marital-status | categorical | |
| occupation | categorical | |
| relationship | categorical | family status |
| race | categorical | ethnicity |
| sex | categorical | gender |
| capital-gain | continuous | |
| capital-loss | continuous ||
| hours-per-week | continuous | hours worked per week |
| native-country | categorical | |
| salary | categorical | $50k or less, $50k+ |

The categorical fields were transformed into  "one-hot" encodings  

The label fields (salary) was converted to a binary field, with 0 representing $50k or less and 1 representing  $50k+



## Evaluation Data

The development data was randomly split into training and test dataset in an 80:20 proportion, resulting in 26,048 records in the training data and 6,513 records in the test dataset.

 

## Metrics
Both the training and test datasets we evaluated against the following performance criteria
* [Precision](https://en.wikipedia.org/wiki/Precision_and_recall#Precision)
* [Recall](https://en.wikipedia.org/wiki/Precision_and_recall#Recall)
* [Fbeta](https://en.wikipedia.org/wiki/F-score)

The performance metrics suggests that over fitting is not present in the model
| Segment | Precision | Recall | Fbeta |
| :-: | :-: | :-: | :-: |
| Train | 0.819 | 0.424 | 0.559 |
| Test | 0.831 | 0.440 | 0.575 |



## Ethical Considerations
A slice analysis has indicated that  model performance changes significantly across categorical levels

An analysis using the Aequitas tool has indicated that the model is not fair for at least some classes within the data, specifically

* Some groups are not represented equally in the training data and that a classifier gives equal probability with respect to a target variable, independent of whether a subject is or is not contained within a protected group

* Not all groups have proportionately equal type 1 (false positive) and type 2 (false negative) rates made by the model


Highlighting a few illustrative cases, the model is...
* 1.27 times more likely to incorrectly predict a black individual has a lower income when compared to a white individual
* 1.65 times more likely to incorrectly predict a female individual has a lower income when compared to a male individual
* generally more likely to incorrectly predict individual from "non-professional" occupation (i.e. machine operator, service, cleaner) has a lower income when compared to a "professional" occupation



## Caveats and Recommendations
It is acknowledged that the developed model is very rudimentary. Its is suggested that performance may be increased by additional feature engineering and hyperparameter tuning.
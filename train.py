import datetime
import pandas as pd
import numpy as np
from datetime import datetime
'''for implementing simple logisticregression'''
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
'''for saving models'''
from joblib import dump
import os

TRAINING_FOLDER = "training"
VALIDATION_FOLDER = "evaluation"
BASE_PATH = os.getcwd()
TRAINING_PATH = os.path.join(BASE_PATH, TRAINING_FOLDER)
VALIDATION_PATH = os.path.join(BASE_PATH, VALIDATION_FOLDER)

training_files = os.listdir(TRAINING_PATH)
training_files = [os.path.join(TRAINING_PATH, filename) for filename in training_files]

def add_woman_data():
    gs = pd.read_csv(training_files[3])
    person = pd.read_csv(training_files[7])
    person = person.merge(gs, on="person_id")

    positive = person.loc[person.status == 1.0].reset_index(drop=True)
    negative = person.loc[person.status == 0.0].reset_index(drop=True)

    len_pos = positive.shape[0]
    negative_sample = negative.iloc[:len_pos]

    new_data = positive.append(negative_sample)
    print(new_data.status.value_counts())

    return new_data


def add_COVID_measurement_date():
    measurement = pd.read_csv(training_files[4],usecols =['person_id','measurement_date','measurement_concept_id','value_as_concept_id'])
    """
    measurement = measurement.loc[measurement['measurement_concept_id']==706163]
    measurement['value_as_concept_id'] = measurement['value_as_concept_id'].astype(int)
    measurement = measurement.loc[(measurement['value_as_concept_id']==45877985.0) | (measurement['value_as_concept_id']==45884084.0)]
    """
    measurement = measurement.sort_values(['measurement_date'],ascending=False).groupby('person_id').head(1)
    covid_measurement = measurement[['person_id','measurement_date']]
    print(covid_measurement)
    return covid_measurement

def add_demographic_data(covid_measurement):
    columns_to_use =  ['person_id','gender_concept_id','year_of_birth','race_concept_id']
    #person = pd.read_csv(training_files[7],usecols = ['person_id','gender_concept_id','year_of_birth','race_concept_id'])
    person = add_woman_data()[columns_to_use]
    print("person shape is", person.shape)
    demo = pd.merge(covid_measurement,person,on=['person_id'], how='inner')
    demo['measurement_date'] = pd.to_datetime(demo['measurement_date'], format='%Y-%m-%d')
    demo['year_of_birth'] = pd.to_datetime(demo['year_of_birth'], format='%Y')
    demo['age'] = demo['measurement_date'] - demo['year_of_birth']
    demo['age'] = demo['age'].apply(lambda x: x.days/365.25)
    gs = pd.read_csv(training_files[3])
    demo = demo.merge(gs, on="person_id")
    demo.drop(columns=["person_id", "measurement_date", "year_of_birth"], inplace=True)
    print("demo \n", demo)

def logit_model(predictors):
    X = predictors.drop(['person_id'], axis = 1)
    gs = pd.read_csv(training_files[3])
    res = predictors.merge(gs,on = ['person_id'], how ='left')
    res.fillna(0,inplace = True)
    X = np.array(X)
    Y = np.array(res[['status']]).ravel()
    print("X ", X)
    print("Y ", Y)
    clf = LogisticRegressionCV(cv = 5, penalty = 'l2', tol = 0.0001, fit_intercept = True, intercept_scaling = 1, class_weight = None, random_state = None,
    max_iter = 100, verbose = 0, n_jobs = None).fit(X,Y)
    dump(clf, 'baseline.joblib')
    print("Training stage finished", flush = True)


    covid_measurement = add_COVID_measurement_date()
    predictors = add_demographic_data(covid_measurement)
    logit_model(predictors)
if __name__ == '__main__':
    
    covid_measurement = add_COVID_measurement_date()
    predictors = add_demographic_data(covid_measurement)
    logit_model(predictors)
    
    #add_woman_data()

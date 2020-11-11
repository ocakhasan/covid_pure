import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

import os

TRAINING_FOLDER = "training"
VALIDATION_FOLDER = "evaluation"
BASE_PATH = os.getcwd()
TRAINING_PATH = os.path.join(BASE_PATH, TRAINING_FOLDER)
VALIDATION_PATH = os.path.join(BASE_PATH, VALIDATION_FOLDER)

training_files = os.listdir(TRAINING_PATH)
training_files = [os.path.join(TRAINING_PATH, filename) for filename in training_files]

def get_data():
    person = pd.read_csv(training_files[7])
    gs = pd.read_csv(training_files[3])

    person = person.merge(gs, on="person_id")

    
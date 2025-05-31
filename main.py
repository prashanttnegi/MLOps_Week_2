### Initialize Vertex AI SDK for Python

'''To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com). '''

from google.cloud import aiplatform

### Set Google Cloud project information 
''' Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).'''

PROJECT_ID = "sharp-cursor-461116-t1"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

### Create a Cloud Storage bucket

''' Create a storage bucket to store intermediate artifacts such as datasets.'''

BUCKET_URI = f"gs://mlops-course-sharp-cursor-461116-t1-week2"  # @param {type:"string"}

aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

### Import the required libraries
import os
import sys

### Configure resource names

'''Set a name for the following parameters:

`MODEL_ARTIFACT_DIR` - Folder directory path to your model artifacts within a Cloud Storage bucket, for example: "my-models/fraud-detection/trial-4"

`REPOSITORY` - Name of the Artifact Repository to create or use.

`IMAGE` - Name of the container image that is pushed to the repository.

`MODEL_DISPLAY_NAME` - Display name of Vertex AI model resource. '''

MODEL_ARTIFACT_DIR = "my-models/iris-classifier-week-2"  # @param {type:"string"}
REPOSITORY = "iris-classifier-repo"  # @param {type:"string"}
IMAGE = "iris-classifier-img"  # @param {type:"string"}
MODEL_DISPLAY_NAME = "iris-classifier"  # @param {type:"string"}

# Set the defaults if no names were specified
if MODEL_ARTIFACT_DIR == "[your-artifact-directory]":
    MODEL_ARTIFACT_DIR = "custom-container-prediction-model"

if REPOSITORY == "[your-repository-name]":
    REPOSITORY = "custom-container-prediction"

if IMAGE == "[your-image-name]":
    IMAGE = "sklearn-fastapi-server"

if MODEL_DISPLAY_NAME == "[your-model-display-name]":
    MODEL_DISPLAY_NAME = "sklearn-custom-container"

## Simple Decision Tree model
'''Build a Decision Tree model on iris data '''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

data = pd.read_csv('data/iris.csv')
data.head(5)
train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species
mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)
prediction=mod_dt.predict(X_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))
import pickle
import joblib

joblib.dump(mod_dt, "artifacts/model.joblib")
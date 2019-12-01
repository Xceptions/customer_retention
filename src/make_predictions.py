import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import KFold
import pickle
import os, sys


class MakePredictions():
    def __init__(self):
        self.model_preds = pd.DataFrame({})
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file = 'test.csv'
        self.read_data()
        
    def read_data(self):
        data_path = os.path.join(self.repo_path, 'data', 'advanced_features', self.file)
        self.df = pd.read_csv(data_path)
        self.df = self.df.drop('Target', axis='columns')
        model_path = os.path.join(self.repo_path, 'data', 'nn_model', 'nn_model.h5')
        self.model = keras.models.load_model(model_path)
        self.predict_test()

    def predict_test(self):
        self.predictions = self.model.predict(self.df)
        self.predictions = (self.predictions>0.5)
        self.predictions = self.predictions.ravel()
        self.predictions = [int(x) for x in self.predictions]
        self.prepare_to_save()

    def prepare_to_save(self):
        preds = pd.DataFrame({})
        preds['predictions'] = self.predictions
        # preds['Id'] = preds['predictions'].index
        self.model_preds = preds
        self.savepredictions()

    def savepredictions(self):
        data_path = os.path.join(self.repo_path, 'data', 'model_predictions')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        combined_path_test = os.path.join(data_path, 'predictions.csv')
        self.model_preds.to_csv(combined_path_test, index=False)


if __name__ == "__main__":
    MakePredictions()
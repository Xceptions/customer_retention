import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import KFold
import pickle
import os, sys


class Train_NN_Model():
    def __init__(self):
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file = 'train.csv'
        self.read_data()
        
    def read_data(self):
        data_path = os.path.join(self.repo_path, 'data', 'advanced_features', self.file)
        self.df = pd.read_csv(data_path)
        self.feature_target_split()

    def feature_target_split(self):
        self.target = self.df['Target'].copy()
        self.df = self.df.drop('Target', axis='columns')
        self.df = self.df.values
        self.target = self.target.values
        self.train_nn()

    def train_nn(self):
        input_dim = self.df.shape[1]
        model = Sequential()
        model.add(Dense(2000, activation='relu', input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1000, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(500, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(125, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        es = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=10,
            verbose=1,
            mode='auto',
            restore_best_weights=True
        )
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                    )
        model.fit(
            self.df, self.target,
            epochs=1,
            batch_size=5000,
            verbose=1,
            validation_split=0.25,
            callbacks=[es]
        )
        self.model = model
        self.savemodel()

    def savemodel(self):
        data_path = os.path.join(self.repo_path, 'data', 'nn_model')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        combined_path_test = os.path.join(data_path, 'nn_model.h5')
        self.model.save(combined_path_test)

if __name__ == '__main__':
    Train_NN_Model()
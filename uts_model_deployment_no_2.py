# -*- coding: utf-8 -*-
"""UTS Model Deployment No 2

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1N4qtnz3WZb57Z_gl8X9AKVXs9F4gn7ek
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, log_loss
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle

class predata:
  def __init__(self, file_name, delimiter = ','):
    self.file_name = file_name
    self.file = pd.read_csv(file_name, delimiter = ',')
    self.input = None
    self.output = None

  def drop_file(self, columns):
    self.file.drop(columns = columns, inplace=True)

  def head_info(self):
    print(self.file.head())

  def information(self):
    print(self.file.shape)
    print()
    print(self.file.info())
    print()
    print(self.file.columns)
    print()
    print(self.file.describe())

  def missing(self):
    print(self.file.isna().sum())

  def feature_encoding(self, label_encoding):
    self.file = self.file.replace(label_encoding)

  def mean(self, columns):
    print(np.mean(self.file[columns]))

  def missing_value(self, columns, value):
    self.file[columns] = self.file[columns].fillna(value)

  def split(self, column):
    self.input = self.file.drop(columns = [column])
    self.output = self.file[column]
    x_train, x_test, y_train, y_test = train_test_split(self.input, self.output, test_size = 0.2, random_state = 42)
    return [x_train,y_train],[x_test,y_test]

"""...
class Handling:
  def __init__(self, train_data, test_data):
    self.x_train, self.y_train = train_data
    self.x_test, self.y_test = test_data
    self.rf_model = None
    self.xgb_model = None

  def modeling1(self):
    self.rf_model = RandomForestClassifier()
    self.xgb_model = XGBClassifier()

  def fitting(self):
    self.rf_model = self.rf_model.fit(self.x_train, self.y_train)
    self.xgb_model = self.xgb_model.fit(self.x_train, self.y_train)
    self.model = self.model.fit(self.x_train, self.y_train)

  def predict(self):
    self.rf_model = self.rf_model.fit(self.x_test)
    self.xgb_model = self.xgb_model.fit(self.x_test)

  def report(self):
    print(classification_report(self.y_test, self.y_predict))

"""

class Handling:
  def __init__(self, train_data, test_data):
    self.x_train, self.y_train = train_data
    self.x_test, self.y_test = test_data
    self.model = None
    self.y_predict = None

  def modeling(self, model):
    self.model = model

  def fitting(self):
    self.model = self.model.fit(self.x_train, self.y_train)

  def predict(self):
    self.y_predict = self.model.predict(self.x_test)

  def report(self, ):
    print(classification_report(self.y_test, self.y_predict))

  def save_pickle(self, filename):
    with open(filename, 'wb') as f:
      pickle.dump(self.model, f)

df = predata('data_C.csv')

df.head_info()

df.information()

df.missing()

df.drop_file('id')
df.drop_file('Surname')
df.drop_file('Geography')

df.mean('CreditScore')

encoding = {"Gender": {"Male": 1 , "Female" : 0}}
df.feature_encoding(encoding)

df.missing_value('CreditScore', 655.80)

df.head_info()

train_data, test_data = df.split('churn')

model = Handling(train_data, test_data)
model2 = Handling(train_data, test_data)

model.modeling(RandomForestClassifier())

model.fitting()

model.predict()

model.report()

model2.modeling(XGBClassifier())

model2.fitting()

model2.predict()

model2.report()

model.save_pickle('rf_classifier.pkl')
model2.save_pickle('xgb_classifier.pkl')
""" 
    Dag has a goal of fetching the data from the data/raw folder,
    generating features, training, and predicting on test data
"""
from datetime import timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator
# from airflow.utils.dates import days_ago
import sys

# sys.path.insert(0,"./src/")
# print(sys.path[0])
from data_preprocessing import Preprocess_Data, \
    Generate_historical_features, Generate_advanced_features
from train_model import Train_NN_Model
from make_predictions import MakePredictions

default_args = {"start_date": "2019-10-24",
                "email": ["agbokenechukwu.k@gmail.com"],
                "email_on_failure": True,
                "eamil_on_retry": True,
                "retries": 0,
                "retry_delay": timedelta(minutes=5)}

dag = DAG("project_pipeline",
          description="Building the entire project",
          # train every first day of the month
          schedule_interval="@monthly",
          default_args=default_args,
          catchup=False)

with dag:
    task_1_preprocess_data = PythonOperator(task_id="preprocess_data",
                                                 python_callable=Preprocess_Data)

    task_2_create_historical_features = PythonOperator(task_id="generate_historic_features",
                                                     python_callable=Generate_historical_features)

    task_3_create_advanced_features = PythonOperator(task_id="generate_advanced_features",
                                                     python_callable=Generate_advanced_features)

    task_4_train_nn_model = PythonOperator(task_id="train_nn_model",
                                                     python_callable=Train_NN_Model)

    task_5_make_predictions = PythonOperator(task_id="make_predictions",
                                                     python_callable=MakePredictions)


    task_1_preprocess_data >> task_2_create_historical_features >> task_3_create_advanced_features \
        >> task_4_train_nn_model >> task_5_make_predictions


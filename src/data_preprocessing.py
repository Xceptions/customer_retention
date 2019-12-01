import numpy as np
import pandas as pd
import gc
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import pickle

class Preprocess_Data():
    """
    Class engineers new features based on EDA and domain knowledge of the data
    """
    def __init__(self):
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for self.file in ['train.csv', 'test.csv']:
            self.read_data()
        
    def read_data(self):
        data_path = os.path.join(self.repo_path, 'data', 'raw', self.file)
        self.df = pd.read_csv(data_path)
        self.reduce_mem_usage()

    def reduce_mem_usage(self):
        """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.        
        """
        start_mem = self.df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        
        for col in self.df.columns:
            col_type = self.df[col].dtype
            
            if col_type != object:
                c_min = self.df[col].min()
                c_max = self.df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.df[col] = self.df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.df[col] = self.df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.df[col] = self.df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.df[col] = self.df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.df[col] = self.df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.df[col] = self.df[col].astype(np.float32)
                    else:
                        self.df[col] = self.df[col].astype(np.float64)
            else:
                self.df[col] = self.df[col].astype('category')

        end_mem = self.df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        self.fill_missing_values()

    def fill_missing_values(self):
        self.df['E'] = self.df['E'].fillna(-999).astype(np.int16)
        self.df['J'] = self.df['J'].fillna(-999).astype(np.int16)
        self.generate_col_group()

    def group_col(self, x):
        if ((x>=18 and x<30) or (x>=60 and x<70)):
            return "one"
        elif ((x>=30 and x<40) or (x>=40 and x<50) or (x>=50 and x<60)):
            return "two"
        else:
            return "three"
    
    def generate_col_group(self):
        self.df['B_group'] = self.df['B'].apply(self.group_col)
        self.savefile()

    def savefile(self):
        data_path = os.path.join(self.repo_path, 'data', 'preprocessed_data')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        combined_path_test = os.path.join(data_path, self.file)
        self.df.to_csv(combined_path_test, index=False)
        print("saved")



class Generate_historical_features():
    """
    Historical features are based on data contained in the train.
    I generate features such as statistical mean, median, std.
    These features will be applied to both train and test data in
    a way that ensures no leakage.
    """
    def __init__(self):
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file = 'train.csv'
        self.read_data()
        
    def read_data(self):
        data_path = os.path.join(self.repo_path, 'data', 'preprocessed_data', self.file)
        self.df = pd.read_csv(data_path)
        self.create_group_stats()

    def create_group_stats(self):
        df_group =  self.df.groupby('B_group')['B']
        B_group_min = df_group.min().astype(np.float16).to_dict()
        B_group_mean = df_group.mean().astype(np.float16).to_dict()
        B_group_median = df_group.median().astype(np.float16).to_dict()
        B_group_std = df_group.std().astype(np.float16).to_dict()
        B_group_max = df_group.max().astype(np.float16).to_dict()
        B_group_count = df_group.count().to_dict()
        self.group_stats = [B_group_min,
                            B_group_mean,
                            B_group_median,
                            B_group_max,
                            B_group_std,
                            B_group_count]
        self.savefile()

    def savefile(self):
        data_path = os.path.join(self.repo_path, 'data', 'group_statistics')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        combined_path_test = os.path.join(data_path, 'group_stats.pkl')
        pickle.dump(self.group_stats, open(combined_path_test, 'wb'))


class Generate_advanced_features():
    """
    Features generated in the historical class are applied to both train
    and test dataframes
    """
    def __init__(self):
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for self.file in ['train.csv', 'test.csv']:
            self.read_data()
        
    def read_data(self):
        data_path = os.path.join(self.repo_path, 'data', 'preprocessed_data', self.file)
        pkl_path = os.path.join(self.repo_path, 'data', 'group_statistics', 'group_stats.pkl')
        self.df = pd.read_csv(data_path)
        self.pkl = pickle.load(open(pkl_path, 'rb'))
        self.map_features()

    def map_features(self):
        B_group_min = self.pkl[0]
        B_group_mean = self.pkl[1]
        B_group_median = self.pkl[2]
        B_group_max = self.pkl[3]
        B_group_std = self.pkl[4]
        B_group_count = self.pkl[5]
        self.df['group_mean'] = self.df['B_group'].map(B_group_mean)
        self.df['group_median'] = self.df['B_group'].map(B_group_median)
        self.df['group_min'] = self.df['B_group'].map(B_group_min)
        self.df['group_max'] = self.df['B_group'].map(B_group_max)
        self.df['group_std'] = self.df['B_group'].map(B_group_std)
        self.df['group_count'] = self.df['B_group'].map(B_group_count)
        self.df = self.df.drop(['B_group'], axis='columns')
        self.savefile()

    def savefile(self):
        data_path = os.path.join(self.repo_path, 'data', 'advanced_features')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        combined_path_test = os.path.join(data_path, self.file)
        self.df.to_csv(combined_path_test, index=False)

if __name__== "__main__":
    Preprocess_Data()
    Generate_historical_features()
    Generate_advanced_features()
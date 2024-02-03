import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    '''
        This function is responsible for dat transformation.
    '''
    def __init__(self):
        self.data_trf_config = DataTransformationConfig()
    
    def get_data_trf_obj(self):
        try:
            num_features = ['math_score', 'reading_score', 'writing_score']
            cat_features = ['gender', 'race_ethnicity', 'lunch', 
                            'parental_level_of_education', 'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy= 'median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Numerical columns standard scaling completed.')

            cat_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy= 'most_frequent')),
                    ('onehotencoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Categorical columns encoding completed.')

            preprocess = ColumnTransformer(
                [
                    ('numerical_pipeline', num_pipeline, num_features),
                    ('categorical_pipeline', cat_pipeline, cat_features)
                ]
            )

            return preprocess

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_trf(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed.')

            logging.info('Obtaining preprocessing object.')

            preprocessing_obj = self.get_data_trf_obj()

            target_column = 'total_score'
            num_columns = ['math_score', 'reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns= [target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('Applying preprocessing object on training and testing dataframe.')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info('Saved preprocessing object.')

            save_obj(
                file_path= self.data_trf_config.preprocess_obj_file_path,
                obj= preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_trf_config.preprocess_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)









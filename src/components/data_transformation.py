import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """
        This method creates and returns a ColumnTransformer object that preprocesses
        the data by applying separate transformations to numerical and categorical columns.
    
        Numerical columns undergo imputation and scaling, while categorical columns 
        undergo imputation, one-hot encoding, and scaling.
    
        Returns:
            ColumnTransformer: A preprocessor object to apply transformations to the data.
        """
        try:
            numerical_columns = ["reading_score", "writing_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f"Numerical columns: {numerical_columns}")
            
            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        """
        This method reads training and testing data from CSV files, applies the 
        preprocessing transformations to the features (numerical and categorical),
        and returns the transformed training and testing data arrays, along with the 
        file path of the saved preprocessing object.
        
        Args:
            train_path (str): Path to the training data CSV file.
            test_path (str): Path to the testing data CSV file.
        
        Returns:
            tuple: A tuple containing the transformed training data, transformed testing data,
                and the file path of the saved preprocessing object.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "math_score"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe"
            )
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # The transformed features are combined with the target labels
            # into a single array (train_arr and test_arr)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saving the preprocessing object as a pkl file")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
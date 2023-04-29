import os
import numpy as np
from logger import logging
from exception import CustomException
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import pandas as pd
import sys
from utils import save_object
@dataclass
class DataTransformationConfig:
     preprocessor_obj_file_path=os.path.join('Artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
          self.data_transformation=DataTransformationConfig()

    def get_data_transformer_object(self):
          '''
          This function is responsible for Data Transformation
          '''     
          try:
            categorical_columns = ['batting_team','bowling_team',]
            numerical_columns = [
                "runs_left",
                "balls_left",
                "wickets",
                "total_runs_x",
                "crr",
                'rrr'
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
          except Exception as e:
            raise CustomException(e,sys)
  
    def initiate_data_transformation(self,train_path,test_path):

        logging.info('Initiation of Data Transformation')

        try: 
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            logging.info('Train and test data loading has completed')

            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name="result"
            numerical_columns = [
                "runs_left",
                "balls_left",
                "wickets",
                "total_runs_x",
                "crr",
                'rrr'
            ]

            input_feature_train_df=train_data.drop(columns=[target_column_name,'city'],axis=1)
            target_feature_train_df=(train_data[target_column_name])

            input_feature_test_df=test_data.drop(columns=[target_column_name,'city'],axis=1)

            target_feature_test_df=test_data[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            print(input_feature_train_arr)
            print((input_feature_train_arr.shape))
            print((target_feature_train_df.shape))

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
           # train_arr=np.concatenate((input_feature_train_arr,target_feature_train_df.reshape(-1,1)),axis=1)
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            print(self.data_transformation.preprocessor_obj_file_path)
            save_object(

                file_path=self.data_transformation.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
             
            return (
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
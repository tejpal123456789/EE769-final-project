
# All the data ingestion wali cheze will be done here only
# In this I will load load data and split it into train and text before applyinh transformatuon onut
import sys

from logger import logging
#from logger import logging
from exception import CustomException

from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass
from components.data_transformation import DataTransformation
from components.data_transformation import DataTransformationConfig
from components.model_trainer import ModelTrainerConfig
from components.model_trainer import ModelTrainer
print('tej')
import os
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('Artifacts','train.csv')
    test_data_path:str=os.path.join('Artifacts','test.csv')
    raw_data_path:str=os.path.join('Artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion has started')
        try:
            match=pd.read_csv('Data/matches.csv')
            delivery=pd.read_csv('Data/deliveries.csv')
            total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
            match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')
            
            logging.info('Reading the csv file as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            teams = [
                    'Sunrisers Hyderabad',
                    'Mumbai Indians',
                    'Royal Challengers Bangalore',
                    'Kolkata Knight Riders',
                    'Kings XI Punjab',
                    'Chennai Super Kings',
                    'Rajasthan Royals',
                    'Delhi Capitals'
                     ]
            match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
            match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

            match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
            match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
            match_df = match_df[match_df['team1'].isin(teams)]
            match_df = match_df[match_df['team2'].isin(teams)]
            match_df = match_df[match_df['dl_applied'] == 0]
            match_df = match_df[['match_id','city','winner','total_runs']]
            delivery_df = match_df.merge(delivery,on='match_id')
            delivery_df = delivery_df[delivery_df['inning'] == 2]
            delivery_df['current_score'] = delivery_df.groupby('match_id').cumsum()['total_runs_y']
            delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
            delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])
            delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
            delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")
            delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
            wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
            delivery_df['wickets'] = 10 - wickets
            
            # crr = runs/overs
            delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])
            delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']
            def result(row):
                return 1 if row['batting_team'] == row['winner'] else 0
            delivery_df['result'] = delivery_df.apply(result,axis=1)
            final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]
            final_df = final_df.sample(final_df.shape[0])
            final_df = final_df[final_df['balls_left'] != 0]
            
            final_df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)





          #  logging.info('Train test split initiated')

            train_set,test_set=train_test_split(final_df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

           # logging.info("Ingestion of the data is completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
    obj1=DataTransformation()
    train_arr,test_arr,_=obj1.initiate_data_transformation(train_path,test_path)
    model=ModelTrainer()
    print(model.initiate_model_trainer(train_arr,test_arr))
    
        

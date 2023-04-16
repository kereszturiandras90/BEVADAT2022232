import pandas as pd

csv_path = "diabetes.csv"

class NJCleaner:
    k_neighbors = 0

    def __init__(self, path :str) -> None:
        self.data = pd.read_csv(path)

    def order_by_scheduled_time(self) -> pd.DataFrame:
        self.data.sort_values('scheduled_time')
        return self.data
        #print(self.data.sort_values('scheduled_time'))

    def drop_columns_and_nan(self) -> pd.DataFrame:
        self.data = self.data.drop(['from', 'to'], axis=1).dropna(axis=0)
        return self.data
    def convert_date_to_day(self) -> pd.DataFrame:
        self.data['date'] = pd.to_datetime(self.data['date']).dt.day_name()
        #pd.to_datetime(self.data['date']).dt.day_name()
        return self.data
    
  
    
    def convert_scheduled_time_to_part_of_the_day(self) -> pd.DataFrame:
        
        self.data['scheduled_time'] = pd.to_datetime(self.data['scheduled_time'])
        hour = self.data['scheduled_time'].dt.hour
    
        self.data['part_of_the_day'] = hour.apply(lambda x: 'late_night' if 0 <= x < 4 else
                                                 'early_morning' if 4 <= x < 8 else
                                                 'morning' if 8 <= x < 12 else
                                                 'afternoon' if 12 <= x < 16 else
                                                 'evening' if 16 <= x < 20 else 'night')

        self.data = self.data.drop(['scheduled_time'], axis=1)
        return self.data
    
    def convert_delay(self) -> pd.DataFrame:
        delaymin = self.data['delay_minutes']
        self.data['delay'] = delaymin.apply(lambda x: 0 if 0<= x <5 else 1)
        return self.data
    
    def drop_unnecessary_columns(self) -> pd.DataFrame:
        
        self.data = self.data.drop(['train_id', 'actual_time', 'delay_minutes'], axis=1)
        return self.data
    def save_first_60k(self, path: str) ->None:
        self.data.head(60000).to_csv(path)
        return
    
    def prep_df(self, path = 'data/NJ.csv'):
        self.order_by_scheduled_time()
        self.drop_columns_and_nan()
        self.convert_date_to_day()
        self.convert_scheduled_time_to_part_of_the_day()
        self.convert_delay()
        self.drop_unnecessary_columns()
        self.save_first_60k(path)





       
import pandas as pd
full_record_df = 'df_red.csv' 
prepared_record_df = 'df_prepared.csv'
print( prepared_record_df, pd.read_csv(prepared_record_df).drop(['Unnamed: 0'], axis=1).columns)
print( pd.read_csv(full_record_df).drop(['Unnamed: 0'], axis=1).columns)


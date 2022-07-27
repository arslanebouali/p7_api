import pandas as pd

class data():

    def __init__(self,  full_record_df = 'data/df_red.csv' , prepared_record_df = 'data/df_prepared.csv'):

        try:
            self.full= pd.read_csv(full_record_df,header=0,sep=',').drop(['Unnamed: 0','TARGET'], axis=1)
        except ValueError:
            print(pd.read_csv(full_record_df,header=0,sep=',').columns)



        if 'TARGET' in self.full.columns:
            self.full.drop(['Unnamed: 0'],axis=1)

        self.prepared = pd.read_csv(prepared_record_df,index_col='SK_ID_CURR',header=0,sep=',').drop(['Unnamed: 0','TARGET'], axis=1)

        self.Target = pd.read_csv(full_record_df,header=0,sep=',').drop(['Unnamed: 0'], axis=1)
        

        



    def full_records(self, client_id):
        full = self.full
        prepared = self.prepared

        client_full_predict = full[full['SK_ID_CURR'] == int(client_id)].drop('SK_ID_CURR',axis=1)
        client_full = prepared.loc[int(client_id),:]


        return client_full ,client_full_predict






import pandas as pd
import numpy as np

def seqs_in_cluster(df:pd.DataFrame, cluster_col:str, cluster_nums:list, patid = 'enrolid', event_column = 'event_real', sep = ', ') -> pd.Series:
    df = df.sort_values(by = [patid,'occ'])
    if cluster_nums == 'all':
        return df.groupby(patid)[event_column].apply(lambda x:sep.join(x)).value_counts()
    elif isinstance(cluster_nums,list):
        return df[df[cluster_col].isin(cluster_nums)].groupby(patid)[event_column].apply(lambda x:sep.join(x)).value_counts()#.sort_index()
    else:
        return df[df[cluster_col]== cluster_nums].groupby(patid)[event_column].apply(lambda x:sep.join(x)).value_counts()#.sort_index()

def num_unique_seqs(df:pd.DataFrame, patid = 'enrolid', event_column = 'event_real') -> int:
    return len(seqs_in_cluster(df,None,'all',patid,event_column))

def cluster_representatives(df:pd.DataFrame,num_reps = 2, patid = 'enrolid',event_column = 'event_real') -> pd.DataFrame:
    c = []
    for i in df.Clusters.unique():
        c.append((i,seqs_in_cluster(df,'Clusters',[i],patid,event_column, sep = '- ').sort_values(ascending = False)[:num_reps].index.tolist()))
    return   pd.DataFrame(c).explode(1).sort_values(by = 0).rename({0:'Clusters',1:'seq'},axis = 1)

def print_cluster_results(df:pd.DataFrame,cluster_col:str,ignore_less_than = 0,patid = 'enrolid',event_column = 'event_real', clusts = []):
    if len(clusts) == 0:
        for i in df[cluster_col].unique():
            print(f'CLUSTER: {i}')
            print(seqs_in_cluster(df,cluster_col,i,patid,event_column).sort_values(ascending = False)[seqs_in_cluster(df,cluster_col,i,patid,event_column).sort_values(ascending = False)>ignore_less_than])
    else:
        for i in df[cluster_col].unique():
            if i in clusts:
                print(f'CLUSTER: {i}')
                print(seqs_in_cluster(df,cluster_col,i,patid,event_column).sort_values(ascending = False)[seqs_in_cluster(df,cluster_col,i,patid,event_column).sort_values(ascending = False)>ignore_less_than])

                
def patients_per_cluster(df:pd.DataFrame,cluster_col:str,patid = 'enrolid') -> pd.Series:
    return df.groupby(cluster_col)[patid].unique().str.len().sort_values(ascending = False)
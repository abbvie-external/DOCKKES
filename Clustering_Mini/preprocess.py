'''preprocessing functions'''

import pandas as pd
import numpy as np

'''Threshold is the decimal value at which you drop events with a frequency of appearance below. e.g threshold of .0001 means
"drop all events that are below .01% in terms of frequency"'''
def remove_below_thresh(df:pd.DataFrame,threshold:float,event_column:str,print_on = False):
    thresh = len(df) * threshold
    uniques = [(i,len(j)) for i,j in df.groupby([event_column]) if len(j) > thresh]
    if print_on:
        for i in sorted(uniques,key = lambda x: x[1], reverse = True):
            print(i)
    to_keep = pd.DataFrame(uniques, columns = ['event','count'])
    return df[df[event_column].isin(to_keep.event)]

def get_event_frequencies_as_column(df:pd.DataFrame,event_columns:list,name = 'freq_total') -> pd.DataFrame:
    freqs = df.groupby(event_columns).count().unique_row.reset_index().rename({'unique_row':name},axis = 1)
    return pd.merge(freqs,df, on = event_columns)

def add_no_therapy_event(df:pd.DataFrame, patid = 'enrolid', num_event_for_no_therapy = 1, remove_no_therapy_patients = False) -> pd.DataFrame:
    if remove_no_therapy_patients:
        return df.groupby([patid]).filter(lambda x: x[patid].count() > num_event_for_no_therapy)
    a = df.groupby([patid]).filter(lambda x: x[patid].count() > num_event_for_no_therapy)
    b = df.groupby([patid]).filter(lambda x: x[patid].count() <= num_event_for_no_therapy)
    no_ther = b.copy(deep = True)
    no_ther.event_real = 'No Therapy:'
    conc = pd.concat([b,no_ther],axis = 0)
    return pd.concat([conc,a],axis = 1)
    
#def preprocess(df: pd.DataFrame, patid = 'enrolid', event_column = 'event_real', freq_threshold = .0001, event_threshold = 1,num_event_for_no_therapy = 1, remove_no_therapy_patients = False,occ_number = 0, num_even = 10) -> pd.DataFrame:
#    
#    # Add No Therapy to patients that have an episode, but no medication treatment
#    df = add_no_therapy_event(df,patid,num_event_for_no_therapy,remove_no_therapy_patients)
#    
#    # Removes patients with <= event_threshold number of events
#    df = df.groupby([patid]).filter(lambda x: x[patid].count() > event_threshold)
#    
#    # Remove events where frequencies are below decimal valued freq_threshold
#    df = remove_below_thresh(df,freq_threshold,event_column)
#    df = df.reset_index().drop('index',axis = 1).reset_index().rename({'index':'unique_row'},axis = 1)
#    
#    #Remove leading and trailing whitespace
#    df[event_column] = df[event_column].apply(lambda x: x.strip())
#    
#    # Create occurrence column for when the event occurs out of all events based on grouping patients by patid
#    df['occ'] = df.groupby([patid]).cumcount()
#    
#    # Filter events greater or equal to a position in sequence
#    df = filter_events_after_occ(df,occ_number)
#    
#    # Create times column for when the event occurs within the same type of event (for repeat events) based on grouping patients by patid and event
#    df['times'] = df.groupby([patid,event_column]).cumcount() + 1
#    
#    # Get Event frequencies
#    df = get_event_frequencies_as_column(df,[patid,event_column],'frequency')
#    
#    #Event frequencies total
#    df = get_event_frequencies_as_column(df,[event_column],'freq_total')
#    
#    # Get Lag of total frequency for each event
#    df = freq_total_lag(df,patid)
#    
#    # Create the iDF calculation or inverse document frequency. In this case it is inverse patient frequency
#    df = get_idf(df,patid,event_column)
#    
#    # Create weighted order by number of events that patients have with 
#    df = weighted_order(df,patid,event_column,num_even)
#    
#    # Create Weighted order lag column by 1
#    df = weighted_order_lag(df,patid)
#    
#    # Turn events into categorical numbers
#    df = categorical_event_and_value(df,event_column)
#    
#    # Create the tfidf calculation with frequency of event * log (number of patients / patients with event)
#    return tfidf(df,patid)
#  
  

def preprocess(df:pd.DataFrame,patid = 'enrolid',event_column = 'event_real', remove_no_therapy_events = True) -> pd.DataFrame:
    if remove_no_therapy_events:
        df = df[~df[event_column].str.strip().str.startswith('No Therapy:')]
    
    # Create Occ column which contains the Position within sequence
    df['occ'] = df.groupby([patid]).cumcount()

    #Event frequencies total
    df = get_event_frequencies_as_column(df,[event_column],'freq_total')

    # Get Lag of total frequency for each event
    df = freq_total_lag(df,patid)
    
    # Create weighted order by number of events that patients have with 
    df = weighted_order(df,patid,event_column,0)

    # Create Weighted order lag column by 1
    df = weighted_order_lag(df,patid)

    # Sort values
    df.sort_values(by =[patid,'occ'])
    return df


def weighted_order_lag(df:pd.DataFrame,patid = 'enrolid') -> pd.DataFrame:
    df['w_occ_lag'] = df.groupby([patid])['w_occ'].shift(1).fillna(0)
    return df

def freq_total_lag(df:pd.DataFrame,patid = 'enrolid')-> pd.DataFrame:
    df['freq_total_lag'] = df.groupby([patid])['freq_total'].shift(1).fillna(1)
    return df

    
def categorical_event_and_value(df:pd.DataFrame,column:str) -> pd.DataFrame:
    df['catevent'] = pd.Categorical(df[column])
    df['cat_val'] = df.catevent.cat.codes + 1
    return df

def filter_events_after_occ(df:pd.DataFrame, occ_number:int):
     return df[df['occ']>=occ_number]

#def weighted_order(df:pd.DataFrame,patid = 'enrolid',event_column = 'event_real',num_even = 10) -> pd.DataFrame:
#    df1 = df.groupby([event_column,'occ']).count().sort_values(by = [patid,'occ'],ascending = False)[patid].reset_index()
#    df1['ranking'] = df1.groupby(['occ'])[patid].rank(method = 'first',ascending = False)
#    df1.sort_values(by = ['occ','ranking'],ascending = True,inplace = True)
#    df1.rename({'enrolid':'num_people_with_event','ranking':'w_occ'},axis = 1,inplace = True)
#  
#    return pd.merge(df,df1,on = ['occ',event_column], how = 'left')

def weighted_order(df:pd.DataFrame,patid = 'enrolid',event_column = 'event_real',num_even = 10) -> pd.DataFrame:
    
    df1 = df.groupby([event_column,'occ']).count().sort_values(by = [patid,'occ'],ascending = False)[patid].reset_index()
    #print(df1.sort_values(by = 'enrolid').head())
    df1['ranking'] = df1.groupby(['occ'])[patid].rank(method = 'first',ascending = False)
    df1.sort_values(by = ['occ','ranking'],ascending = True,inplace = True)
    print(df1)
    df1.rename({patid:'num_people_with_event','ranking':'w_occ'},axis = 1,inplace = True)
    return pd.merge(df,df1,on = ['occ',event_column], how = 'left')

def tfidf(df:pd.DataFrame,patid = 'enrolid') -> pd.DataFrame:
    df['tfidf'] = df['frequency'] * np.log((len(df[patid].unique())/df['idf'])+1)
    return df
    
def get_idf(df:pd.DataFrame, patid = 'enrolid',event_column = 'event_real') -> pd.DataFrame:
    df1 = df.groupby([event_column]).count()[patid].reset_index().rename({patid:'idf'},axis = 1)
    return pd.merge(df1,df, on = event_column)

def prepreprocess_bipolar(df:pd.DataFrame, event_column = 'event_real',episode_column = 'episode', remove_no_therapy_events = True, mixed_to_manic = False, manic_to_mixed = True) -> pd.DataFrame:
    df['appropriate'] = df.apply(lambda x: 'inappropriate' if ('ANTIDEPRESSANT' in x[event_column]) and (x[episode_column] == 'Manicw/woPsych' or x[episode_column] == 'Mixedw/woPsych') else 'appropriate',axis = 1)
    if remove_no_therapy_events:
        df = df[df[event_column].str.strip() != 'No Therapy:']
    if manic_to_mixed:
        df.loc[:,event_column] = df.loc[:,event_column].str.replace('Manic','Mixed')
    elif mixed_to_manic:
        df.loc[:,event_column] = df.loc[:,event_column].str.replace('Mixed','Manic')
    
    return df

def patients_per_cluster(df:pd.DataFrame,cluster_col:str,patid = 'enrolid') -> pd.Series:
    return df.groupby(cluster_col)[patid].unique().str.len().sort_values(ascending = False)

def reweight(df:pd.DataFrame, event_column = 'event_real',patid = 'enrolid') -> pd.DataFrame:
    
    #Drop previous cols
    df.drop(['freq_total','freq_total_lag','w_occ','w_occ_lag'],axis = 1,inplace = True)

    #Event frequencies total
    df = get_event_frequencies_as_column(df,[event_column],'freq_total')

    # Get Lag of total frequency for each event
    df = freq_total_lag(df,patid)

    # Create weighted order by number of events that patients have with 
    df = weighted_order(df,patid,event_column,0)

    # Create Weighted order lag column by 1
    df = weighted_order_lag(df,patid)
    return df

def write_all_clusts_to_csv(df:pd.DataFrame, event_column: str, remove_threshold = .0005):
    
    for clust in df.Clusters.unique():
        to_df = df[df['Clusters'] == clust]
        thresh = len(to_df)*remove_threshold
        print(f'clust{clust}')
        uniques = [(i,len(j)) for i,j in to_df.groupby([event_column]) if len(j) > thresh]
        to_keep = pd.DataFrame(uniques, columns = ['event','count'])
        print(f'before {len(to_df)}, uniques = {len(to_df.event_real.unique())}')
        to_df = to_df[to_df['event_real'].isin(to_keep.event)]
        print(f'after {len(to_df)}, uniques = {len(uniques)}')
        to_df.to_csv(f"clust{clust}tpsum.csv",index = False)
        
        

def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df
        








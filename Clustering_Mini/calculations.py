import pandas as pd
import numpy as np

def sum_transition_score(df:pd.DataFrame,cluster_col:str,event_column = 'event_real', patid = 'enrolid') -> pd.DataFrame:
    ## Calculate transition score per cluster/occurrence
    a = num_people_per_group(df,[cluster_col,'occ'],event_column).reset_index()
    a['occ'] = a['occ'] + 1
    a['transitionscore'] = (a[event_column]) /(a['occ'])
    
    ## Add up transition scores per cluster
    b = a.groupby(cluster_col).sum().transitionscore.reset_index().rename({'transitionscore':'Sum_transition_score_all_clusters'},axis = 1)
    
    ## Merge back with original df
    c = pd.merge(b,num_people_per_group(df,[cluster_col],patid).sort_values(ascending = False).reset_index(),how = 'left',on = cluster_col)
    return c

def summary_scores(df:pd.DataFrame,cluster_col:str,event_column = 'event_real',patid = 'enrolid') -> pd.DataFrame:
    c = sum_transition_score(df,cluster_col,event_column,patid)

    # Create and count unique treatment patterns within each cluster
    d = df.groupby([cluster_col,patid])[event_column].apply(lambda x:', '.join(x)).reset_index().rename({event_column:'tp'},axis = 1).groupby([cluster_col,'tp']).count()
    d['unique_tps'] = 1
    
    # Sum Unique Treatment Patterns for each cluster
    d = d.groupby(cluster_col).sum().unique_tps
    
    ## Merge back with original df
    e = pd.merge(c,d,how = 'left', on = cluster_col)
    e['FinalScore'] = e[patid]/(e['Sum_transition_score_all_clusters'])

    return e
  
def avg_transition(df:pd.DataFrame,cluster_col:str,event_column = 'event_real',patid = 'enrolid', threshold = .5, occ_col = 'occ') -> float:
  '''Calculate transition score per cluster/occurrence, transition limit = #events types / position, default is the mean transition score across all clusters, if threshold is median, then limit will be the median transition score out of all clusters and if it is a decimal between 0 and 1, it will be the normalized transition limit between the minimum transition score possible and the maximum transition score possible'''

  a = num_people_per_group(df,[cluster_col,'occ'],event_column).reset_index()
  a['occ'] = a['occ'] + 1
  a['transitionscore'] = (a[event_column]) /(a['occ'])

  minimum = _calc_sum(df.groupby([cluster_col,patid])[event_column].count().min(),1)
  maximum = _calc_sum(df['occ'].max(),len(df[event_column].unique()))
  sum_ = a.groupby(cluster_col).sum()
  mean = sum_.mean().transitionscore
  median = sum_.median().transitionscore
  std = sum_.std().transitionscore

  if isinstance(threshold,float):
      avg = (maximum - minimum) * threshold
      return avg
  elif threshold == 'median':
      return median
  elif threshold == 'min':
      return min(median,mean)
  elif threshold == 'max':
      return max(median,mean)
  elif threshold == 'std':
      return mean - ((std)/2)
  else:
      return mean
    
def _calc_sum(series_length: int, numerator:int) -> float:
    '''Used for calculating the minimum and maximum transition limits, numerator is 1 for minimum since best possible scenario 
       would be 1 for each position, transition limit = #events types / position'''
    return sum(numerator/(np.arange(series_length)+1))
  
def summary_scores_aggregated(df:pd.DataFrame,cluster_col:str,event_column = 'event_real',patid = 'enrolid', fin_score_metric = 'median', fin_score_column = 'FinalScore') -> float:
    return df.agg({'sum','mean','median','std','max','min'}).drop(cluster_col,axis = 1).rename({patid:'patients_per_cluster'},axis = 1).loc[fin_score_metric,fin_score_column]

def num_people_per_group(df:pd.DataFrame,group_cols:list,id_col:str) -> pd.Series:
  '''gets the number of people (id_col) per each group (group_cols)''' 
  return df.groupby(group_cols)[id_col].unique().str.len()

def final_prep_calc(df:pd.DataFrame,groupby_col:str,score_col:str,patid = 'enrolid') -> pd.DataFrame:
    return df.groupby([patid,groupby_col])[score_col].aggregate('sum').unstack().fillna(0).reset_index().reset_index().rename({'index':'Index'},axis = 1).set_index('Index')
    
def my_func1(episode_val:int,episode_val_lag:pd.Series, occurrence:int, discount_rate:float, freq_multiplier:float,slope:float,ascending = False,discount_type = 'exponential',clust_num = 5) -> float:
    '''Score equation options'''
    if not ascending:
        discount_rate = -discount_rate

    if discount_type == 'exponential':
        return ( episode_val + episode_val_lag)  / (((occurrence) ** (-discount_rate/clust_num)))
    elif discount_type == 'constant':
        return episode_val + episode_val_lag
    elif discount_type == 'logarithmic':
        return (episode_val + episode_val_lag) * np.log(occurrence + 2) ** (discount_rate/clust_num)
    elif discount_type == 'linear':
        return ((freq_multiplier * episode_val) / (-discount_rate * (occurrence + 1)))
    elif discount_type == 'sqrt':
        return (((occurrence + 1) ** discount_rate) * np.sqrt(freq_multiplier * episode_val)) + 100
    else:
        raise AssertionError('Invalid discount_type: options are "exponential","constant", "logarithmic", "linear", or "sqrt"')


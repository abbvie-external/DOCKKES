import pandas as pd
import numpy as np
import multiprocessing as mp
import sys
import psutil
import os

def remove_col(df:pd.DataFrame,col:str) -> pd.DataFrame:
    '''Remove a column from a DataFrame in place and return the dataframe'''
    if col in df:
        df.drop(col,axis = 1,inplace = True)
    return df
  
def get_cpu_num(cpus = None) -> int:
    '''Check how many cpus are available and take 3/4 of them for use if cpus is not given'''
    if not cpus:
        return int(round(mp.cpu_count()- (mp.cpu_count()/2),0))
    else:
        return cpus
    
def get_batch_step(dfs:list,cluster_range,init_search_iters) -> int:
    avail = psutil.virtual_memory().available
    max_mem = max([sys.getsizeof(df) for df in dfs])
    clusters = (cluster_range[1] - cluster_range[0]) + 1
    batch = min(int(round(avail/(max_mem),0)),clusters)
    #print(f'AVAILABLE MEM:{round(avail/(1024*1024),3)} MB\nMAXMEM:{round(max_mem/(1024*1024),3)} MB\nCLUSTERS:{clusters}\nBATCH SIZE: {batch}')
    return batch
  
def get_max_step(df:pd.DataFrame, column = 'Clusters') -> int:
    '''Look at all columns and find the farthest clustering that has been performed. All columns start with Clusters followed by a number'''
    max_step = 0
    for col in df.columns:
        if col != column and col.startswith(column) and int(col[len(column):]) > max_step:
            max_step = int(col[len(column):])
    return max_step
  
def print_warning(freq_multiplier:float,exponent:int,clusters:int) -> list:
    if freq_multiplier < 1:
        #print(f'WARNING: Frequency multiplier must be <= 1, changing {freq_multiplier} to 1 for this iteration')
        freq_multiplier = 1
    elif exponent > 6*clusters:
        #print(f'WARNING: Exponent must be <= 6, changing {exponent} to 6 for this iteration')
        exponent = 6*clusters
    return [freq_multiplier,exponent]
  
def get_appropriate_step(df:pd.DataFrame) -> float:
    step = 1
    num_cols = len([c for c in df.columns.values if 'score' in c])
    total = 0
    while total < 120 and step > 0:
        step -= .1
        total = num_cols * len(list(np.arange(0,5,step)))
    return step
  
def gen(step: int, start: int, stop:int, params: list) -> list: 
    i= start
    while i <= stop:
        if (i + step)-1 > stop:
            step = (stop - i) + 1
        yield [[i + j for j in range(step)]] + [[item] * step for item in params]
        i += step
        
def gen_breakdown(step:int,params:list,dfs:list) -> list:
    start = 0
    while start <= len(dfs):
        if (start + step)-1 >= len(dfs):
            step = (len(dfs) - start)
            print(step)
        yield [dfs[start:(start + step)]] +  [[item] * step for item in params]
        start += step
        
def positive_or_negative(total: float, fin_score_column = 'FinalScore') -> float:
    
    if fin_score_column == 'FinalScore':
        return total
    else:
        return -total
      
def get_final_score_metric(fin_score_column = 'FinalScore') -> str:
    if fin_score_column == 'FinalScore':
        return 'min'
    else:
        return 'max'
      

import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import silhouette_score,davies_bouldin_score
from preprocess import patients_per_cluster


def package_results(df:pd.DataFrame, kwargs:dict, patid = 'enrolid', event_column = 'event_real', cluster_col = 'Clusters',root = 'test',end = 0,start = 0, parent_dir = 'results'):
    
    root = name_file_prefix(kwargs)
    
    parent_dir = parent_dir
      
    directory = root
    # path 
    path = os.path.join(parent_dir, directory) 
    try: 
        os.makedirs(path,exist_ok = True) 
    except OSError as error: 
        print(error) 
        
    patients_per_cluster(df,cluster_col,patid).to_csv(path + '/patients_per_cluster.csv')
    prepped = df.groupby([patid, event_column])['score_fin'].aggregate('sum').unstack().fillna(0).reset_index().drop(patid,axis = 1)
    ids = df[[patid,cluster_col]].reset_index().drop('index',axis = 1).drop_duplicates()[cluster_col]
    df = pd.concat([summary_scores(df,'Clusters',event_column,patid)], keys=['DOCKKES'], names=['Model']).sort_index(level = 1,ascending = False)
    df['silhouette_avg'] = silhouette_score(prepped.to_numpy(),ids,sample_size = 10000)
    df['davies_bouldin'] = davies_bouldin_score(prepped.to_numpy(),ids)
    df['time_elapsed'] = end-start
    df.to_csv(path + '/quality_metrics.csv')
    if 'df' in kwargs.keys():
        del kwargs['df']
       
    with open(path + "/input_parameters.json", "w") as outfile:
        json.dump(kwargs, outfile)
    

def name_file_prefix(kwargs:dict) -> str:
    '''
    #c = clusters x to y
    #e = max exponent
    #s = max_slope
    #disc = discount type
    #i = maxiter
    #is = init_search_iters
    #h = hierarchy steps
    #th = threshold
    #solv = solver
    #clusmeth = clustering method
    '''
    s = ''
    for i,j in kwargs.items():
        if i == 'cluster_range':
            s+= f"c{j[0]}to{j[1]}"
        elif i == 'max_exponent':
            s+= f"_e{j}"
        elif i == 'max_slope':
            s += f"_s{j}"
        elif i == 'discount_type':
            s += f"_disc{j[0:3]}"
        elif i == 'maxiter':
            s += f"_i{j}"
        elif i == 'init_search_iters':
            s+= f"_is{j}"
        elif i == 'hierarchy_steps':
            s+= f"_h{j}"
        elif i == 'threshold':
            s+= f"_th{j}"
        elif i == 'solver':
            s+= f"_solv{j}"
        elif i == 'transition_setting_clusters':
            s+= f"_tsetc{j}"
        elif i == 'final_score_column':
            s+= f"_finscorecol{j}"
        elif i == 'clust_method':
            s+= f"_clusmeth{j}"
            
    return s + '_'
        
   # return f"c{kwargs['cluster_range'][0]}to{kwargs['cluster_range'][1]}_e{kwargs['max_exponent']}_s{kwargs['max_slope']}_disc{kwargs['discount_type'][0:3]}_i{kwargs['maxiter']}_is{kwargs['init_search_iters']}_h{kwargs['hierarchy_steps']}_th{kwargs['threshold']}_solv{kwargs['solver']}_tsetc{kwargs['transition_setting_clusters']}_finscorecol{kwargs['fin_score_column']}_clusmeth{kwargs['clust_method']}_"

def collect_results_from_runs(directory= 'results'):
    # iterate over files in
    # that directory
    quality = []
    patients = []
    
    for d in os.listdir(directory):
        f = os.path.join(directory, d)
        # checking if it is a file
        for file in os.listdir(f):
            read_file = pd.read_csv(os.path.join(f, file))
            read_file['Model'] = d
            read_file.rename({'Unnamed: 1':'metrics'},axis = 1,inplace = True)
            if 'quality' in file:
                quality.append(read_file)
            elif 'patients' in file:
                patients.append(read_file)
    return pd.concat(quality),pd.concat(patients)

def summary_scores(df:pd.DataFrame,cluster_col:str,event_column = 'event_real',patid = 'enrolid', medians = False) -> pd.DataFrame:
    ## Calculate transition score per cluster/occurrence
    a = df.groupby([cluster_col,'occ'])[event_column].unique().str.len().reset_index()
    a['occ'] = a['occ'] + 1
    a['transitionscore'] = (a[event_column]) /(a['occ'])

    b = a.groupby(cluster_col).sum().transitionscore.reset_index().rename({'transitionscore':'Sum_transition_score_all_clusters'},axis = 1)
    
    ## Merge back with original df
    c = pd.merge(b,df.groupby(cluster_col)[patid].unique().str.len().sort_values(ascending = False).reset_index(),how = 'left',on = cluster_col)

    if medians:
        medians = (df.groupby(cluster_col)['occ'].median() + 1).reset_index().rename({'occ':'median_occ'},axis = 1)
        d = pd.merge(df,medians,how = 'left',on = cluster_col)
        df = d[d['occ']<d['median_occ']]
        
    d = df.groupby([cluster_col,patid])[event_column].apply(lambda x:', '.join(x)).reset_index().rename({event_column:'tp'},axis = 1).groupby([cluster_col,'tp']).count()
    d['unique_tps'] = 1
    
    # Sum Unique Treatment Patterns for each cluster
    d = d.groupby(cluster_col).sum().unique_tps
    
    ## Merge back with original df
    e = pd.merge(c,d,how = 'left', on = cluster_col)
    e['FinalScore'] = e[patid]/(e['Sum_transition_score_all_clusters'])
    
    e['people_per_unique_tp'] = e[patid] / e['unique_tps']
    return e.agg({'sum','mean','median','std','max','min'}).drop(cluster_col,axis = 1).rename({patid:'patients_per_cluster'},axis = 1).T[['sum','min','max','median','mean','std']].round(2)


def rank_silhouette_or_db(df:pd.DataFrame,col:str, ascending = False, norm_fn = None) -> pd.DataFrame:
    t = df.groupby(['Model'])[col].mean().to_frame()
    t[col + '_rank'] = t.rank(method = 'min',ascending = ascending)
    t[col + '_norm'] = norm_fn(t[col])
    t.reset_index(inplace = True)
    t.drop(col,axis = 1,inplace = True)
    return pd.merge(df,t,on = 'Model', how = 'left')

def split_model_params(df:pd.DataFrame) -> pd.DataFrame:
    model_params = ['cluster_range','max_exponent','max_slope','discount','maxiter','init_search_iters','hierarchy_steps','threshold','solver','transition_setting_clusters','fin_score_col','clust_meth']
    n = df.Model.str.split('_',n = len(model_params), expand = True)
    for i,j in zip(model_params,n):
        df[i] = n[j]
        if i == 'transition_setting_clusters':
            df.loc[:,i] = df.loc[:,i].fillna('tsetc(7, 14)').replace({'':'tsetc(7, 14)'})
        elif i == 'fin_score_col':
            df.loc[:,i] = df.loc[:,i].fillna('finscorecolFinalScore')
        elif i == 'clust_meth':
            df.loc[:,i] = df.loc[:,i].fillna('clusmethkmeans')
            
    return df


def gen_model_param_analysis_file(location:str,name:str):
    '''Create the model param analysis file that is used to compare all hyperparameter sets'''
    x,y = collect_results_from_runs(location)  
    neg_norm = lambda x: (x - min(x)) / (max(x) - min(x))
    norm = lambda x: (x - max(x)) / (min(x) - max(x))
    df_1 = x[['Model','metrics','sum','median','mean','max','min', 'std']]
    df_2 = x[['Model','silhouette_avg','davies_bouldin','time_elapsed']]
    df_1 = df_1.pivot(index = 'Model',columns = 'metrics', values = ['sum','median','mean','max','min', 'std'])
    df_1.columns = [' '.join(col).strip() for col in df_1.columns.values]
    df_1.reset_index(inplace = True)

    for col in df_1.columns:
        if col != 'Model':
            if 'FinalScore' in col or ('patients_per_cluster' in col and 'max' not in col):
                df_1[col + '_norm'] = norm(df_1[col])
                df_1[col + '_rank'] = df_1[col].rank(method = 'min',ascending = False)
            else:
                df_1[col + '_norm'] = neg_norm(df_1[col])
                df_1[col + '_rank'] = df_1[col].rank(method = 'min',ascending = True)

    for col in df_2.columns:
        if col == 'silhouette_avg': 
            df_2 = rank_silhouette_or_db(df_2,col,ascending = False, norm_fn = norm)
        elif col == 'davies_bouldin':
            df_2 = rank_silhouette_or_db(df_2,col,ascending = True, norm_fn = neg_norm)
        elif col == 'time_elapsed':
            df_2[col] = pd.to_timedelta(df_2[col])
            df_2[col + '_rank'] = df_2[col].dt.total_seconds().rank(method = 'dense')

    df_2 = split_model_params(df_2)
    df_2.drop_duplicates(inplace = True)
    clusts = y.groupby('Model').Clusters.max().reset_index()
    clusts.loc[:,'Clusters'] = clusts.loc[:,'Clusters'] +1
    df_1 = pd.merge(df_1, clusts, on = 'Model', how = 'left')
    z = pd.merge(df_1,df_2,on = 'Model',how = 'left')
    z.to_csv('model_param_output_folder/' + name)
    
#gen_model_param_analysis_file('clouderatest','testoutput.py')
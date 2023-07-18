#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.optimize import minimize_scalar,minimize
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score,calinski_harabasz_score
from sklearn.model_selection import ShuffleSplit
from datetime import datetime
pd.set_option("display.min_rows", 101)
from pybobyqa import solve
from scipy.cluster.hierarchy import dendrogram
import warnings
from multiprocess import Pool
import multiprocessing as mp
warnings.filterwarnings("ignore")
from tqdm import tqdm
from p_tqdm import p_map
import faiss
from paramholder import LimitedParameters
faiss.omp_set_num_threads(1)
from util import print_warning, remove_col
from calculations import final_prep_calc, my_func1



def initial_search(c:int,df:pd.DataFrame,p:LimitedParameters,ascending:bool,discount_type:str,use_pca:bool,min_exponent = 1,max_exponent = 5,max_slope = 1, max_fun = 5, patid = 'enrolid', events_repeat_often = True,score_col = None):
    np.random.seed(0)
    if c > len(df[patid].unique()) - 1:
        return (LimitedParameters(),pd.DataFrame())
    return (p,solve(func_min, x0 = [(4 * c)-.000001,.05,1.2] , args = (c,df,p,ascending,discount_type,use_pca,False,patid,events_repeat_often,score_col),  bounds=(np.array([c,0.001,1]),np.array([max_exponent * c,max_slope,2.5])), 
        npt=None,rhobeg=.35, rhoend=1e-4, maxfun=max_fun, nsamples=None,user_params=None, objfun_has_noise=False,seek_global_minimum=False,scaling_within_bounds=True,do_logging=False, print_progress=False))

def cluster(df:pd.DataFrame,n:int,use_pca=False, plot = False, plot_type = '2d',p = None, patid = 'enrolid', model = 'kmeans', last = False) -> pd.DataFrame:
    
    if patid in df.columns:
        df1 = df.drop([patid],axis = 1)
    else:
        df1 = df
        
    df1 = np.ascontiguousarray(np.float32(df1.to_numpy()))
    
    D = df1.shape[1]
    kmeans = faiss.Kmeans(d=D, k=n,niter = 5, seed = 0)

    N_small = len(df1)
    kmeans.train(df1[:N_small])
    _, ids = kmeans.index.search(df1, 1)
    ids = ids.reshape(1,N_small)[0]
    del kmeans
    
    #Fit kmeans with data
    if use_pca:
        pca = PCA(n_components = 10, random_state = 0).fit_transform(df1)
        kmeans.train(pca[:N_small])
        if plot and plot_type == '2d': 
            _plot_pca_2d(pca,kmeans,df1,interactive = False, p = p)
        elif plot and plot_type == '3d':
            _plot_pca_3d(pca,kmeans,df1,interactive = False, p = p)
        df1 = pca
          
    #Save all labels from the kmeans
    labels = []
    for i, label in enumerate(ids):
        labels.append((label,i))


    # #Create dataframe with the cluster numbers
    clusters = pd.DataFrame(labels,columns = ['Clusters','Index'])

    # # Merge data with scores used to cluster
    clusters_n_scores = pd.merge(clusters,df,on = 'Index',how = 'inner')
    
    return clusters_n_scores[['Clusters','Index','enrolid']],ids
  
  


def func_min(params,c,df,p:LimitedParameters,ascending = False,discount_type = 'exponential',use_pca = False, last = False, patid = 'enrolid', events_repeat_often = True, score_col = None) -> float:
    
    exponent,slope,freq_multiplier = params
    freq_multiplier,exponent = print_warning(freq_multiplier,exponent,c)

    fin2,davies_bouldin,silhouette_avg,calinski_harabasz = score_and_cluster(df=df,exponent=exponent,clust_num=c,freq_multiplier=freq_multiplier,slope=slope,ascending=ascending,discount_type=discount_type,use_pca = use_pca,plot = False, p = p, events_repeat_often = events_repeat_often, score_col = score_col)

    clust_count_min = fin2.groupby('Clusters')[patid].unique().str.len().min()
    
    var_total = silhouette_avg  / (davies_bouldin+1)
    total =  var_total

    p.update(vari = abs(total),exp = exponent,clust = c,freq_multiplier = freq_multiplier,slope = slope,ascending = ascending,
             discount_type = discount_type,davies_bouldin = davies_bouldin,
             silhouette_avg = silhouette_avg,calinski_harabasz=calinski_harabasz,total = total,
             matthew_index = var_total,last_clust_count_min = clust_count_min)
            
    p.clust_metrics(fin2,c,exponent,freq_multiplier,slope,ascending,discount_type,
                            davies_bouldin,silhouette_avg,calinski_harabasz,total,var_total)
    if last:
        p.end_timer()
    return -total

def prep_for_cluster(df:pd.DataFrame,exponent:float,clust_num:int,freq_multiplier= 1.2,slope = .05, ascending = False,discount_type = 'exponential', patid = 'enrolid',event_column = 'event_real',event_or_eventtrans = 'eventtrans', score_col = None) -> pd.DataFrame:
    
    if event_or_eventtrans == 'eventtrans':
        groupby = 'Eventtrans'
    else:
        groupby = event_column
        
    #Filter for relavent columns
    filtered_columns = [patid,event_column,'freq_total','freq_total_lag','occ','w_occ','w_occ_lag',score_col]
    df1 = df[[col for col in filtered_columns if col is not None]]
    
    #Calculate columns for scoring
    df1['Eventlag'] = df1.groupby([patid])[event_column].shift(-1).fillna('')
    df1['Eventtrans'] = df1[event_column] + ',' + df1['Eventlag']
    df1['w_occ_fina'] = ((slope * (df1['w_occ'] + 1)) ** freq_multiplier)*(df1['occ'] + 1)
    df1['w_occ_finb'] = ((slope*(df1['w_occ_lag'] + 1))** freq_multiplier)*(df1['occ'] + .5)
    df1['w_occ_fin'] = df1['w_occ_fina'] + df1['w_occ_finb']
    df1['add_freqs'] = df1['freq_total'] + df1['freq_total_lag']
    df1['denominator'] = df1['w_occ_fin'] ** (exponent/clust_num)
    #Update original dataframe with calculated values
    df['w_occ_fina'] = df1.loc[:,'w_occ_fina']
    df['w_occ_finb'] = df1.loc[:,'w_occ_finb']
    df['w_occ_fin'] = df1.loc[:,'w_occ_fin']
    df['add_freqs'] = df1.loc[:,'add_freqs']
    df['denominator'] = df1.loc[:,'denominator']
    
    #Prep Dataframe by aggregating score by patient id and event then unstacking so each event is now a column
    if score_col is None:
        df1['score'] = my_func1(df1['freq_total'],df1['freq_total_lag'],df1['w_occ_fin'],exponent,freq_multiplier,slope,ascending,discount_type,clust_num)
        df1.loc[:,'score'] = df1.loc[:,'score'].apply(lambda x: 0 if x <= 0 else x)
        df.loc[:,'score'] = df1.loc[:,'score']
        return final_prep_calc(df1,groupby,'score',patid)
    else:
        df.loc[:,score_col] = df1.loc[:,score_col]
        return final_prep_calc(df1,groupby,score_col,patid)

def score_and_cluster(df:pd.DataFrame,exponent:float,clust_num:int,freq_multiplier= 1.2,slope = .05, ascending = False,discount_type = 'exponential',use_pca = False,plot = False, p = None,patid = 'enrolid',event_column = 'event_real', plot_type = '2d', last = False, step = None, events_repeat_often = True, score_col = None) -> pd.DataFrame:
    
    if events_repeat_often:
        event_or_eventtrans = 'eventtrans'
    else:
        event_or_eventtrans = event_column
        
    cluster_prepped = prep_for_cluster(df,exponent,clust_num,freq_multiplier,slope,ascending,discount_type,patid = patid,event_column=event_column,event_or_eventtrans = event_or_eventtrans, score_col = score_col)

    #Cluster Data
    clusts2,ids= cluster(cluster_prepped,clust_num,use_pca = use_pca,plot = plot, p = p,patid = patid,plot_type = plot_type, last = last)
        
    # Merge clusters with original data
    if 'Clusters' in df:
        df.drop('Clusters',axis = 1,inplace = True)
        
    clust_fin = pd.merge(clusts2,df, on = patid, how = 'right')
    
    #Validate Clusters
    davies_bouldin,silhouette_avg,calinski_harabasz = validate_clusters(cluster_prepped,ids,patid)
    
    if step is not None:
        clust_fin.rename({'Clusters':'Clusters' + str(step)},axis = 1,inplace=True)
 
    if last:
        return clust_fin#,davies_bouldin,silhouette_avg,calinski_harabasz, kmeans
    else:
        return clust_fin,davies_bouldin,silhouette_avg,calinski_harabasz


def validate_clusters(df:pd.DataFrame,ids:list,patid = 'enrolid') -> list:
    
    df = remove_col(df,patid)
    df1 = df.to_numpy()
    
    davies_bouldin = davies_bouldin_score(df1,ids)
    silhouette_avg = silhouette_score(df1,ids,sample_size = 10000, random_state = 0)
    calinski_harabasz = calinski_harabasz_score(df1,ids)
    
    return [davies_bouldin,silhouette_avg,calinski_harabasz]

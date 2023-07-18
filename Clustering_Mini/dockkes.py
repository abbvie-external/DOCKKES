#!/usr/bin/env python
# coding: utf-8

# ## Imports - Packages required - scipy,sklearn,matplotlib,numpy,pandas,pybobyqa
# #### !pip install Py-BOBYQA
# #### !pip3 install multiprocess
# #### !pip3 install p_tqdm
# #### use conda to install faiss with the cmd conda install -c pytorch faiss-cpu or !pip3 install faiss-cpu --no-cache

#!pip3 install --upgrade gap-stat[rust]


# #### !conda install tslearn

import pandas as pd
import numpy as np
import json
import os

pd.set_option("display.min_rows", 101)

import warnings
warnings.filterwarnings("ignore")

import faiss
from multiprocessing import freeze_support
from package_res import package_results
from paramholder import LimitedParameters,Parameters

from preprocess import *
from calculations import *
from util import *
from optimize import *
from cluster import *
from functools import reduce
from operator import add
from datetime import datetime


def is_stuck(prev:pd.DataFrame,df:pd.DataFrame,cluster_col:str,patid = 'enrolid',found_gems = False) -> bool:
    '''Checks clusters to see if they are the same groupings as the previous iteration. If so, then no progress is made and is_stuck = True'''
    if not found_gems:
      return num_people_per_group(prev,[cluster_col],patid).sort_values(ascending = False).tolist() == num_people_per_group(df,[cluster_col],patid).sort_values(ascending = False).tolist()
    else:
      return False

def found_all_gems(prev:pd.DataFrame,df:pd.DataFrame,cluster_col:str,patid = 'enrolid') -> bool:
    '''Checks dataframe to see if all good clusters have been found. If the dataframe is empty, or the previous run assigned all people then all "gems" have been found'''
    return len(df) == 0 or len(num_people_per_group(prev,[cluster_col],patid)) == 0 or len(num_people_per_group(df,[cluster_col],patid)) == 0

def split_clusters(df:pd.DataFrame, cluster_col:str, min_clust_size:int, patid = 'enrolid', in_one_column = False, score_col = None) -> pd.DataFrame:
    '''Once reweights have been tried and results are the same, algorithm will split remaining unsuccessful clusters'''
    df = group_small_clusters(df,cluster_col,min_clust_size,patid, in_one_column, score_col = 'score' + str(get_max_step(df,'score')) if score_col is None else score_col)
    return [df[df[cluster_col] == i].drop([j for j in df[df[cluster_col] == i].columns if j.startswith('Clusters') == True],axis = 1) for i in df[cluster_col].unique()]

def unworthy_clusters(df:pd.DataFrame, patid = 'enrolid', event_column = 'event_real', min_clust_size = 100, transition_limit = None) -> pd.DataFrame:

    max_step = get_max_step(df)
    cluster_col = 'Clusters' + str(max_step)
    temp = num_people_per_group(df,[cluster_col],patid).sort_values(ascending = False)
    s = sum_transition_score(df,cluster_col,event_column,patid)
    s = s[s['Sum_transition_score_all_clusters']> transition_limit] #Filter for clusters above the transition limit
    
    #find clusters above transiton limit or too small
    unworthy = temp.loc[(temp < min_clust_size)].index.tolist() + _clusters_list(s,cluster_col) 
    if len(unworthy) != 0:
        farthest_cluster_col = 'Clusters' + str(max_step+1)
        
        # Anything in the unworthy category label with the same label
        df[farthest_cluster_col] = np.where(df[cluster_col].isin(unworthy),max(unworthy),df[cluster_col])
    return df

def _clusters_list(df:pd.DataFrame,cluster_col:str) -> list:
    '''gets the list of the clusters within the dataframe'''
    return df[cluster_col].values.tolist()


def filter_for_clusters(df:pd.DataFrame, cluster_col:str, patid = 'enrolid', min_clust_size = 100, transition_limit = None, event_column = 'event_real') -> pd.DataFrame:
    temp = num_people_per_group(df,[cluster_col],patid).sort_values(ascending = False)
    small_clusts = temp.loc[(temp < min_clust_size)].index.tolist()
    s = sum_transition_score(df,cluster_col,event_column,patid)
    s = s[s['Sum_transition_score_all_clusters']> transition_limit]
    clust_list = _clusters_list(s,cluster_col)
    if len(clust_list) == 0 or all([i in small_clusts for i in clust_list]) or len(temp) == len(small_clusts):
        return pd.DataFrame()

    df = df[df[cluster_col].isin(small_clusts + clust_list)]
    df = remove_col(df,'Index')
    return df


def dockkes_clustering(df:pd.DataFrame,event_column:str,patid = 'enrolid',cluster_range = (3,6), max_exponent = 5,min_exponent = 1, min_slope = 1,max_slope = 1,ascending = False,discount_type = 'exponential',maxiter = 100,use_pca = False,p = None, plot = None, interactive = False, plot_type = '2d', init_search_iters = 5, cpus = None, hierarchy_steps = 5, min_clust_size = 100, threshold = 'mean', transition_limit = 0, enter_recursion = True, hierarchy_start = 1, transition_setting_clusters = (5,14),reduction_rate =0,solver = 'hyper', n_iter_no_change = 15, early_stop_tol = .005, times_reweight = 0, events_repeat_often = True, allow_reweighting = True,score_col = None,fin_score_column = 'FinalScore', comorbidities_df = None, clust_method = 'faiss') -> Parameters:
   # currently clustmethod only used for naming
    tried_everything = False
    p_overall = Parameters()
    cpus = get_cpu_num(cpus)
    conc = None
  
    # First iteration of Clustering
    if df.empty:
        return
    else:
        prev = perform_clustering(df,None,ascending,discount_type,use_pca,min_exponent,max_exponent,max_slope,init_search_iters,
                                     transition_setting_clusters,cpus,patid,
                                  event_column,maxiter,min_clust_size,0,plot,plot_type,solver,n_iter_no_change,
                                  early_stop_tol, times_reweight, events_repeat_often, score_col,fin_score_column)
        if prev is None:
            return pd.DataFrame()
        
        # Transition limit setting
        if transition_limit == 0:
            cluster_str_number = 'Clusters' + str(get_max_step(prev))
            transition_limit = avg_transition(prev,cluster_str_number,event_column,patid,threshold)  

    # Start Main Loop
    for step in range(hierarchy_start,hierarchy_steps):
        df = prev.copy(deep = True)
        transition_limit = reduce_transition_limit(transition_limit,reduction_rate,step)
        cluster_str_number = 'Clusters' + str(get_max_step(df))

        # returns an empty dataframe if there are no more low quality clusters or the low quality cluster is already too small
        df = filter_for_clusters(df,cluster_str_number,patid,min_clust_size = min_clust_size, transition_limit = transition_limit)
        found_all = found_all_gems(prev,df,cluster_str_number,patid)
        stuck = is_stuck(prev,df,cluster_str_number,patid, found_all)
            
        if found_all or (tried_everything and not enter_recursion):
           break
            
                
        elif (stuck and tried_everything and enter_recursion):
            #returns list of dfs where each df is a cluster
            dfs = split_clusters(df,cluster_str_number,min_clust_size,patid, in_one_column = False, score_col = None)
            df_storage = []
            for index,d in enumerate(dfs):
                returned_df,_ = dockkes_clustering(df = d,event_column = event_column,patid = patid,cluster_range = cluster_range, max_exponent = max_exponent,min_exponent = min_exponent, min_slope = min_slope,max_slope = max_slope,ascending = ascending,discount_type = discount_type,maxiter = maxiter, use_pca = use_pca,p = p, plot = plot, interactive = interactive, plot_type = plot_type, init_search_iters = init_search_iters, cpus = cpus, hierarchy_steps = hierarchy_steps, min_clust_size = min_clust_size, threshold = threshold, transition_limit = transition_limit,enter_recursion = False,hierarchy_start = step,reduction_rate=reduction_rate,solver=solver,n_iter_no_change=n_iter_no_change,early_stop_tol=early_stop_tol,times_reweight = times_reweight,events_repeat_often=events_repeat_often, allow_reweighting = allow_reweighting,score_col = score_col, fin_score_column=fin_score_column, clust_method = clust_method)
                if len(returned_df) != 0:
                    df_storage.append(returned_df)
                else:
                    continue
 
            conc = pd.concat(df_storage)
            break
        
        elif (stuck and not tried_everything and not allow_reweighting):
            tried_everything = True
            prev = perform_clustering(df,prev,ascending,discount_type,use_pca,min_exponent,max_exponent,max_slope,init_search_iters,
                                     cluster_range,cpus,patid,event_column,maxiter,min_clust_size,step,plot,
                                     plot_type,solver,n_iter_no_change,early_stop_tol,times_reweight,events_repeat_often,score_col,
                                      fin_score_column)

        elif (stuck and not tried_everything and allow_reweighting):
            tried_everything = True
            times_reweight +=1
            df = reweight(df,event_column,patid) # Reweight existing unsuccessful clusters, reweight located in preprocess.py
            prev = perform_clustering(df,prev,ascending,discount_type,use_pca,min_exponent,max_exponent,max_slope,init_search_iters,
                                     cluster_range,cpus,patid,event_column,maxiter,min_clust_size,step,plot,
                                     plot_type,solver,n_iter_no_change,early_stop_tol,times_reweight,events_repeat_often,score_col,
                                      fin_score_column)
            
        else:
            tried_everything = False
            prev = perform_clustering(df,prev,ascending,discount_type,use_pca,min_exponent,max_exponent,max_slope,init_search_iters,
                                     cluster_range,cpus,patid,event_column,maxiter,min_clust_size,step,plot,plot_type,
                                     solver,n_iter_no_change,early_stop_tol,times_reweight,events_repeat_often,score_col,
                                      fin_score_column)

    print('OVERALL ELAPSED TIME:\n')
    p_overall.end_timer()
    df = merge_cols_from_new_clustering(prev,conc,patid,event_column, final_merge = True) 
    df = collect_all_clusters_from_steps(df,patid,event_column,min_clust_size,transition_limit,cluster_range[1])
    df = group_small_clusters(df,'Clusters',min_clust_size,patid, in_one_column = True, score_col = get_max_step(df,'score') if score_col is None else score_col)
    
    return df,transition_limit


def _one_score_col(df:pd.DataFrame) -> bool:
    return len([c for c in df.columns if 'score' in c]) == 1

def main1(storage = 'results',**kwargs):
    start = datetime.now()
    print(f'START CLOCK: {start}\n________________________________\n')
    df,transition_limit = dockkes_clustering(**kwargs)
    df_copy = df.copy(deep = True)
    if _one_score_col(df):
        df.rename({'score0':'score_fin'},axis = 1,inplace = True)
        end = datetime.now()
        print(f'END CLOCK: {end}\nELAPSED TIME: {str(end - start)}')
        package_results(df,kwargs,kwargs['patid'],end = end,start = start, parent_dir = storage)
        
        return df
      
    df_copy.to_csv('beforecoefs.csv')
    
    print('CALCULATING OPTIMIZED SCORE COLUMN AFTER REWEIGHTS\n##################\n')
    df = coefficients_for_scores(df,kwargs['patid'],maxiter = kwargs['maxiter'], 
                                 n_iter_no_change = kwargs['n_iter_no_change'],
                                 early_stop_tol = kwargs['early_stop_tol'], events_repeat_often = kwargs['events_repeat_often'], 
                                 fin_score_column = kwargs['fin_score_column'])

    df = last_optimize2(df,kwargs['patid'],min_clust_size = kwargs['min_clust_size'], transition_limit = transition_limit, hierarchy_steps = kwargs['hierarchy_steps'],reduction_rate = kwargs['reduction_rate'], plot = kwargs['plot'],events_repeat_often = kwargs['events_repeat_often'])
    

    end = datetime.now()
    print(f'FINAL END CLOCK: {end}\nELAPSED TIME: {str(end - start)}')
    package_results(df,kwargs,kwargs['patid'],end = end,start = start, parent_dir = storage)
    return df

  
    
def forward_fill_scores(df:pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if 'score' not in c] + sorted([c for c in df.columns if 'score' in c])].ffill(axis=1)


def coefficients_for_scores(df_fin:pd.DataFrame,patid = 'enrolid',date_col = 'date', event_column = 'event_real',maxiter = 100, n_iter_no_change = 15, early_stop_tol = .005,events_repeat_often = True, fin_score_column = 'FinalScore', comorbidities_df = None) -> pd.DataFrame:
    patid = patid
    fin_score_column = fin_score_column
    event_column = event_column
    total_num_clusters = len(df_fin.Clusters.unique())
    df_fin = forward_fill_scores(df_fin)
    df_fin['Eventlag'] = df_fin.groupby([patid])[event_column].shift(-1).fillna('')
    df_fin['Eventtrans'] = df_fin[event_column] + ',' + df_fin['Eventlag']
    comorbidities_df = comorbidities_df
    if comorbidities_df is not None:
        df_fin = pd.merge(df_fin,comorbidities_df,how = 'left',on = patid)
    
    if events_repeat_often:
        event_or_eventtrans = 'Eventtrans'
    else:
        event_or_eventtrans = event_column
    
    def final_func(opt) -> float:
            
        df = df_fin.copy(deep = True)
        df = calc_score_fin(opt.para_dict,df)
        
        prepped = final_prep_calc(df,event_or_eventtrans,'score_fin',patid)
        if comorbidities_df is not None:
            prepped = pd.merge(prepped,comorbidities_df,on = patid,how = 'left').reset_index().rename({'index':'Index'},axis = 1).set_index('Index')

        #Cluster Data
        clusts2,ids= cluster(prepped,total_num_clusters)

        #Remove clusters col
        df = remove_col(df,'Clusters')
        
        # Merge clusters with original data
        clust_fin = pd.merge(clusts2,df, on = patid, how = 'right')
        
        # In the case that all items in the same cluster
        if len(clust_fin.Clusters.unique()) == 1:
            return 0
        else:
            sum_scores = summary_scores(clust_fin,'Clusters',event_column,patid)
            var_total = summary_scores_aggregated(sum_scores,'Clusters',event_column,patid,get_final_score_metric(fin_score_column),fin_score_column)

            return positive_or_negative(var_total,fin_score_column)

    search_space = {k: list(np.arange(0,5,step = get_appropriate_step(df_fin))) for k in [c for c in df_fin.columns.values if 'score' in c]}
    h = optimize_hyperactive(search_space, final_func, maxiter, early_stop_tol, n_iter_no_change)
    df = calc_score_fin(h.best_para(final_func),df_fin)
    return df
  

def last_optimize2(df:pd.DataFrame,patid = 'enrolid', min_clust_size = 50, transition_limit = 10,hierarchy_steps = 10,reduction_rate = 0, plot = False, events_repeat_often = True, comorbidities_df = None):
    norm = lambda x: (x - min(x)) / (max(x) - min(x))
    
    if len(df.Clusters.unique()) == 0:
      raise AssertionError(f'No Clusters were found that satisfied the min clust size of {min_clust_size}')
      
    best_df = None
    xs = []
    ys = []
    clusts = []
    #for clust in range(int(round(len(df.Clusters.unique())/2)), 2 * len(df.Clusters.unique())):
    for clust in range(len(df.Clusters.unique()), 2 * len(df.Clusters.unique())):
        df_copy = df.copy(deep = True)
        df_copy = get_final_df(df_copy, patid, clust,plot = plot, events_repeat_often = events_repeat_often, comorbidities_df = comorbidities_df)
        df_copy = group_small_clusters(df_copy,'Clusters',min_clust_size,patid,in_one_column = False,score_col = 'score_fin')
        df_copy['Clusters'].replace({j:i for i,j in enumerate(df_copy['Clusters'].unique())},inplace = True)
        sum_scores = summary_scores(df_copy,'Clusters','event_real',patid)
        var_total = summary_scores_aggregated(sum_scores,'Clusters','event_real',patid,'median','Sum_transition_score_all_clusters')
        fin_total = summary_scores_aggregated(sum_scores,'Clusters','event_real',patid,'mean','FinalScore')
        xs.append(var_total)
        ys.append(fin_total)
        clusts.append(clust)
        

    for _ in range(2):
        max_x = np.argmax(xs)
        del xs[max_x]
        del ys[max_x]
        del clusts[max_x]
    best_clust = clusts[np.argmin(norm(xs) - norm(ys))]

    df_copy = df.copy(deep = True)
    df_copy = get_final_df(df_copy, patid, best_clust,plot = plot, events_repeat_often = events_repeat_often, comorbidities_df = comorbidities_df)
    df_copy = group_small_clusters(df_copy,'Clusters',min_clust_size,patid,in_one_column = False,score_col = 'score_fin')
    df_copy['Clusters'].replace({j:i for i,j in enumerate(df_copy['Clusters'].unique())},inplace = True)
            
    return df_copy



def similarity_euclid_score(df:pd.DataFrame,clust_col = 'Clusters',patid = 'enrolid',score_col = 'score_fin') -> pd.DataFrame:
    
    df[score_col] = df[score_col].astype(float)
    combos = []

    euclid_dist = lambda x,y: np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
    
    values = []
    mean_of_means = df.groupby([patid,clust_col])[score_col].sum().reset_index().groupby(clust_col)[score_col].mean()
    for index,a in enumerate(mean_of_means):   
        for index1,b in enumerate(mean_of_means):
            if index1!=index:
                combos.append((index,index1))
                values.append(-round(euclid_dist(b,a),0))
     
    l = pd.DataFrame(combos,values).reset_index()
    l.sort_values(by = [0,1],inplace = True)
    return l.pivot(index=0, columns=1, values='index') 

# Gets the closest cluster to the provided "original_clusters" which is a list assuming dataframe in format x by x 
def find_closest_clusters(df:pd.DataFrame, original_clusters:list) -> int:
    df = df[[c for c in df.columns if c not in original_clusters]].T
    return [df.loc[df[original_cluster] == df[original_cluster].max()].index[0] for original_cluster in original_clusters]


# Add all scores to each other after multiplying by weights
def calc_score_fin(opt:dict,df:pd.DataFrame) -> pd.DataFrame:
    for k,v in opt.items():
        if 'score' in k or 'comorb' in k:
            df.loc[:,k] = df.loc[:,k] * v
            df.loc[:,k] = df.loc[:,k].fillna(0)
            
    # add each score column to each other
    add_scores = reduce(add, [df[c] for c in df.columns.values if 'score' in c], 0)
    med_val = add_scores.median()
    mean_val = add_scores.mean()
    half_val = min(mean_val,med_val) 
    df['score_fin'] = reduce(add,[add_scores] + [df[c] * half_val for c in df.columns.values if 'comorb' in c] )
    return df

def get_final_df(df:pd.DataFrame, patid = 'enrolid', clusters = 15, events_repeat_often = True, event_column = 'event_real', plot = False, comorbidities_df = None) -> pd.DataFrame:

    if events_repeat_often:
        event_or_eventtrans = 'Eventtrans'
    else:
        event_or_eventtrans = event_column
        
    prepped = final_prep_calc(df,event_or_eventtrans,'score_fin',patid)
    if comorbidities_df is not None:
        prepped = pd.merge(prepped,comorbidities_df,on = patid,how = 'left').reset_index().rename({'index':'Index'},axis = 1).set_index('Index')
        
    #Cluster Data
    clusts2,ids= cluster(prepped,clusters, plot = plot)
        
    #Remove clusters col
    df = remove_col(df,'Clusters')
       
    # Merge clusters with original data
    clust_fin = pd.merge(clusts2,df.drop(['Index'],axis = 1), on = patid, how = 'right')
    return clust_fin
  
                           
def reduce_transition_limit(transition_limit:float,reduction_rate:float, step:int) -> float:
    assert reduction_rate < 1
    if step == 1:
        return transition_limit
    return transition_limit - (transition_limit * reduction_rate)

def perform_clustering(df:pd.DataFrame,prev:pd.DataFrame,ascending:bool,discount_type:str,use_pca:bool,min_exponent = 1,max_exponent = 5,max_slope = 1,init_search_iters = 5, cluster_range = (5,10),cpus = 2, patid = 'enrolid',event_column = 'event_real',maxiter = 100, min_clust_size = 20, step = None, plot = False, plot_type = '2d', solver = 'hyper', n_iter_no_change = 15, early_stop_tol = .005, times_reweight = 0,events_repeat_often = True,score_col = None, fin_score_column = 'FinalScore') -> pd.DataFrame:

    if len(df[patid].unique()) < min_clust_size or len(df[patid].unique()) < cluster_range[0]:
        return    
     
    df,finp = multiprocess_clusters(df,ascending,discount_type,use_pca,
                                     min_exponent,max_exponent,max_slope,init_search_iters,
                                     cluster_range,cpus,patid,maxiter,min_clust_size,
                                     solver, n_iter_no_change, early_stop_tol,events_repeat_often,score_col,fin_score_column)
    
    finp = optimize(df,finp,discount_type,max_exponent,max_slope,use_pca,ascending,maxiter,patid,solver,n_iter_no_change,early_stop_tol,events_repeat_often,score_col,fin_score_column)
    
    fin2 = score_and_cluster(df,finp.max_exponent,finp.max_cluster,
                                     finp.max_freq_multiplier,finp.max_slope,finp.ascending,
                                     finp.discount_type,use_pca = use_pca, plot = plot, p = finp,
                                     plot_type = plot_type, last = True,step = step, times_reweight = times_reweight,
                                     events_repeat_often = events_repeat_often, score_col = score_col, validate_clusters = False)
    
    d = {'exp':finp.max_exponent,'clust':finp.max_cluster,'slope':finp.max_slope,'freq_mult':finp.max_freq_multiplier,'df':df,'finp':finp,
        'asc':finp.ascending,'disc_type':finp.discount_type,'use_pca':use_pca,'last':True,'patid':patid, 
        'events_repeat_often':events_repeat_often, 'score_col':score_col, 'fin_score_column':fin_score_column}
    func_minl(d)
    
    return update_clusts_and_scores(prev,fin2,step,patid,event_column)


def update_clusts_and_scores(prev:pd.DataFrame,fin2:pd.DataFrame,step:int,patid = 'enrolid',event_column = 'event_real') -> pd.DataFrame:
    
    if step == 0 or prev is None:
        return fin2
    else:
        clusts = fin2[[patid,'Clusters'+str(step)]].drop_duplicates()
        
        # Merge fin2 with Prev dataframe (saved outside of loop) and overwrite prev
        prev = pd.merge(prev,clusts,on = [patid],how = 'left')
        return merge_cols_from_new_clustering(prev,fin2,patid,event_column)

def merge_cols_from_new_clustering(prev:pd.DataFrame,df:pd.DataFrame,patid = 'enrolid',event_column = 'event_real', final_merge = False) -> pd.DataFrame:
    if df is None:
        return prev
    
    filt_cols = [patid,event_column,'occ','w_occ_fina','w_occ_finb', 'w_occ_fin', 'add_freqs', 'denominator'] + [c for c in df.columns if 'score' in c]
    if final_merge:
        df.rename({j:'Clusters' + str(get_max_step(prev) + 1 + i) for i,j in enumerate(df.columns) if j.startswith('Clusters')},axis = 1,inplace = True)
        filt_cols += [i for i in df.columns if i.startswith('Clusters')]
    scores = df[filt_cols]
    prev = pd.merge(prev,scores,on = [patid,event_column,'occ'], how = 'left',suffixes=(None, '_1'))
    return  update_cols(prev,['w_occ_fina','w_occ_finb', 'w_occ_fin', 'add_freqs', 'denominator'] + [c for c in df.columns if 'score' in c],suffix = '_1')


def update_cols(df:pd.DataFrame,cols:list,suffix = '_1') -> pd.DataFrame:
    for col in cols:
        if col + suffix in df.columns:
            df.loc[:,col] = np.where(pd.notnull(df[col + suffix]),df[col + suffix],df[col])
    df.drop([col + suffix for col in cols if col + suffix in df.columns],axis = 1,inplace = True)
    return df


def collect_all_clusters_from_steps(df:pd.DataFrame, patid = 'enrolid',event_column='event_real', min_clust_size = 10,transition_limit = None,max_clusters_per_round = 10) -> pd.DataFrame:
    df = unworthy_clusters(df,patid,event_column,min_clust_size,transition_limit)
    for col in df.columns:
        if col.startswith('Clusters'):
            df[col] = (df[col].values + 1) * max_clusters_per_round * (int(col[8:])+1)
    df['Clusters'] = df.loc[:,df.columns.str.startswith('Clusters')].max(axis=1)
    df['Clusters'].replace({j:i for i,j in enumerate(df['Clusters'].unique())},inplace = True)
    df.sort_values(by = [patid,'occ'],inplace = True)
    
    return df

def group_small_clusters(df:pd.DataFrame,cluster_col:str,min_clust_size:int, patid = 'enrolid', in_one_column = True, score_col = 'score_fin') -> pd.DataFrame:
    
    small_clusts = _find_small_clusters(df,cluster_col,min_clust_size,patid)
    if len(small_clusts) != 0:
        if in_one_column:
            df[cluster_col] = np.where(df[cluster_col].isin(small_clusts),df[cluster_col].max()+1,df[cluster_col])
        else:
            dists = similarity_euclid_score(df,cluster_col,patid,score_col)
            closest_clusters = find_closest_clusters(dists,small_clusts)
            df['was_small'] = df[cluster_col].isin(small_clusts)
            df[cluster_col].replace({og:new for og,new in zip(small_clusts,closest_clusters)},inplace= True)
            
    return df

def _find_small_clusters(df:pd.DataFrame,cluster_col:str,min_clust_size:int,patid = 'enrolid') -> list:
    return df[df[cluster_col].isin(np.sort(df[cluster_col].unique())[num_people_per_group(df,[cluster_col],patid) < min_clust_size])][cluster_col].unique()

  
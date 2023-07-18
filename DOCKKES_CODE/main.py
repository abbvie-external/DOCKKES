#!/usr/bin/env python
# coding: utf-8


# ### DOCKKES (Divisive Optimized Clustering using Kernel KMeans on Event Sequences) 

# In[1]:
#!pip3 install Py-BOBYQA
#!pip3 install multiprocess
#!pip3 install p_tqdm
#!pip3 install faiss-cpu --no-cache
#!pip3 install hyperactive==3.3.3
#!pip3 install pandas==1.2.4
#!pip3 install psutil
#!pip3 install hyperactive

import os

if os.getcwd() != '/home/cdsw/Clustering/Clustering_Code/Clustering_Mini':
    os.chdir('Clustering/Clustering_Code/Clustering_Mini')
from dockkes import main1
from model import Dockkes
import pandas as pd
from sklearn.metrics import silhouette_score,davies_bouldin_score
import numpy as np
from package_res import name_file_prefix
import itertools


# ## Define Patient Event Sequences
patid = 'enrolid'
event_column = 'event_real'
min_clust_size = 10
#min_clust_size = .01 * num_seqs

cluster_ranges = [(4,20), (5,20), (6,14)]
max_exponents = [1.5,1.7,1.9,2]
max_slopes = [4,2]
transition_clusts = [(5,20),(14,30)]
hierarchy_steps = [20]

def save_to_final_clusterings(df:pd.DataFrame, name = ''):
    df.to_csv(f'final_clusterings/{name}')



def get_options(*args) -> list:
    '''Given lists of hyperparameters, produce a large list of combinations of those hyperparameter sets'''
    return list(itertools.product(*args))




if __name__ == '__main__':
    df_read = pd.read_csv('Input_Files/seqs_example.csv')
    #df_read = pd.read_csv('Input_Files/Skyrizi_Treatment_Patterns.csv')
    #Skyrizi_Treatment_Patterns.csv
      
    output_folder = 'example_output_folder'
  
    dks = Dockkes(df_read,'example_output_folder',10)
    
    #print(dks)
    print(dks.__dict__)
    
    
    
    results = main1(output_folder,**dks.return_method_cols())
    save_to_final_clusterings(results,'best_total_and_DB.csv')
    
#    kwargs = dict(df = df,
#                  event_column = event_column,
#                  patid = patid,
#                  cluster_range = (4,20), 
#                  max_exponent = 2,min_exponent = 1,
#                  min_slope = .0001,max_slope = 2,
#                  ascending = False,
#                  discount_type = 'exponential',
#                  maxiter = 35,
#                  use_pca = False,
#                  p = None, 
#                  plot = False, interactive = False, plot_type = '2d', 
#                  init_search_iters = 4,  
#                  hierarchy_steps = 20, 
#                  min_clust_size = 11827, 
#                  threshold = 'min', transition_limit = 0,transition_setting_clusters = (14,30),
#                  enter_recursion = True, 
#                  hierarchy_start = 1, 
#                  reduction_rate =0,
#                  solver = 'hyper', n_iter_no_change = 15, early_stop_tol = .005, 
#                  times_reweight = 0,
#                  events_repeat_often = False,
#                  score_col = None,
#                  fin_score_column = 'FinalScore',
#                 comorbidities_df = None,
#                 clust_method = 'faiss')
#
#    results = main1(**kwargs)
#    save_to_final_clusterings(results,'skyrizi_best_total_and_DB.csv')
  


#    
#    kwargs = dict(df = df,
#                  event_column = event_column,
#                  patid = patid,
#                  cluster_range = (4,8), 
#                  max_exponent = 2,min_exponent = 1,
#                  min_slope = .0001,max_slope = 4,
#                  ascending = False,
#                  discount_type = 'exponential',
#                  maxiter = 35,
#                  use_pca = False,
#                  p = None, 
#                  plot = False, interactive = False, plot_type = '2d', 
#                  init_search_iters = 4,  
#                  hierarchy_steps = 30, 
#                  min_clust_size = 250, 
#                  threshold = 'min', transition_limit = 0,transition_setting_clusters = (7,14),
#                  enter_recursion = True, 
#                  hierarchy_start = 1, 
#                  reduction_rate =0,
#                  solver = 'hyper', n_iter_no_change = 15, early_stop_tol = .005, 
#                  times_reweight = 0,
#                  events_repeat_often = False,
#                  score_col = None,
#                  fin_score_column = 'Sum_transition_score_all_clusters',
#                 comorbidities_df = None)
#    
#    results = main1(**kwargs)
#
#    #results = main(**kwargs)
#    results.to_csv('res1.csv')
    
    
    
#  #   For gathering all results
#    kwargs = dict(df = df,
#                  event_column = event_column,
#                  patid = patid,
#                  cluster_range = (4,8), 
#                  max_exponent = 2,min_exponent = 1,
#                  min_slope = .0001,max_slope = 1,
#                  ascending = False,
#                  discount_type = 'exponential',
#                  maxiter = 35,
#                  use_pca = False,
#                  p = None, 
#                  plot = False, interactive = False, plot_type = '2d', 
#                  init_search_iters = 4,  
#                  hierarchy_steps = 20, 
#                  min_clust_size = 1046, 
#                  threshold = 'min', transition_limit = 0,transition_setting_clusters = (7,14),
#                  enter_recursion = True, 
#                  hierarchy_start = 1, 
#                  reduction_rate =0,
#                  solver = 'hyper', n_iter_no_change = 15, early_stop_tol = .005, 
#                  times_reweight = 0,
#                  events_repeat_often = False,
#                  score_col = None,
#                  fin_score_column = 'FinalScore',
#                 comorbidities_df = None,
#                 clust_method = 'faiss')
#
#    options = get_options()
#    for opt in options:
#        cluster_range,max_exponent,max_slope,hierarchy_steps, transition_clusts = opt
#        print(f'HYPERPARAMS {opt}')
#        kwargs.update({'cluster_range': cluster_range,'max_exponent':max_exponent, 'max_slope':max_slope,'hierarchy_steps':hierarchy_steps, 'transition_setting_clusters':transition_clusts})
#        directory = os.path.join(output_folder, name_file_prefix(kwargs))
#        if os.path.isdir(directory) and len(os.listdir(directory))> 0:
#          print('Completed run already')
#          pass
#        else:
#          main1(storage = output_folder,**kwargs)

#
#

#
#print(get_options(cluster_ranges,max_exponents,max_slopes,hierarchy_steps,transition_clusts))
##print(get_options1())


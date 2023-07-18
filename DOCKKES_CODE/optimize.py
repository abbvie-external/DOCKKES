import pandas as pd
import numpy as np
from paramholder import LimitedParameters,Parameters
from util import print_warning, positive_or_negative, get_final_score_metric
from calculations import num_people_per_group, summary_scores, summary_scores_aggregated
from pybobyqa import solve
from hyperactive import Hyperactive 
from hyperactive.optimizers import RepulsingHillClimbingOptimizer, RandomAnnealingOptimizer, BayesianOptimizer
from cluster import score_and_cluster


def func_minl(opt) -> float:
    
    exponent = opt['exp']
    slope = opt['slope']
    freq_multiplier = opt['freq_mult']
    c = opt['clust']
    df = opt['df']
    p = opt['finp']
    ascending = opt['asc']
    discount_type = opt['disc_type']
    use_pca = opt['use_pca']
    last = opt['last']
    patid = opt['patid']
    events_repeat_often = opt['events_repeat_often']
    score_col = opt['score_col']
    fin_score_column = opt['fin_score_column']
    
    total,p = core_func_min(df,exponent,slope,freq_multiplier,c,p,ascending,discount_type,use_pca,last,patid,events_repeat_often,score_col,
                            fin_score_column)
    
    if last:
        p.end_timer()
    
    #optimizer tries to find the maximum, but the ideal transition score is the minimum, so we return - total
    return total

def func_min(params,c,df,p:Parameters,ascending = False,discount_type = 'exponential',use_pca = False, last = False, patid = 'enrolid', events_repeat_often = True,score_col = None) -> float:
    
    exponent,slope,freq_multiplier = params
    
    total,p = core_func_min(df,exponent,slope,freq_multiplier,c,p,ascending,discount_type,use_pca,last,patid,events_repeat_often,score_col)
    
    if last:
        p.end_timer()
    return -total
  

def core_func_min(df:pd.DataFrame,exponent:int,slope:float,freq_multiplier:float,c:int,p:Parameters,ascending = False,discount_type = 'exponential',use_pca = False, last = False, patid = 'enrolid', events_repeat_often = True, score_col = None,fin_score_column = 'FinalScore') -> list:
    
    freq_multiplier,exponent = print_warning(freq_multiplier,exponent,c)
    fin2,davies_bouldin,silhouette_avg,calinski_harabasz = score_and_cluster(df=df,exponent=exponent,clust_num=c,freq_multiplier=freq_multiplier,slope=slope,ascending=ascending,discount_type=discount_type,use_pca = use_pca,plot = False, p = p,events_repeat_often = events_repeat_often,score_col = score_col)
    
    clust_count = num_people_per_group(fin2,['Clusters'],patid)
 
    sum_scores = summary_scores(fin2,'Clusters','event_real',patid)
    var_total = summary_scores_aggregated(sum_scores,'Clusters','event_real',patid,
                                          get_final_score_metric(fin_score_column),fin_score_column)
    total =  positive_or_negative(var_total,fin_score_column)
    p.update(vari = abs(total),exp = exponent,clust = c,freq_multiplier = freq_multiplier,slope = slope,ascending = ascending,discount_type = discount_type,input_df = df,davies_bouldin = davies_bouldin,silhouette_avg = silhouette_avg,calinski_harabasz=calinski_harabasz,total = total,matthew_index = var_total,last_clust_count_min = clust_count.min())
    p.clust_metrics(fin2,c,exponent,freq_multiplier,slope,ascending,discount_type,
                        davies_bouldin,silhouette_avg,calinski_harabasz,total,var_total)
    return [total,p]
  
  
def optimize(df:pd.DataFrame,finp:Parameters,discount_type:str,max_exponent = 5,max_slope = 1,use_pca = False,ascending = False,maxiter = 100, patid = 'enrolid', solver = 'hyper', n_iter_no_change = 15, early_stop_tol = .005, events_repeat_often = True, score_col = None, fin_score_column = 'FinalScore') -> Parameters:
        
    score_col = score_col
    events_repeat_often = events_repeat_often
    fin_score_column = fin_score_column
    
    def func_mina(opt) -> float:

        exponent = opt['exp']
        slope = opt['slope']
        freq_multiplier = opt['freq_mult']
        p = finp
        c = finp.max_cluster
        ascending = finp.ascending
        discount_type = finp.discount_type
        last = False

        total,p = core_func_min(df,exponent,slope,freq_multiplier,c,p,ascending,discount_type,use_pca,last,patid,events_repeat_often,score_col,fin_score_column)

        #optimizer tries to find the maximum, but the ideal transition score is the minimum, if fin_score_column is FinalScore then return total because we want the max
        return total

    
    if solver == 'hyper':
  
        search_space = {'exp':list(range(finp.max_cluster, max(int(round(finp.max_cluster * max_exponent,0)),finp.max_cluster+1))),
                        'slope': list(np.arange(.001,max_slope,step = .05)),
                        'freq_mult':list(np.arange(1,2.5,.005))
                        }
    
        optimize_hyperactive(search_space, func_mina, maxiter, early_stop_tol, n_iter_no_change)
      
        
    elif solver == 'pybobyqa':
        res = solve(func_min, x0 = [finp.max_exponent * finp.max_cluster
                                ,finp.max_slope,finp.max_freq_multiplier] , args = (finp.max_cluster,df,finp,ascending,discount_type,use_pca,False,patid,events_repeat_often,score_col),  bounds=(np.array([finp.max_cluster,0.001,1]),np.array([max_exponent * finp.max_cluster,max_slope,2.5])), 
            npt=None,rhobeg=.35, rhoend=1e-4, maxfun=maxiter, nsamples=None,user_params = {'logging.save_diagnostic_info': False,'logging.save_poisedness':False, 'logging.save_xk':False}, objfun_has_noise=False,seek_global_minimum=True,scaling_within_bounds=True,do_logging=False, print_progress=False)
        
    else:
        raise AssertionError(f'Solver Expected to be one of "hyper" or "pybobyqa", you entered solver: {solver}')
        
    finp.final()
    return finp
  
  
def optimize_hyperactive(search_space:dict,opt_func, maxiter = 30, early_stop_tol = .005, n_iter_no_change = 10 ):
  h = Hyperactive(["progress_bar", "print_results", "print_times"], distribution = 'pathos')
  #h = Hyperactive(False, distribution = 'pathos')

  h.add_search(opt_func, search_space = search_space, n_iter = maxiter, optimizer = BayesianOptimizer(
  rand_rest_p=0.03,xi=0.03,
  warm_start_smbo=None), n_jobs = 1, max_score = None, early_stopping = {'tol_rel':early_stop_tol, 'n_iter_no_change':n_iter_no_change},random_state = 0, memory= False, memory_warm_start = None)

  h.add_search(opt_func, search_space = search_space, n_iter = maxiter, optimizer = RepulsingHillClimbingOptimizer(epsilon=0.05,
  distribution="normal",
  n_neighbours=3,
  rand_rest_p=0.03,
  repulsion_factor=3), n_jobs = 1, max_score = None, early_stopping = {'tol_rel':early_stop_tol, 'n_iter_no_change':n_iter_no_change},random_state = 0, memory= False, memory_warm_start = None)

  h.run()

  return h


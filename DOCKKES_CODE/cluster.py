import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score,calinski_harabasz_score
from plotfx import _plot_pca_2d
from util import remove_col, get_batch_step, gen
from tqdm import tqdm
from p_tqdm import p_umap
from initial_search_worker import initial_search
from paramholder import Parameters,LimitedParameters
from calculations import final_prep_calc, my_func1
import faiss
faiss.omp_set_num_threads(1)

def cluster(df:pd.DataFrame,n:int,use_pca=False, plot = False, plot_type = '2d',p = None, patid = 'enrolid', last = False, model = False) -> pd.DataFrame:
    
    if patid in df.columns:
        df1 = df.drop([patid],axis = 1)
    else:
        df1 = df
    
    #Initialize Kmeans
    kmeans = KMeans(n_clusters=n,random_state=0)
    
    #Fit kmeans with data
    if use_pca:
        pca = PCA(n_components = 1, random_state = 0).fit_transform(df1.to_numpy())
        kmeans.fit(pca)
        if plot and plot_type == '2d': 
            _plot_pca_2d(pca,kmeans,df1,interactive = False, p = p)
        elif plot and plot_type == '3d':
            _plot_pca_3d(pca,kmeans,df1,interactive = False, p = p)
        df1 = pca
        
    else:
        dists = kmeans.fit_transform(df1)
        df1 = df1.to_numpy()
        if plot and plot_type == '2d': 
            _plot_pca_2d(0,kmeans,df1,interactive = False, p = p)
        elif plot and plot_type == '3d':
            _plot_pca_3d(0,kmeans,df1,interactive = False, p = p)
        
    #Save all labels from the kmeans
    labels = []
    for i, label in enumerate(kmeans.labels_):
        labels.append((label,i))

    # #Create dataframe with the cluster numbers
    clusters = pd.DataFrame(labels,columns = ['Clusters','Index'])
    
    # Change to ints
    clusters.Clusters = clusters.Clusters.astype(int)

    # # Merge data with scores used to cluster
    clusters_n_scores = pd.merge(clusters,df,on = 'Index',how = 'inner')
    
    if model:
        return clusters_n_scores[['Clusters','Index',patid]],kmeans.labels_,kmeans
    else:
        return clusters_n_scores[['Clusters','Index',patid]],kmeans.labels_
    
def cluster_w_faiss(df:pd.DataFrame,n:int,use_pca=False, plot = False, plot_type = '2d',p = None, patid = 'enrolid', model = 'kmeans', last = False) -> pd.DataFrame:
    
    if patid in df.columns:
        df1 = df.drop([patid],axis = 1)
    else:
        df1 = df
        

#     np.random.seed(0)
    df1 = np.ascontiguousarray(np.float32(df1.to_numpy()))
    
    D = df1.shape[1]
    kmeans = faiss.Kmeans(d=D, k=n,niter = 10, seed = 0)

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
    
    #return clusters_n_scores[['Clusters','Index','enrolid']],kmeans,ids
    return clusters_n_scores[['Clusters','Index','enrolid']],ids 

def prep_for_cluster(df:pd.DataFrame,exponent:float,clust_num:int,freq_multiplier= 1.2,slope = .05, ascending = False,discount_type = 'exponential', patid = 'enrolid',event_column = 'event_real',groupby = 'eventtrans', times_reweight = 0,score_col = None) -> pd.DataFrame:
    
    #Filter for relavent columns
    filtered_columns = [patid,event_column,'freq_total','freq_total_lag','occ','w_occ','w_occ_lag',score_col]
    df1 = df.filter([col for col in filtered_columns if col is not None])

    #Calculate columns for scoring
    df1['Eventlag'] = df1.groupby([patid])[event_column].shift(-1).fillna('')
    df1['Eventtrans'] = df1[event_column] + ',' + df1['Eventlag']
    df1['w_occ_fina'] = ((slope * (df1['w_occ'] + 1)) ** freq_multiplier)*(df1['occ'] + 1)
    df1['w_occ_finb'] = ((slope*(df1['w_occ_lag'] + 1))** freq_multiplier)*(df1['occ'] + .5)
    df1['w_occ_fin'] = df1['w_occ_fina'] + df1['w_occ_finb']
    df1['add_freqs'] = df1['freq_total'] + df1['freq_total_lag']
    df1['denominator'] = df1['w_occ_fin'] ** (exponent/clust_num)
    df['w_occ_fina'] = df1.loc[:,'w_occ_fina']
    df['w_occ_finb'] = df1.loc[:,'w_occ_finb']
    df['w_occ_fin'] = df1.loc[:,'w_occ_fin']
    df['add_freqs'] = df1.loc[:,'add_freqs']
    df['denominator'] = df1.loc[:,'denominator']
    
    #Prep Dataframe by aggregating score by patient id and event then unstacking so each event is now a column
    if score_col is None:
        df1['score' + str(times_reweight)] = my_func1(df1['freq_total'],df1['freq_total_lag'],df1['w_occ_fin'],exponent,freq_multiplier,slope,ascending,discount_type,clust_num)
        df1.loc[:,'score'+ str(times_reweight)] = df1.loc[:,'score' + str(times_reweight)].apply(lambda x: 0 if x <= 0 else x)
        #Update original dataframe with calculated values
        if 'score' + str(times_reweight) in df.columns:
            df.loc[:,'score'+ str(times_reweight)] = df1.loc[:,'score'+ str(times_reweight)]
        else:
            df['score' + str(times_reweight)] = df1.loc[:,'score'+ str(times_reweight)]
        return final_prep_calc(df1,groupby,'score'+ str(times_reweight),patid)
    else:
        df[score_col] = df.loc[:,score_col]
        return final_prep_calc(df1,groupby,score_col,patid)
      
      
def score_and_cluster(df:pd.DataFrame,exponent:float,clust_num:int,freq_multiplier= 1.2,slope = .05, ascending = False,discount_type = 'exponential',use_pca = False,plot = False, p = None,patid = 'enrolid',event_column = 'event_real', plot_type = '2d', last = False, step = None, times_reweight = 0, events_repeat_often = True, score_col = None, validate_clusters = False) -> pd.DataFrame:
    
    if events_repeat_often:
        event_or_eventtrans = 'eventtrans'
    else:
        event_or_eventtrans = event_column
    
    cluster_prepped = prep_for_cluster(df,exponent,clust_num,freq_multiplier,slope,ascending,discount_type,patid = patid,event_column=event_column, groupby = event_or_eventtrans, times_reweight = times_reweight, score_col = score_col)

    #Cluster Data
    try:
        clusts,ids= cluster_w_faiss(cluster_prepped,clust_num,use_pca = use_pca,plot = plot, p = p,patid = patid,plot_type = plot_type, last = last)
    except:
        clusts,ids= cluster(cluster_prepped,clust_num,use_pca = use_pca,plot = plot, p = p,patid = patid,plot_type = plot_type, last = last)
   
    #Remove clusters col
    df = remove_col(df,'Clusters')
    
    # Merge clusters with original data  
    clust_fin = pd.merge(clusts,df, on = patid, how = 'right')
    
    if validate_clusters:
        #Validate Clusters
        davies_bouldin,silhouette_avg,calinski_harabasz = validate_clusters(cluster_prepped,ids,patid)

    if step is not None:
        clust_fin.rename({'Clusters':'Clusters' + str(step)},axis = 1,inplace=True)
 
    if last:
        return clust_fin
    elif last and validate_clusters:
        return clust_fin,davies_bouldin,silhouette_avg,calinski_harabasz
    else:
        return clust_fin,0,0,0

def validate_clusters(df:pd.DataFrame,ids:list,patid = 'enrolid') -> list:
    '''Functions used for validating Cluster results, currently davies bouldin, silhouette average and calinski harabasz'''
    df = remove_col(df,patid)
    df1 = df.to_numpy()
    
    davies_bouldin = davies_bouldin_score(df1,ids)
    silhouette_avg = silhouette_score(df1,ids,sample_size = 10000, random_state = 0)
    calinski_harabasz = calinski_harabasz_score(df1,ids)
    
    return [davies_bouldin,silhouette_avg,calinski_harabasz]
  
def multiprocess_clusters(df:pd.DataFrame,ascending:bool, discount_type:str,use_pca:bool,min_exponent = 1,max_exponent = 5,max_slope = 1,init_search_iters = 5, cluster_range = (5,10),cpus = 2, patid = 'enrolid',maxiter = 100, min_clust_size = 20,solver = 'hyper', n_iter_no_change = 15, early_stop_tol = .005, events_repeat_often = True, score_col = None, fin_score_column = 'FinalScore') -> Parameters:           
    '''Look for the best cluster number among a range of clusters'''
    p = LimitedParameters(discount_type = discount_type)
    p_step = Parameters()
    inputs = [df,p,ascending,discount_type,use_pca,min_exponent,max_exponent,max_slope,
              init_search_iters,patid,events_repeat_often,score_col]
    batch_step = get_batch_step([df],cluster_range,init_search_iters)
    input_generator = gen(batch_step,cluster_range[0],cluster_range[1],inputs)
    runs = []
    for batch in range(0,(cluster_range[1]-cluster_range[0] + 1),batch_step):
        try:
            results = p_umap(initial_search,*next(input_generator),num_cpus = cpus,disable = False)
            runs.append(results)
        except StopIteration:
            pass
            
    best = _best_run(runs)
    best.start = p_step.start
    finp = Parameters(best)
        
    return df,finp

def _best_run(results:list) -> Parameters:
    results = [item for sublist in results for item in sublist]
    if any([i[1].empty if isinstance(i[1],pd.DataFrame) else False for i in results]):
        results = [i for i in results if not isinstance(i[1],pd.DataFrame)]
    return max(results, key = lambda i : i[0].max_var)[0]
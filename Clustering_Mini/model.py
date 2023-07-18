import pandas as pd
from preprocess import *

class Dockkes():
  
    '''
       DOCKKES is a machine learning clustering algorithm created by Matthew Littman
       it stands for Divisive Optimized Clustering using Kernel KMeans on Event Sequences.
       it is used to cluster Event Sequences in the format of an Event log with a 
       Person, Date, and Event column. It has many different hyperparameters which can be adjusted
       to allow the algorithm to search shallower, deeper and even allows you to set a minimum cluster size
       
       DOCKKES finds the number of clusters for you and returns a dataframe with each person 
       and their cluster label along with calculations performed along the way
       
       
       
    '''
  
    def __init__(self,df:pd.DataFrame, quality_metric_output_folder:str,min_clust_size:int,
                 person_column = 'enrolid', event_column = 'event_real',date_column = 'svcdate_real',
                 na_filter_col = 'lot', remove_threshold = .0005,
                 cluster_range = (4,20),
                 max_exponent = 2,min_exponent = 1,
                 min_slope = .0001,max_slope = 2,
                 ascending = False,
                 discount_type = 'exponential',
                 maxiter = 35,
                 use_pca = False,
                 p = None, 
                 plot = False, interactive = False, plot_type = '2d', 
                 init_search_iters = 4,  
                 hierarchy_steps = 20,  
                 threshold = 'min', transition_limit = 0,transition_setting_clusters = (14,30),
                 enter_recursion = True, 
                 reduction_rate =0,
                 solver = 'hyper', n_iter_no_change = 15, early_stop_tol = .005, 
                 events_repeat_often = False,
                 score_col = None,
                 fin_score_column = 'FinalScore',
                 clust_method = 'faiss'):
                
    
        assert remove_threshold >= 0 and remove_threshold < 1
        assert isinstance(cluster_range,tuple) and isinstance(transition_setting_clusters,tuple)
        assert max_exponent > 0 and min_exponent > 0
        assert min_exponent < max_exponent 
        assert min_slope >0 and max_slope > 0
        assert min_slope < max_slope
        assert isinstance(ascending,bool)
        assert discount_type in ["exponential","constant", "logarithmic", "linear", "sqrt"]
        assert maxiter > 0 
        if use_pca:
           raise AssertionError('Use_pca = True is not yet implemented, please provide False instead')
        
        
        ################
        # Data and Prep#
        ################
        self.df = df
        #self.person_column = person_column
        self.patid = person_column
        self.event_column = event_column
        self.date_column = date_column
        self.quality_metric_output_folder = quality_metric_output_folder
        
        self._preprocess_data(na_filter_col,remove_threshold)
        
        #######################
        # Algorithm Parameters#
        #######################
        self.times_reweight = 0
        self.hierarchy_start = 1
        self.min_clust_size = min_clust_size
        self.cluster_range = cluster_range
        self.max_exponent = max_exponent
        self.min_exponent = min_exponent
        self.min_slope = min_slope
        self.max_slope = max_slope
        self.ascending = ascending
        self.discount_type = discount_type
        self.maxiter = maxiter
        self.use_pca = use_pca
        self.p = p
        self.plot = plot
        self.interactive = interactive
        self.plot_type = plot_type
        self.init_search_iters = init_search_iters
        self.hierarchy_steps = hierarchy_steps
        self.threshold = threshold
        self.transition_limit = transition_limit
        self.transition_setting_clusters = transition_setting_clusters
        self.enter_recursion = enter_recursion
        self.reduction_rate = reduction_rate
        self.solver = solver
        self.n_iter_no_change = n_iter_no_change
        self.early_stop_tol = early_stop_tol
        self.events_repeat_often = events_repeat_often
        self.score_col = score_col
        self.fin_score_column = fin_score_column
        self.clust_method = clust_method
        
        
        
    def return_method_cols(self) -> dict:
        return dict((k, self.__dict__[k]) for k in self.__dict__.keys() if k not in ['date_column','quality_metric_output_folder'])
      
    def _preprocess_data(self, na_filter_col = 'lot', remove_threshold = .0005):
        '''
           Use .notna filter to eliminate nas from na_filter_col,
           calculate a unique_id for each row,
           apply the remove_threshold,
           reduce the memory of the given dataframe,
           and preprocess the dataframe by calculating needed columns      
        '''
        if na_filter_col and na_filter_col in self.df.columns:
            self.df = self.df[self.df[na_filter_col].notna()]
        
        self.df['row_id'] = self.df[[self.patid,self.date_column]].sum(axis=1).map(hash)
        
        if remove_threshold:
            remove_below_thresh(self.df,remove_threshold,self.event_column)
            
        # Filter to needed columns and rename accordingly
        d = self.df[[self.patid,self.event_column,self.date_column,'row_id']].rename({'row_id':'unique_row',self.patid:'enrolid',self.date_column:'svcdate'},axis = 1)
        
        # Reduce Memory usage in df
        self.df = reduce_memory_usage(d)

        # Preprocess df by calculating needed columns
        self.df = preprocess(self.df, event_column = self.event_column, patid= self.patid)
       
'''Param classes'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class LimitedParameters:
    def __init__(self, **kwargs):
        self.max_var = 0
        self.max_exponent = 0
        self.max_cluster = 0
        self.max_freq_multiplier = 0
        self.max_slope = 0
        self.discount_type = ''
        self.ascending = False
        self.updated = False
        self.davies_bouldin = 0
        self.silhouette_avg = 0
        self.calinski_harabasz = 0
        self.matthew_index = 0
        self.final_iteration = False
        self.total = 0
        self.db = []
        self.sil = []
        self.ch = []
        self.tot = []
        self.matthew = []
        self.last_clust_count_min = 0
        self.start = datetime.now()
        self.end = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self,vari:float,exp:float,clust:int,freq_multiplier:float,slope:float, ascending:bool,discount_type:str,davies_bouldin:float,silhouette_avg:float,calinski_harabasz:float,total:float, matthew_index:float, last_clust_count_min:int):
        if not self.final_iteration:
            self.db.append(davies_bouldin)
            self.sil.append(silhouette_avg)
            self.ch.append(calinski_harabasz)
            self.tot.append(total)
            self.matthew.append(matthew_index)
            self.last_clust_count_min = last_clust_count_min
        if vari > self.max_var:
            #print('\nUPDATE! New Maximum:\n---------------------')
            self.max_var = vari
            self.max_exponent = exp
            self.max_cluster = clust
            self.max_freq_multiplier = freq_multiplier
            self.max_slope = slope
            self.discount_type = discount_type
            self.ascending = ascending
            self.davies_bouldin = davies_bouldin
            self.silhouette_avg = silhouette_avg
            self.calinski_harabasz = calinski_harabasz
            self.matthew_index = matthew_index
            self.total = total
            self.updated = True
        else:
            self.updated = False
            
    def __repr__(self):
        return f'Max_MI: {self.max_var}\n    Exponent: {self.max_exponent}\n    Clusters = {self.max_cluster}\n    Freq Multiplier = {self.max_freq_multiplier}\n    Slope = {self.max_slope}\n    Discount Type = {self.discount_type}'

    def clust_metrics(self,df:pd.DataFrame,c:int,exponent:float,freq_multiplier:float,slope:float,ascending:bool,
                      discount_type:str,davies_bouldin:float,silhouette_avg:float,calinski_harabasz:float, total :float, 
                      matthew_index: float, patid = 'enrolid'):

#        if self.updated or self.final_iteration:
#            print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~\nDiscount Rate: {exponent}\nCluster Number:{c}\nFreq Multiplier:{freq_multiplier}\nDiscount Direction:{ascending}\nSlope:{slope}\nDiscount Type:{discount_type}\n_________________________________")
#            print(df.groupby(['Clusters'])[patid].unique().str.len().sort_values(ascending= False))
#        else:
#            print(f'\nInferior Clustering~~~~~~~~~~~~~~~~~~~~~~~~~~\nDiscount Rate: {exponent}, Cluster Number:{c}, Freq Multiplier:{freq_multiplier}, Slope:{slope}\n')
#
#        print(f"\nClustering Metrics\n\tMatthew's Index: {matthew_index}\t-no range,\tHIGHER = better")
#        print(f'\tSilhouette Avg: {silhouette_avg}\t-range(-1...1),\tHIGHER = better')
#        print(f'\tCalinski-Harabasz: {calinski_harabasz}\t-no range,\tHIGHER = better')
#        print(f'\tDavies Bouldin: {davies_bouldin}\t-range(>= 0),\tLOWER = better')
#        print(f'\tSilhouette/Davies Bouldin: {silhouette_avg/(davies_bouldin + 1)}\t-no range, HIGHER = better\n')
#        print(self)
        return

    def validation_metrics(self,matthew_index,silhouette_avg,calinski_harabasz,davies_bouldin,total):
#        print(f"\nClustering Metrics\n\tMatthew's Index: {matthew_index}\t-no range,\tHIGHER = better")
#        print(f'\tSilhouette Avg: {silhouette_avg}\t-range(-1...1),\tHIGHER = better')
#        print(f'\tCalinski-Harabasz: {calinski_harabasz}\t-no range,\tHIGHER = better')
#        print(f'\tDavies Bouldin: {davies_bouldin}\t-range(>= 0),\tLOWER = better')
#        print(f'\tSilhouette/Davies Bouldin: {silhouette_avg/(davies_bouldin + 1)}\t-no range, HIGHER = better\n')
        return
    
    def final(self):
        #print('\n\nFINAL CLUSTERING\n========================================\n')
        self.final_iteration = True
        
    def end_timer(self):
        self.end = datetime.now()
        print(f'ELAPSED TIME: {self.elapsed()}')
        
    def elapsed(self) -> str:
        return str(self.end - self.start)
        
    def norm(self,score_type:str):
        scores = getattr(self,score_type)
        return [(score - min(scores)) / (max(scores) - min(scores)) if isinstance(score,float) else 0 for score in scores]

class Parameters(LimitedParameters):
    def __init__(self, p = None):
        if p:
            self.max_var = p.max_var
            self.max_exponent = p.max_exponent
            self.max_cluster = p.max_cluster
            self.max_freq_multiplier = p.max_freq_multiplier
            self.max_slope = p.max_slope
            self.discount_type = p.discount_type
            self.ascending = p.ascending
            self.updated = False
            self.input_df = None
            self.davies_bouldin = p.davies_bouldin
            self.silhouette_avg = p.silhouette_avg
            self.calinski_harabasz = p.calinski_harabasz
            self.matthew_index = p.matthew_index
            self.final_iteration = False
            self.total = p.total
            self.db = p.db 
            self.sil = p.sil
            self.ch = p.ch
            self.tot = p.tot
            self.matthew = p.matthew
            self.last_clust_count_min = p.last_clust_count_min
            self.start = p.start
            self.end = None
        else:
        
            self.max_var = 0
            self.max_exponent = 0
            self.max_cluster = 0
            self.max_freq_multiplier = 0
            self.max_slope = 0
            self.discount_type = ''
            self.ascending = False
            self.updated = False
            self.input_df = None
            self.davies_bouldin = 0
            self.silhouette_avg = 0
            self.calinski_harabasz = 0
            self.matthew_index = 0
            self.final_iteration = False
            self.total = 0
            self.db = []
            self.sil = []
            self.ch = []
            self.tot = []
            self.matthew = []
            self.last_clust_count_min = 0
            self.start = datetime.now()
            self.end = None
    
    
    def update(self,vari:float,exp:float,clust:int,freq_multiplier:float,slope:float, ascending:bool,discount_type:str,input_df:pd.DataFrame,davies_bouldin:float,silhouette_avg:float,calinski_harabasz:float,total:float, matthew_index:float, last_clust_count_min:int):
        if not self.final_iteration:
            self.db.append(davies_bouldin)
            self.sil.append(silhouette_avg)
            self.ch.append(calinski_harabasz)
            self.tot.append(total)
            self.matthew.append(matthew_index)
            self.last_clust_count_min = last_clust_count_min
            #Switched to < because transition score is better when it is lower
        if vari < self.max_var:
            self.max_var = vari
            self.max_exponent = exp
            self.max_cluster = clust
            self.max_freq_multiplier = freq_multiplier
            self.max_slope = slope
            self.discount_type = discount_type
            self.input_df = input_df
            self.ascending = ascending
            self.davies_bouldin = davies_bouldin
            self.silhouette_avg = silhouette_avg
            self.calinski_harabasz = calinski_harabasz
            self.matthew_index = matthew_index
            self.total = total
            self.updated = True
        else:
            self.updated = False
            

    def clust_metrics(self,df:pd.DataFrame,c:int,exponent:float,freq_multiplier:float,slope:float,
                      ascending:bool,discount_type:str,davies_bouldin:float,silhouette_avg:float,calinski_harabasz:float, 
                      total :float, matthew_index: float, patid = 'enrolid'):

#        if self.updated or self.final_iteration:
#            print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~\nDiscount Rate: {exponent}\nCluster Number:{c}\nFreq Multiplier:{freq_multiplier}\nDiscount Direction:{ascending}\nSlope:{slope}\nDiscount Type:{discount_type}\n_________________________________")
#            print(df.groupby(['Clusters'])[patid].unique().str.len().sort_values(ascending= False))
#        else:
#            print(f'\nInferior Clustering~~~~~~~~~~~~~~~~~~~~~~~~~~\nDiscount Rate: {exponent}, Cluster Number:{c}, Freq Multiplier:{freq_multiplier}, Slope:{slope}\n')
#
#        print(f"\nClustering Metrics\n\tMatthew's Index: {matthew_index}\t-no range,\tLOWER = better")
#        print(f'\tSilhouette Avg: {silhouette_avg}\t-range(-1...1),\tHIGHER = better')
#        print(f'\tCalinski-Harabasz: {calinski_harabasz}\t-no range,\tHIGHER = better')
#        print(f'\tDavies Bouldin: {davies_bouldin}\t-range(>= 0),\tLOWER = better')
#        print(f'\tSilhouette/Davies Bouldin: {silhouette_avg/(davies_bouldin + 1)}\t-no range, HIGHER = better\n')
#        print(self)
        return


    def validation_metrics(self,matthew_index,silhouette_avg,calinski_harabasz,davies_bouldin,total):
#        print(f"\nClustering Metrics\n\tMatthew's Index: {matthew_index}\t-no range,\tLOWER = better")
#        print(f'\tSilhouette Avg: {silhouette_avg}\t-range(-1...1),\tHIGHER = better')
#        print(f'\tCalinski-Harabasz: {calinski_harabasz}\t-no range,\tHIGHER = better')
#        print(f'\tDavies Bouldin: {davies_bouldin}\t-range(>= 0),\tLOWER = better')
#        print(f'\tSilhouette/Davies Bouldin: {silhouette_avg/(davies_bouldin + 1)}\t-no range, HIGHER = better\n')
        return
    
    def plot_avgscore(self):
        
        if isinstance(self.input_df,pd.DataFrame):
            plotting = self.input_df.groupby(['occ']).mean().reset_index()
            stdeviation = self.input_df.groupby(['occ']).std().reset_index()

            x = plotting.occ.values[3:]
            y = plotting.score.values[3:]

            plt.figure(figsize = (15,13))
            axes = plt.gca()
            plt.plot(x, y, label = 'Avgscore', linestyle="-")
            plt.fill_between(x, (y-stdeviation.score[3:]), (y+stdeviation.score[3:]), color='b', alpha=.1)
            plt.plot()

            plt.title('Average Scores Used for Clustering by Event Occurrence',fontsize = 25)
            plt.xlabel('Occurrence',fontsize = 20)
            plt.ylabel('Average Score', fontsize = 20)
            plt.xticks(np.arange(min(x), max(x)+1, 3), fontsize = 12)
            plt.yticks(np.arange(min(y), max(y), max(y)/10), fontsize = 12)
            axes.set_ylim([0,max(y+stdeviation.score[3:])])

            plt.legend(fontsize = 15)
            plt.show()
        else:
            print('No Input DataFrame, cannot plot avgscore')
        
    def plot_validation_measures(self):
        ch = self.norm('ch')
        db = self.norm('db')
        sil = self.norm('sil')
        total = self.norm('tot')
        matthew = self.norm('matthew')
        x = np.arange(0,len(sil),1)
        fig, axs = plt.subplots(3, 2, figsize = (15,9))
        fig.suptitle('Cluster Validation Scores', fontsize = 20)
        axs[0, 0].plot(x, ch)
        axs[0, 0].set_title('Calinski-Harabasz',fontsize = 15)
        axs[0, 1].plot(x, db, 'tab:orange')
        axs[0, 1].set_title('Davies-Bouldin',fontsize = 15)
        axs[1, 0].plot(x, sil, 'tab:green')
        axs[1, 0].set_title('Silhouette Average',fontsize = 15)
        axs[1, 1].plot(x, total, 'tab:red')
        axs[1, 1].set_title('Total: Sil / DB',fontsize = 15)
        axs[2, 0].plot(x, matthew, 'tab:purple')
        axs[2, 0].set_title('Matthew Index',fontsize = 15)

        count = 0
        for ax in axs.flat:
            count += 1
            plt.subplots_adjust(hspace = 0.5)
            ax.set(xlabel='Iteration', ylabel='Score')
            if count == 4 and len(total)>0:
                ymax = max(total)
                xpos = np.where(total == ymax)
                xmax = x[xpos[0][0]]

                ax.annotate(f'Max: i={xpos[0][0]}', xy=(xmax, ymax), xytext=(xmax, ymax-.5),
                            arrowprops=dict(facecolor='black', shrink=0.05))
        plt.show()

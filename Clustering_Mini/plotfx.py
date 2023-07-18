'''plotfx'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# ## Plots
def _plot_pca_2d(pca,kmeans, dataframe:pd.DataFrame, interactive = False, p = None,labels = None):
    '''Plot the TSNE (t-distributed stochastic neighbor embedding) of the given PCA (Principal Component Analysis) performed on the Kmeans clustering
    This will give a 2-D representation of the clusters and save the resulting picture to "clustering.png"
    '''
    if labels is not None:
        y_pred = labels
    else:
        y_pred = kmeans.labels_
   
    if isinstance(dataframe,pd.DataFrame):
        dataframe = dataframe.to_numpy()
    tsne = TSNE(n_components = 2).fit_transform(dataframe)

    if interactive:
        import plotly.express as px

        df = dataframe.copy(deep = True).reset_index()
        tsne_df = pd.DataFrame(tsne,columns = ['x','y'])
        conc = pd.concat([df,tsne_df],axis = 1)

        title = "K-means Clustering of Treatment Patterns - Bipolar" 
        colors = [str(i) for i in y_pred]
        hover_data = {'x':False,'y':False}
        hover_data[col_to_predict] = True
        conc.loc[:,original_col] = conc.loc[:,original_col].apply(lambda x: x if len(x) < 100 else x[:100] + '...')

        fig = px.scatter(conc, x='x', y ='y',color = colors, labels = {'color':'Cluster'}, title = title, hover_name = 'Index', hover_data = {'x':False,'y':False} )

        fig.update_traces(marker_size = 3)
        fig.update_layout(hovermode = 'x unified')
        fig.show()
    else:
        plt.figure(figsize = (17,13))
        scatter = plt.scatter(tsne[:,0], tsne[:,1],c=y_pred, cmap='gist_rainbow')
        plt.legend(*scatter.legend_elements())
        plt.suptitle("K-means Clustering of Treatment Patterns - Bipolar", y=.95, fontsize=32)
        plt.title(f'Discount_Type:{p.discount_type}, Clusters:{p.max_cluster}', fontsize=20)
        plt.savefig(f'Clustering{p.max_cluster}{p.discount_type}{str(p.max_exponent.round(2)) if isinstance(p.max_exponent,float) else str(p.max_exponent)}.png')
        
def _plot_pca_3d(pca,kmeans,dataframe:pd.DataFrame, interactive = False, p = None):
    '''Plot the TSNE (t-distributed stochastic neighbor embedding) of the given PCA (Principal Component Analysis) performed on the Kmeans clustering
    This will give a 3-D representation of the clusters and save the resulting picture to "clustering.png"
    If Interactive flag set to true then interactivity performed with plotly
    '''
    y_pred = kmeans.labels_
    if isinstance(dataframe,pd.DataFrame):
        dataframe = dataframe.to_numpy()
    tsne = TSNE(n_components = 3).fit_transform(dataframe)
    
    if interactive:
        import plotly.express as px
        
        #df = dataframe.copy(deep = True).reset_index()
        tsne_df = pd.DataFrame(tsne,columns = ['x','y','z'])
        #conc = pd.concat([df,tsne_df],axis = 1)
        title = "K-means Clustering of Treatment Patterns - Bipolar"
        colors = [str(i) for i in y_pred]
        #hover_data = {'x':False,'y':False,'z':False}
        fig = px.scatter_3d(tsne_df, x='x', y ='y',z='z',color = colors, labels = {'color':'Cluster'}, title = title)
        fig.update_traces(marker_size = 3)
        fig.update_layout(hovermode = 'x unified')
        fig.show()
    
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize = (17,13))
        ax = fig.add_subplot(111,projection = '3d')
        scatter = ax.scatter(*zip(*tsne[:,:2]),c=y_pred, cmap='gist_rainbow')
        plt.legend(*scatter.legend_elements())
        plt.suptitle("K-means Clustering of Treatment Patterns - Bipolar", y=.95, fontsize=32)
        plt.title(f'Discount_Type:{p.discount_type}, Clusters:{p.max_cluster}', fontsize=20)
        plt.savefig(f'Clustering{p.max_cluster}{p.discount_type}{str(p.max_exponent.round(2)) if isinstance(p.max_exponent,float) else str(p.max_exponent)}.png')

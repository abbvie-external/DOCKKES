import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import silhouette_score,davies_bouldin_score
if os.getcwd() != '/home/cdsw/Clustering/Clustering_Code/Clustering_Mini':
    os.chdir('Clustering/Clustering_Code/Clustering_Mini')
from package_res import gen_model_param_analysis_file

gen_model_param_analysis_file (location ="Dawid_skyrizi", name ="dl_vraylar_param_output.csv")
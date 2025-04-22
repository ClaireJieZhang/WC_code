import configparser
import sys
import timeit
from pathlib import Path
import numpy as np 
import pandas as pd 
from fair_clustering_util import fair_clustering_util
from util.configutil import read_list
from util.utilhelpers import get_clust_sizes, max_Viol_multi_color, x_for_colorBlind, max_RatioViol_multi_color, find_proprtions_two_color_deter 
import time
import os


num_colors = 2

# Set LowerBound = 0
LowerBound = 0

#
# alpha0: is the first POF  
alpha0= 1.001



# alphaend: is the last POF   
alphaend= 1.001

# 
alpha_step = 0.01/2

# set ml_model_flag=False, p_acc=1.0
ml_model_flag = False
p_acc = 1.0 

''' POF ''' 
# flag two color util 
two_color_util=True
r = 2**7
epsilon = 1/r
alpha_POF = 1.01 # ratio between U_POF and the color-blind cost  


config_file = "config/example_2_color_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)


config_str = "creditcard_binary_marriage" if len(sys.argv) == 1 else sys.argv[1]


# Read variables
data_dir = config[config_str].get("data_dir")
dataset = config[config_str].get("dataset")
clustering_config_file = config[config_str].get("config_file")
num_cluster = list(map(int, config[config_str].getlist("num_clusters")))
deltas = list(map(float, config[config_str].getlist("deltas")))
max_points = config[config_str].getint("max_points")


# ready up for the loop 
alpha = np.linspace(
    start=alpha0,
    stop=alphaend,
    num=int((alphaend - alpha0) / alpha_step) + 1,
    endpoint=True
)




df = pd.DataFrame(columns=['num_clusters','POF','UtilValue','UtilLP','LP Iters','opt_index','epsilon','Epsilon_set_size','minclustsize','Run_Time']) # ,'MaxViolFair','MaxViolUnFair','Fair Balance','UnFair Balance','Run_Time','ColorBlindCost','FairCost'])
iter_idx = 0 


#
initial_score_save = 0 
pred_save = [] 
cluster_centers_save = [] 
 
if type(num_cluster) is list:
    num_cluster = num_cluster[0] 

counter = 0 

for a in alpha:

	start_time = timeit.default_timer()
	output, initial_score_save, pred_save, cluster_centers_save = fair_clustering_util(counter, initial_score_save, pred_save, cluster_centers_save, dataset, clustering_config_file, data_dir, num_cluster, deltas, max_points, LowerBound, p_acc, ml_model_flag, two_color_util,epsilon,a)
	elapsed_time = timeit.default_timer() - start_time
	scaling = output['scaling']
	clustering_method = output['clustering_method']
	x_rounded = output['assignment']
	color_flag = output['color_flag']
	num_points = sum(x_rounded)
	clust_sizes = np.min(get_clust_sizes(x_rounded,num_cluster)) 


	fair_cost = output['objective']
	colorBlind_cost = output['unfair_score']

	minclustsize = np.min(clust_sizes)

	POF = a
	utilValue = output['util_objective']
	LP_iters = output["bs_iterations"]
	opt_index = output['opt_index']
	epsilon = output["epsilon"]
	epsilon_set_size = output["epsilon set size "]
	util_lp = output['util_lp']
	df.loc[iter_idx] = [num_cluster,POF,utilValue,util_lp,LP_iters,opt_index,epsilon,epsilon_set_size,minclustsize,elapsed_time]
	iter_idx += 1
	counter += 1



scale_flag = 'normalized' if scaling else 'unnormalized' 
filename = dataset + '_' + clustering_method + '_' + str(int(num_cluster))+ '_' + str(deltas) +'_' + str(int(num_points)) + '_' + scale_flag  
filename = filename + '_' 
filename = filename + '.csv'


os.makedirs('Results', exist_ok=True)

# do not over-write 
filepath = Path('Results' + '/'+ filename)
while filepath.is_file():
    filename='new' + filename 
    filepath = Path('Results' + '/'+ filename)

df.to_csv('Results' + '/'+ filename, sep=',',index=False)

















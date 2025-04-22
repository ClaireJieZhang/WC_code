import configparser
import sys
import timeit
from pathlib import Path
import numpy as np 
import pandas as pd 
from fair_clustering_lex import fair_clustering_lex
from util.configutil import read_list
from util.utilhelpers import get_clust_sizes, max_Viol_multi_color, x_for_colorBlind, max_RatioViol_multi_color, find_proprtions_two_color_deter 

# num_colors: number of colors according to the data set 
#num_colors = 2

# Set LowerBound = 0
LowerBound = 0

#
# alpha0: is the first POF  
alpha0= 1.01

# alphaend: is the last POF 
alphaend= 1.51
#alphaend= 1.03

# delta_lex_arr
alpha_step = 0.10

# POF lower bound on size for rounding 
L_POF = 100 


# set ml_model_flag=False, p_acc=1.0
ml_model_flag = False
p_acc = 1.0

''' POF ''' 
# flag two color util 
two_color_util=False
r = 2**7
epsilon = 1/r
alpha_POF = 1.01 # ratio between U_POF and the color-blind cost  


config_file = "config/example_multi_color_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)


config_str = "census1990" if len(sys.argv) == 1 else sys.argv[1]


# Read variables
data_dir = config[config_str].get("data_dir")
dataset = config[config_str].get("dataset")
clustering_config_file = config[config_str].get("config_file")
num_cluster = list(map(int, config[config_str].getlist("num_clusters")))
deltas = list(map(float, config[config_str].getlist("deltas")))
max_points = config[config_str].getint("max_points")


# ready up for the loop 
alpha = np.linspace(start=alpha0 , stop=alphaend, num=((alphaend-alpha0)/alpha_step)+1, endpoint=True)

#df = pd.DataFrame(columns=['num_clusters','POF','UtilValue','UtilLP','LP Iters','opt_index','epsilon','Epsilon_set_size','clust_sizes']) # ,'MaxViolFair','MaxViolUnFair','Fair Balance','UnFair Balance','Run_Time','ColorBlindCost','FairCost'])
#df = pd.DataFrame(columns=['num_clusters','POF','color_objectives_rounded','color_objectives_lp','epsilon','Epsilon_set_size','clust_sizes']) # ,'MaxViolFair','MaxViolUnFair','Fair Balance','UnFair Balance','Run_Time','ColorBlindCost','FairCost'])



df = pd.DataFrame(columns=['num_clusters','POF','color_objectives_rounded_1','color_objectives_rounded_2','color_objectives_rounded_3','color_objectives_rounded','color_objectives_lp','epsilon','Epsilon_set_size','minclustsize','Run_Time']) # ,'MaxViolFair','MaxViolUnFair','Fair Balance','UnFair Balance','Run_Time','ColorBlindCost','FairCost'])

iter_idx = 0 


#
initial_score_save = 0 
pred_save = [] 
cluster_centers_save = [] 
 
if type(num_cluster) is list:
    num_cluster = num_cluster[0] 

counter = 0 

for a in alpha:

	print('Now at POF=%f' % a)

	start_time = timeit.default_timer()
	output, initial_score_save, pred_save, cluster_centers_save = fair_clustering_lex(counter, initial_score_save, pred_save, cluster_centers_save, dataset, clustering_config_file, data_dir, num_cluster, deltas, max_points, LowerBound, p_acc, ml_model_flag, two_color_util,epsilon,a,L_POF)
	elapsed_time = timeit.default_timer() - start_time
	scaling = output['scaling']
	clustering_method = output['clustering_method']
	x_rounded = output['assignment']
	num_points = sum(x_rounded)
	clust_sizes= get_clust_sizes(x_rounded,num_cluster) 
	fair_cost = output['objective']
	colorBlind_cost = output['unfair_score']

	print('0---1')
	print(clust_sizes)
	minclustsize = np.min(clust_sizes)

	POF = a
	#color_flag = output['color_flag']
	#utilValue = output['util_objective']
	#LP_iters = output["bs_iterations"]
	#opt_index = output['opt_index']
	color_objectives_rounded = output['color_objectives_rounded']
	color_objectives_lp = output['color_objectives_lp']
	epsilon = output["epsilon"]
	epsilon_set_size = output["epsilon set size "]
	#util_lp = output['util_lp']
	df.loc[iter_idx] = [num_cluster,POF,color_objectives_rounded[0],color_objectives_rounded[1],color_objectives_rounded[2],color_objectives_rounded, color_objectives_lp ,epsilon,epsilon_set_size,minclustsize,elapsed_time]
	iter_idx += 1
	counter += 1



scale_flag = 'normalized' if scaling else 'unnormalized' 
filename = dataset + '_' + clustering_method + 'Leximin' + '_' + str(int(num_cluster))+ '_' + str(deltas) +'_' + str(int(num_points)) + '_' + scale_flag  
filename = filename + '_' 
filename = filename + '.csv'



# do not over-write 
filepath = Path('Results' + '/'+ filename)
while filepath.is_file():
    filename='new' + filename 
    filepath = Path('Results' + '/'+ filename)

df.to_csv('Results' + '/'+ filename, sep=',',index=False)

















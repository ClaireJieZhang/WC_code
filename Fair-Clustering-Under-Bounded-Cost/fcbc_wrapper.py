import configparser
import ast
from Fair_Clustering_Under_Bounded_Cost.fair_clustering_util import fair_clustering_util
from evaluation_utils.utils import read_list

def run_fcbc_pipeline_with_loaded_data(
    df, svar_all,
    dataset_name,
    config_file,
    data_dir,
    num_clusters,
    deltas,
    counter=0,
    initial_score_save="results/fcbc_score.pkl",
    pred_save="results/fcbc_preds.pkl",
    cluster_centers_save="results/fcbc_centers.pkl",
    L=0,
    p_acc=1.0,
    ml_model_flag=False,
    two_color_util=True,
    epsilon=0.0,
    alpha_POF=0
):
    """
    FCBC wrapper that bypasses internal loading and calls fair_clustering_util using preloaded data.
    """
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    return fair_clustering_util(
        counter=counter,
        initial_score_save=initial_score_save,
        pred_save=pred_save,
        cluster_centers_save=cluster_centers_save,
        dataset=dataset_name,
        config_file=config_file,
        data_dir=data_dir,
        num_clusters=num_clusters,
        deltas=deltas,
        L=L,
        p_acc=p_acc,
        ml_model_flag=ml_model_flag,
        two_color_util=two_color_util,
        epsilon=epsilon,
        alpha_POF=alpha_POF,
        max_points=None,  # no subsampling since data is already loaded
        df=df,
        svar_all=svar_all
    )

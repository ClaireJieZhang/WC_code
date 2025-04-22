# README

## Requirements

`Python3.6` is expected for this program because currently the CPLEX solver being used expects that version of Python.

To install the non-CPLEX dependencies, use `pip install -r requirements.txt`.

To install CPLEX, visit the IBM website and navigate to the proper license (student, academic, professional, etc.), and follow the installation guide provided by IBM.

## Running Example Script

The following examples, will vary a the price of fairness PoF and produce a csv file under the Results folder:
* `example_vary_POF_util.py`: This is for the utiltarean objective. 
* `example_vary_POF_lex.py`: This is for the leximin objective.


## Running your Own Tests

To run one of your own tests, edit the following three things:

1. Create an entry in `example_config.ini`. An entry begins with `[your_title]` and contains all the fields specified by the example.

2. Change the objective in `dataset_configs.ini` if you desire.

3. Run the example script using your specifications instead of example by running `python example.py your_title`.



## Description of Output Format

The output from a trial will be a new file for each run with the timestamp: `%Y-%m-%d-%H:%M:%S`. A run is defined as a combination of `num_cluster` and `delta` in the config file. For example, if two values for `num_clusters` and two deltas are specified, then 4 runs will occur.

Each output file is in JSON format, and can be loaded using the `json` package from the standard library. The data is held as a dictionary format and can be accessed by using string key names of the following fields: 
* `num_clusters` : The number of clusters used for this trial.
* `alpha` : Dictionary holding the alphas for various colors. First key is the attribute (ie. sex), and second key is the color within that attribute (ie. male).
* `beta` : Dictionary holding the betas for various colors.
First key is the attribute (ie. sex), and second key is the color within that attribute (ie. male).
* `unfair_score` : Clustering objective score returned by vanilla clustering. 0 if `violating` is True.
* `objective` : Clustering objective returned by the fair clustering algorithm.
* `sizes` : List holding the sizes of the clusters returned by vanilla clustering. Empty list if `violating` is True.
* `attributes` : Dictionary holding the points that belong to each color group. First key is the attribute that is being considered (ie. sex), second key is the color group within that attribute that the point belongs to (ie. male).
* `centers` : List of centers found by vanilla clustering. Empty list if `violating` is True.
* `points` : List of points used for fair clustering or violating LP. Useful if the dataset has been subsampled to know which points were chosen by the subsampling method.
* `assignment`: List (sparse) of points and their assigned cluster. There are (# of points) * (# of centers) entries in assignments. For each point `i`, we say that it is assigned to that cluster `f` if `assignment[i*(# of centers) + f] == 1`.
* `name` : String name of the dataset chosen. Will use name from `dataset_configs.ini` file.
* `clustering_method` : String name of the clustering method used.
* `scaling` : Boolean of whether or not data was scaled.
* `delta` : delta value used for this run. Note that this is not the overlap but rather the variable involved in the reparameterization of alpha and beta. beta = (1 - delta) and alpha = 1 / (1 - delta).
* `time` : Float that is the time taken for the LP to be solved.
* `cluster_time` : Float that is the time taken for the vanilla clustering to occur. 0 if `violating` is true.


## Refernces

The following code was used in:
https://github.com/nicolasjulioflores/fair_algorithms_for_clustering 

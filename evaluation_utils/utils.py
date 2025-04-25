
# Read data in from a given csv_file found in config
# Arguments:
#   config (ConfigParser) : config specification (dict-like)
#   dataset (str) : name of dataset in config file to read from
# Output:
#   df (pd.DataFrame) : contains data from csv_file in `config`
def read_data(config, dataset):
    csv_file = config[dataset]["csv_file"]
    df = pd.read_csv(csv_file,sep=config[dataset]["separator"])

    if config["DEFAULT"].getboolean("describe"):
        print(df.describe())

    return df

# Clean the data. Bucketize text data, convert int to float.
# Arguments:
#   df (pd.DataFrame) : DataFrame containing the data
#   config (ConfigParser) : Config object containing the specifications of
#       which columns are text.
#   dataset (str) : Name of dataset being used.
def clean_data(df, config, dataset):
    # CLEAN data -- only keep columns as specified by the config file
    selected_columns = config[dataset].getlist("columns")
    variables_of_interest = config[dataset].getlist("variable_of_interest")
  
    # Bucketize text data
    text_columns = config[dataset].getlist("text_columns", [])

    print("[DEBUG] text_columns from config:", text_columns)
    for col in text_columns:
        print(f"[DEBUG] Before encoding: unique values in {col}:", df[col].unique())
        df[col] = df[col].astype('category').cat.codes
        print(f"[DEBUG] After encoding {col}:", df[col].unique(), "| dtype:", df[col].dtype)
    for col in text_columns:
        # Cat codes is the 'category code'. Aka it creates integer buckets automatically.

        df[col] = df[col].astype('category').cat.codes

    # Remove the unnecessary columns. Save the variable of interest column, in case
    # it is not used for clustering.
    variable_columns = [df[var] for var in variables_of_interest]
    # df = df[[col for col in selected_columns]]

    # Convert to float, otherwise JSON cannot serialize int64
    for col in df:
        if col in text_columns or col not in selected_columns: continue
        df[col] = df[col].astype(float)

    if config["DEFAULT"].getboolean("describe_selected"):
        pass 
        #print(df.describe())

    return df, variable_columns

# Return a df with N subsamples from df
# Arguments:
#   df (pd.DataFrame) : Dataframe to subsample
#   N (int) : number of samples to take
# Output:
#   df (pd.DataFrame) : Subsampled Dataframe
def subsample_data(df, N):
    return df.sample(n=N).reset_index(drop=True)

# Scales each of the columns to have mean = 0, var = 1
# Arguments:
#   df (pd.DataFrame) : Dataframe to scale
# Output:
#   df (pd.DataFrame) : Dataframe scaled
def scale_data(df):
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df[df.columns]), columns=df.columns)
    return df
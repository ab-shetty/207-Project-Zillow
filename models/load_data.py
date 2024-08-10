import pandas as pd 


def load_mlr_data(base_path):
    # Reading in 2016 property information
    df_properties = pd.read_csv(f'{base_path}properties_2016.csv')

    # Reading in 2016 and 2017 transaction data and the sample submission file
    df_2016 = pd.read_csv(
        f'{base_path}train_2016_v2.csv', parse_dates=["transactiondate"])
    df_2017 = pd.read_csv(
        f'{base_path}train_2017.csv', parse_dates=["transactiondate"])
    sample = pd.read_csv(f'{base_path}sample_submission.csv')

    # Merging the 2016 and 2017 transaction data
    df_logs = pd.concat([df_2016, df_2017])

    # Merging the new transaction data with it's associated properties
    df_train = pd.merge(df_logs, df_properties, on='parcelid', how='inner')

    return df_properties, sample, df_logs, df_train


def load_xgb_data(base_path: str):
    # Read the data
    train = pd.read_csv(f'{base_path}train_2016_v2.csv')
    prop = pd.read_csv(f'{base_path}properties_2016.csv')
    sample = pd.read_csv(f'{base_path}sample_submission.csv')
    return train, prop, sample


# def load_nn_data(base_path):
#     # Read the data
#     train = pd.read_csv("./data/train_2016_v2.csv",
#                         parse_dates=["transactiondate"])
#     prop = pd.read_csv(f'{base_path}properties_2016.csv')
#     sample = pd.read_csv(f'{base_path}sample_submission.csv')
#     return train, prop, sample


def load_nn_data(base_path):
    # Read and clean dataframe
    df_properties = pd.read_csv(f'{base_path}properties_2017.csv')

    # Get logerrors of transactions and merge
    df_2016 = pd.read_csv(f'{base_path}train_2016_v2.csv',
                        parse_dates=["transactiondate"])
    df_2017 = pd.read_csv(f'{base_path}train_2017.csv',
                        parse_dates=["transactiondate"])
    sample = pd.read_csv(f'{base_path}sample_submission.csv')
    df_logs = pd.concat([df_2016, df_2017])

    # Inner join transactions with the dataset of all properties
    df_all = pd.merge(df_logs, df_properties, on='parcelid', how='inner')

    return df_all, df_properties, sample




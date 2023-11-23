import pandas as pd
import os

def main():
    # Read in preprocessed data
    data_eb_test = pd.read_parquet("SSNF2/preprocess/data_eb_test.parquet")
    data_eb_train = pd.read_parquet("SSNF2/preprocess/data_eb_train.parquet")
    mc_eb_test = pd.read_parquet("SSNF2/preprocess/mc_eb_test.parquet")
    mc_eb_train = pd.read_parquet("SSNF2/preprocess/mc_eb_train.parquet")

    # create label column
    data_eb_test["label"] = ["data"] * data_eb_test.shape[0]
    data_eb_train["label"] = ["data"] * data_eb_train.shape[0]
    mc_eb_test["label"] = ["mc"] * mc_eb_test.shape[0]
    mc_eb_train["label"] = ["mc"] * mc_eb_train.shape[0]

    # delete weights column
    del data_eb_test["weight"]
    del data_eb_train["weight"]
    del mc_eb_test["weight"]
    del mc_eb_train["weight"]

    # take the same amount of samples from data and mc for the training
    num_samples_train = min(len(data_eb_train), len(mc_eb_train))
    balanced_data_train = data_eb_train[:num_samples_train]
    balanced_mc_train = mc_eb_train[:num_samples_train]

    num_samples_test = min(len(data_eb_test), len(mc_eb_test))
    balanced_data_test = data_eb_test[:num_samples_test]
    balanced_mc_test = mc_eb_test[:num_samples_test]
    
    # concat the training and test data
    test_data = pd.concat([balanced_data_test, balanced_mc_test], axis=0)
    train_data = pd.concat([balanced_data_train, balanced_mc_train], axis=0)

    # save data in folder as parquet
    folder_name = "SSNF2/classifier/data/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created succesfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    
    test_data.to_parquet("SSNF2/classifier/data/test_data.parquet")
    train_data.to_parquet("SSNF2/classifier/data/train_data.parquet")

if __name__ == "__main__":
    main()
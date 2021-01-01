import pandas as pd

DATASET_FILE_PATH = "./data/train.csv"
TESTSET_FILE_PATH = "./data/test.csv"

def getDataSet() :
    dataset_file = pd.read_csv(DATASET_FILE_PATH)

    return dataset_file

def getTestSet():
    testset_file = pd.read_csv(TESTSET_FILE_PATH)

    return testset_file
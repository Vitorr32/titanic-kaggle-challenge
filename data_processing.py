import pandas as pd
import numpy as np
import feature_engineering as featEng
import feature_completion as featCompl

DATASET_FILE_PATH = "./data/train.csv"
TESTSET_FILE_PATH = "./data/test.csv"

training_data = pd.read_csv(DATASET_FILE_PATH)
test_data = pd.read_csv(TESTSET_FILE_PATH)

# Remove outliers in the Fare attribute
training_data = featCompl.removeRowsWhereColumnIsBeyondValue(
    training_data, 'Fare', 500)

# Added column Title
training_data, test_data = featEng.createTitleFeature(training_data, test_data)

# Added column isAlone
training_data, test_data = featEng.createIsAloneFeature(
    training_data, test_data)

# Added column Family Size
training_data, test_data = featEng.createFamilySize(training_data, test_data)

# Modify Fare column
training_data, test_data = featEng.calculateTrueFare(training_data, test_data)

# Drop PassengerId, Cabin and Ticket
training_data, test_data = featEng.dropInconclusiveColumns(
    training_data, test_data, ['PassengerId', 'Cabin', 'Ticket'])

# Complete Age column with regressor prediction
training_data, test_data = featCompl.completeAgeWithRandomForestRegressor(
    training_data, test_data)

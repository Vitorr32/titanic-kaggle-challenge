import pandas as pd
import numpy as np
import feature_engineering as featEng
import feature_completion as featCompl

DATASET_FILE_PATH = "./data/train.csv"
TESTSET_FILE_PATH = "./data/test.csv"

training_data = pd.read_csv(DATASET_FILE_PATH)
test_data = pd.read_csv(TESTSET_FILE_PATH)
metadata = {
    'dummies_metadata': None
}

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

# Complete Embarked Column with the mode of the datset
training_data, test_data = featCompl.completeEmbarkedNotAssigned(
    training_data, test_data)

# Complete Fare column with the mean of the dataset
training_data, test_data = featCompl.completeValuesWithMean(
    training_data, test_data, 'Fare')

# Drop PassengerId, Cabin and Ticket as we couldn't find a value in they existing
training_data, test_data = featEng.dropInconclusiveColumns(
    training_data, test_data, ['PassengerId', 'Cabin', 'Ticket'])

# Convert non-numeric data into categorical data using dummies of Pandas
training_data, test_data = featEng.createDummyVariables(
    training_data, test_data, ['Title', 'Embarked'])

# Conver the male/female values to 0/1 respectively
training_data, test_data = featEng.convertValuesToBoolean(
    training_data, test_data, 'Sex', 'female')

# Complete Age column with regressor prediction
training_data, test_data = featCompl.completeAgeWithRandomForestRegressor(
    training_data, test_data)

# Drop Name, Family Size, Siblings and Parents columns since they are not nescessary anymore
training_data, test_data = featEng.dropInconclusiveColumns(
    training_data, test_data, ['SibSp', 'Parch', 'FamilySize', 'Name'])

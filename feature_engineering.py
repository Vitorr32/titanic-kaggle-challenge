import pandas as pd
import numpy as np


def createTitleFeature(training_data, test_data):
    for dataset in [training_data, test_data]:
        # Extract the first word that ends with a dot, therefore Mr. Anderson would be replaced with Mr.
        dataset['Title'] = dataset.Name.str.extract(
            ' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Count', 'Capt', 'Col',
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Duke', 'Duchess'], 'Rare')

        # Convert outlier non-rare (typos and Mrs == Miss) female titles to more common Miss
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Miss')

    return (training_data, test_data)


def createFamilySize(training_data, test_data):
    for dataset in [training_data, test_data]:
        dataset['FamilySize'] = dataset.SibSp + dataset.Parch + 1

    return (training_data, test_data)


def createIsAloneFeature(training_data, test_data):
    # Make a conditional assingment using np.where, if siblings/parents and parch count equals to 0, then
    # the passenger was alone, if not he was with some familiar
    for dataset in [training_data, test_data]:
        dataset['isAlone'] = np.where(
            (dataset['SibSp'] == 0) & (dataset['Parch'] == 0), 1, 0)

    return (training_data, test_data)


def calculateTrueFare(training_data, test_data):
    for dataset in [training_data, test_data]:
        dataset['Fare'] = dataset.Fare / dataset.FamilySize

    return (training_data, test_data)


def dropInconclusiveColumns(training_data, test_data, columns):
    for dataset in [training_data, test_data]:
        for column in columns:
            dataset.drop([column], axis=1, inplace=True)

    return (training_data, test_data)

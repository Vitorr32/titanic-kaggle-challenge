import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# Fill up missing data from the Embarked column, since it's only two of them we will just fill the the mode, the
# most common value
def completeEmbarkedNotAssigned(training_data, test_data):
    for dataset in [training_data, test_data]:
        freq_port = dataset.Embarked.dropna().mode()[0]
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    return (training_data, test_data)


def completeValuesWithMean(training_data, test_data, column):
    for dataset in [training_data, test_data]:
        mean = dataset[column].dropna().mean()
        dataset[column] = dataset[column].fillna(mean)

    return (training_data, test_data)


def completeAgeWithRandomForestRegressor(training_data, test_data):
    train = pd.concat([training_data[["Survived", "Age", "Sex", "SibSp",
                                      "Parch"]], training_data.loc[:, "Fare":]], axis=1)
    test = pd.concat([test_data[["Age", "Sex"]],
                      test_data.loc[:, "SibSp":]], axis=1)

    for dataset in [[test, test_data]]:
        # Getting all features expect Survived, notice that the test data already has no Survived column
        ageDataset = dataset[0].loc[:, dataset[0].columns != 'Survived']

        # Separate the datasets into a has/don't have Age value training/prediction sets.
        temp_train = ageDataset.loc[ageDataset.Age.notnull()]
        # Create dataset that will be predicted by the Forest Regressor
        temp_test = ageDataset.loc[ageDataset.Age.isnull()]

        # Create our Y (Expected result) by picking the ages of each line of the training set
        y = temp_train.Age.values
        # Create our X (Training data) by dropping the Age column (the first column)
        x = temp_train.loc[:, "Sex":].values

        # Set our random forest regressor and the fit our X/Y datasets
        rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)
        rfr.fit(x, y)

        # Now we can predict the age of the lines without an Age by dropping the first column of the test set
        predicted_age = rfr.predict(temp_test.loc[:, "Sex":])

        # Now we modify the original training_data/test_data with the predicted ages
        dataset[1].loc[dataset[1].Age.isnull(), "Age"] = predicted_age

    return (training_data, test_data)


def removeRowsWhereColumnIsBeyondValue(training_data, column, value):
    training_data = training_data[getattr(training_data, column) < value]

    return training_data

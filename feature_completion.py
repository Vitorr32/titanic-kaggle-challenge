import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# Fill up missing data from the Embarked column, since it's only two of them we will just fill the the mode, the
# most common value
def completeEmbarkedNotAssigned(training_data, test_data):
    for dataset in [training_data, test_data]:
        freq_port = dataset.Embarked.dropna().mode()[0]
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    return (training_data, test_data)


def completeAgeWithRandomForestRegressor(training_data, test_data):
    train = pd.concat([training_data[["Survived", "Age", "Sex", "SibSp",
                                      "Parch"]], training_data.loc[:, "isAlone":]], axis=1)
    test = pd.concat([test_data[["Age", "Sex"]],
                      test_data.loc[:, "SibSp":]], axis=1)

    for dataset in [[train, training_data], [test, test_data]]:
        # gettting all the features except survived
        ageDataset = dataset[0].loc[:, "Age":]

        # dataset with age values
        temp_train = ageDataset.loc[ageDataset.Age.notnull()]
        print(temp_train)
        # dataset without age values
        temp_test = ageDataset.loc[ageDataset.Age.isnull()]
        print(temp_test)

        y = temp_train.Age.values  # setting target variables(age) in y
        x = temp_train.loc[:, "Sex":].values

        rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)
        rfr.fit(x, y)

        predicted_age = rfr.predict(temp_test.loc[:, "Sex":])

        dataset[1].loc[dataset[1].Age.isnull(), "Age"] = predicted_age

    return (training_data, test_data)


def removeRowsWhereColumnIsBeyondValue(training_data, column, value):
    training_data = training_data[getattr(training_data, column) < value]

    return training_data

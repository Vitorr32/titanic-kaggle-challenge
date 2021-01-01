import numpy as np
import pandas as pd
import tensorflow as tf

from utils import file_import, pre_processing

train_dataset = file_import.getDataSet()
test_dataset = file_import.getTestSet()

X_train = None
Y_train = train_dataset["Survived"]
X_test = None

combine = [train_dataset, test_dataset]

# Create a Titles column of the training and test dataset, extracting from the Name feature values
for dataset in combine:
    # Extract the first word that ends with a dot, therefore Mr. Anderson would be replaced with Mr.
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Count', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Duke', 'Duchess'], 'Rare')

    # Convert outlier non-rare (typos and Mrs == Miss) female titles to more common Miss
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Miss')

# Filling the voids: the Age feature, with is one of the most important have several null training examples, we
# need to fill them up, for that we will use median based in some secondary feature (such as Gender or Pclass)
# We will create age mean based on the Pclass of each gender, therefore a 3,2 matrix
ageMeans = np.zeros((3, 2))

for pClass in range(3):
    # The plus one is because pClass start from 1 not zero
    # Pick by class first so we don't need to query the entire dataset again to filter by gender
    dataset_by_class = combine[0][(combine[0]['Pclass'] == pClass + 1)]
    dataset_by_class = dataset_by_class.dropna(subset=['Age'])

    dataset_males_by_class = dataset_by_class[(
        dataset_by_class['Sex'] == 'male')]
    dataset_females_by_class = dataset_by_class[(
        dataset_by_class['Sex'] == 'female')]

    ageMeans[pClass, :] = [
        dataset_males_by_class['Age'].mean(),
        dataset_females_by_class['Age'].mean()
    ]

ageMeans = np.around(ageMeans)

# Now we will replace the empty Age values in the training dataset in accordance to the class and gender
for i, row in combine[0].iterrows():
    # Skip rows that don't have a null Age
    if not(pd.isna(row['Age'])):
        continue

    rowClass = row['Pclass'] - 1
    rowSex = 0 if row['Sex'] == 'male' else 1
    combine[0].at[i, 'Age'] = ageMeans[rowClass, rowSex]


# The features of sibling number(SibSo) and parent numbers(Parch) don't have a decent correlation with the survival
# rate, but we could simplify this features into a single isAlone feature, that does have a determinable relation
# as people who were alone has a significant increase in survival rate

for dataset in combine:
    dataset['isAlone'] = np.where(
        (dataset['SibSp'] == 0) & (dataset['Parch'] == 0), 1, 0)

# Fill up missing data from the Embarked column, since it's only two of them we will just fill the the mode, the
# most common value
for dataset in combine:
    freq_port = dataset.Embarked.dropna().mode()[0]
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# Fill up missing data from the Fare column, since this is a dollar value we should use the median
for dataset in combine:
    meanFare = dataset.Fare.dropna().mean()
    dataset['Fare'] = dataset['Fare'].fillna(meanFare)


# Feature analysis defined that some features have very weak correlation to the survived rate, and should be
# dropped for better perfomance (ID, Name, Ticket and Cabin), we mantain the id in the test for validation later
# We also can drop the SibSp and Parch since we created the more useful isAlone column

train_dataset = train_dataset.drop(
    ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
test_dataset = test_dataset.drop(
    ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
combine = [train_dataset, test_dataset]

# Now we call the pre_processing file in the utils modules that will do the more advanced data mapping such as
# Indexation of strings, one-hot string lookup, normalization of values and so on
processed_inputs = pre_processing.preprocess_data(
    train_dataset,
    ['Fare', 'Age'],
    ['Sex', 'Embarked', 'Title']
)

body = tf.keras.Sequential(
    [tf.keras.layers.Dense(64), tf.keras.layers.Dense(1)]
)

result = body(processed_inputs['preprocessed_inputs'])
model = tf.keras.Model(processed_inputs['inputs'], processed_inputs['preprocessed_inputs'])
model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.optimizers.Adam())

X_train = train_dataset.copy()
X_test = test_dataset.copy()

print(X_train.head())
print(processed_inputs['inputs'])

# model.fit(x = X_train, y = Y_train, epochs=10)

# Y_pred = model.predict(X_test)

# print(Y_pred)
# titanic_preprocessing = tf.keras.Model(train_dataset, model_inputs)

# titanic_features_dict = {name: np.array(value)
#                          for name, value in train_dataset.items()}
# features_dict = {name: values[:1]
#                  for name, values in titanic_features_dict.items()}
# titanic_preprocessing(features_dict)

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import file_import, pre_processing

train_dataset = file_import.getDataSet()
test_dataset = file_import.getTestSet()

X_train = None
Y_train = train_dataset.pop('Survived')
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
for dataset in combine:
    for i, row in dataset.iterrows():
        # Skip rows that don't have a null Age
        if not(pd.isna(row['Age'])):
            continue

        rowClass = row['Pclass'] - 1
        rowSex = 0 if row['Sex'] == 'male' else 1
        dataset.at[i, 'Age'] = ageMeans[rowClass, rowSex]


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
    ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
test_dataset = test_dataset.drop(
    ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
combine = [train_dataset, test_dataset]

# Now we call the pre_processing file in the utils modules that will do the more advanced data mapping such as
# Indexation of strings, one-hot string lookup, normalization of values and so on
inputs, preprocessed_inputs, feature_columns = pre_processing.preprocess_data(
    train_dataset,
    ['Fare', 'Age'],
    ['Sex', 'Embarked', 'Title']
)

for feature_column in feature_columns:
    print(feature_column)

ds = tf.data.Dataset.from_tensor_slices((dict(train_dataset), Y_train))
ds = ds.shuffle(buffer_size=len(train_dataset))
ds = ds.batch(32)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(ds, epochs=10)

test_ds = tf.data.Dataset.from_tensor_slices((dict(test_dataset), None))

predictions = model.predict(test_ds)

print(predictions)
# preprocessing_model = tf.keras.Model(inputs, preprocessed_inputs)

# X_train = {name: np.array(value) for name, value in train_dataset.items()}

# NUM_EXAMPLES = len(Y_train)

# def make_input_fn(X, y, n_epochs=None, shuffle=True):
#     def input_fn():
#         dataset = tf.data.Dataset.from_tensor_slices((X, y))
#         if shuffle:
#             dataset = dataset.shuffle(NUM_EXAMPLES)
#         # For training, cycle thru dataset as many times as need (n_epochs=None).
#         dataset = dataset.repeat(n_epochs)
#         # In memory training doesn't use batching.
#         dataset = dataset.batch(NUM_EXAMPLES)
#         return dataset
#     return input_fn


# # Training and evaluation input functions.
# train_input_fn = make_input_fn(X_train, Y_train)
# eval_input_fn = make_input_fn(X_train, Y_train, shuffle=False, n_epochs=1)

# print(feature_columns)

# linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# # # Train model.
# linear_est.train(train_input_fn, max_steps=100)
# result = linear_est.evaluate(eval_input_fn)
# print(pd.Series(result))

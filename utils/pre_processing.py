import tensorflow as tf
import numpy as np


def preprocess_data(raw_data, need_norm_inputs=[], need_categorization_inputs=[]):
    symbolic_inputs = create_symbolic_input(raw_data)

    preprocessed_inputs = []
    feature_columns = []

    for key_name in symbolic_inputs:
        if not(key_name in need_norm_inputs) and not(key_name in need_categorization_inputs):
            feature_columns.append(tf.feature_column.numeric_column(
                key_name, dtype=tf.dtypes.float32))
            preprocessed_inputs.append(symbolic_inputs[key_name])

    # GET THE FEATURE COLUMSN FROM NORMALIZE INPUT AND STRING ENCONDING
    preprocessed_inputs.append(
        normalize_numeric_input(symbolic_inputs, raw_data,
                                need_norm_inputs, feature_columns)
    )

    preprocessed_inputs = string_input_to_encoding(
        symbolic_inputs, preprocessed_inputs, raw_data, need_categorization_inputs, feature_columns)

    preprocessed_inputs = tf.keras.layers.Concatenate()(preprocessed_inputs)

    return {
        'inputs': symbolic_inputs,
        'preprocessed_inputs': preprocessed_inputs
    }


def create_symbolic_input(raw_data):
    inputs = {}
    for name, column in raw_data.items():
        dtype = column.dtype

        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    return inputs


def normalize_numeric_input(symbolic_inputs, raw_data, need_norm_inputs, feature_columns):
    numeric_inputs = {
        name: input for name, input in symbolic_inputs.items()
        if input.dtype == tf.float32 and name in need_norm_inputs
    }

    x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
    norm = tf.keras.layers.experimental.preprocessing.Normalization()
    norm.adapt(np.array(raw_data[numeric_inputs.keys()]))

    for key_name in numeric_inputs:
        feature_columns.append(tf.feature_column.numeric_column(
            key_name, dtype=tf.dtypes.float32))

    return {'normalized_inputs': norm(x), 'feature_columns': feature_columns}


def string_input_to_encoding(symbolic_inputs, preprocessed_inputs, raw_data, need_categorization_inputs, feature_columns):
    for name, input in symbolic_inputs.items():
        # Skip non-string column types since they should already be pre-processed in the normalize_numeric_input()
        if input.dtype == tf.float32 or not(name in need_categorization_inputs):
            continue

        # Maps strings from a vocabulary to integer indices https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/StringLookup.
        lookup = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=np.unique(raw_data[name]))
        # Created a one-hot vector categorizer with the size of the vocab from the lookup https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/CategoryEncoding
        one_hot = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
            max_tokens=lookup.vocab_size())

        # Lookup the indices from the input and populate the vocab
        x = lookup(input)
        # Convert the lookup indices into one-hor vector (Example: from a index 3 to [0, 0, 0, 1])
        x = one_hot(x)

        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
            name, np.unique(raw_data[name])))
        preprocessed_inputs.append(x)

    return {preprocessed_inputs, feature_columns}

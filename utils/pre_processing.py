import tensorflow as tf
import numpy as np


def preprocess_data(raw_data):
    symbolic_inputs = create_symbolic_input(raw_data)

    preprocessed_inputs = [
        normalize_numeric_input(symbolic_inputs, raw_data),
        string_input_to_encoding(symbolic_inputs, raw_data)
    ]

    # preprocessed_inputs_concat = tf.keras.layers.Concatenate()(preprocessed_inputs)

    # titanic_preprocessing = tf.keras.Model(raw_data, preprocessed_inputs_concat)

    # tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

    print("Finished Preprocessing")


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


def normalize_numeric_input(symbolic_inputs, raw_data):
    numeric_inputs = {
        name: input for name, input in symbolic_inputs.items()
        if input.dtype == tf.float32
    }

    x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
    norm = tf.keras.layers.experimental.preprocessing.Normalization()
    norm.adapt(np.array(raw_data[numeric_inputs.keys()]))

    return norm(x)


def string_input_to_encoding(symbolic_inputs, raw_data):
    string_inputs = []

    for name, input in symbolic_inputs.items():
        # Skip non-string column types since they should already be pre-processed in the normalize_numeric_input()
        if input.dtype == tf.float32:
            continue

            
        print(raw_data[name])

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

        string_inputs.append(x)

    return string_inputs

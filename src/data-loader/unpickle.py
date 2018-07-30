# Standard libraries
import cPickle
import gzip
import json


def serialize_data(data):
    images, values = data
    return json.dumps(
        dict([
            ("images", images.tolist()),
            ("values", values.tolist())
        ])
    )


"""
The MNIST data is saved as a tuple containing the training data,
the validation data, and the test data.

The ``training_data`` is returned as a tuple with two entries.
The first entry contains the actual training images.  This is a
numpy ndarray with 50,000 entries.  Each entry is, in turn, a
numpy ndarray with 784 values, representing the 28 * 28 = 784
pixels in a single MNIST image.

The second entry in the ``training_data`` tuple is a numpy ndarray
containing 50,000 entries.  Those entries are just the digit
values (0...9) for the corresponding images contained in the first
entry of the tuple.

The ``validation_data`` and ``test_data`` are similar, except
each contains only 10,000 images.
"""


# This script is run from the root of this project, despite it's actual location.
# The file path to the data should also be relative to the root.
f = gzip.open('./data/mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()

# The NPM package `python-shell` listens for data on stdout. Python
print serialize_data(training_data)
print serialize_data(validation_data)
print serialize_data(test_data)

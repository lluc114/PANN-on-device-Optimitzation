# %%'
# MODEL_PATH = '../models/lenet5_notPruned_denses'
MODEL_PATH = '../models/Lenet300100_Fashion_pruned_3'

import os
import sys
import numpy as np
import tensorflow as tf
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

# such a long import might seem an inconvenient but this way it will be easier to
# extend the library in futer to include more NN code generators
from nn_inference.simplest_pruned_mlp_inference_n.builder.NetBuilder import PrunedMLPBuilder

# %% initialize the model builder
nb = PrunedMLPBuilder(MODEL_PATH, out_directory='../out_Lenet300_Fashion_p3')

model = tf.keras.models.load_model(MODEL_PATH)

# %% check and modify model if needed
nb.check_model()

# %% evaluate model on MNIST data sample
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
x = test_images[:1] / 255 # shape is 1, height, width
x = x.reshape((1,-1))

# input_samples = (np.random.random_sample(10)*10000).astype(np.int16)
# samples = test_images[input_samples]
# samples = samples.reshape((10,28,28,1))
# samples = samples/255
# np.save('./fashion_10samples.npy', samples)

# %%
# x_path = MODEL_PATH + '/../x_test_10_Fashion_v2.npy'
# samples = np.load(x_path)
# x = samples

# %% inference using the example data, this is the golden reference for the generated code
# expected_output = nb.model(np.expand_dims(samples[6], 0))

# %%
# x = samples[:1]
expected_output = nb.model(x)
print(expected_output)
# tf.nn.softmax(expected_output)

# %% generate the code
assert type(x) == type(np.array([]))
nb.generate(x, half_precision=False, use_tanh=False)

# %% compile the example code
nb.compile()

# %% run the example code
nb._execute('../out_Lenet300_Fashion_p3/main')
print('TF output:')
print(expected_output.numpy())


# %%

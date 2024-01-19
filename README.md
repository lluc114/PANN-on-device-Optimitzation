# nn-inference
Python based generation of C/C++ libraries to run inference, including sparse (pruned) implementations.

## Running an example

### Example 1: `simplest_pruned_mlp.py`

Change directory to the `examples` folder and run an example to check if the Keras/TensorFlow model is giving the same output:

```
cd examples
python simplest_pruned_mlp.py
```

By default, it is going to generate a `float16` model, i.e. input, weights, and activations are 16-bit floating point numbers. Two types of activation functions are available (Relu and Tanh). The default is Relu.

expected_output = nb.model(x)
print(expected_output)

# generate the code
assert type(x) == type(np.array([]))
nb.generate(x, half_precision=False, use_tanh=False)

# compile the example code
nb.compile()

# run the example code
nb._execute('../out_Lenet300_Fashion_p3/main')
print('TF output:')
print(expected_output.numpy())

Expected output:

```
[...]

C output:

4.295325 13.975469 11.231790 -19.626797 -31.221189 -6.856307 17.750645 -4.327031 -6.929174 1.652373 

TF output:
[[  4.293168   13.977932   11.239425  -19.631868  -31.226746   -6.859765
   17.746508   -4.328249   -6.928644    1.6512883]]
```

Then, you might find the resulting C/C++ program and functions the `out` directory. The main program is `out/main.c` and might modified to capture data from external sources and (cross) compiled for different platforms.

You might also modify these constants in the top of the example python script and use your own model to match the project requirements:



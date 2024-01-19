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

<b>TF output </b>
```
expected_output = nb.model(x)
print(expected_output)
```
<b>generate the code</b>
```
assert type(x) == type(np.array([]))
nb.generate(x, half_precision=False, use_tanh=False)
```
<b>compile the example code</b>
```
nb.compile()
```
<b>run the example code</b>
```
nb._execute('../out_Lenet300_Fashion_p3/main')
print('TF output:')
print(expected_output.numpy())
``` 
Expected output:
```
Running test program for pruned MLP...

C output:
-5.244376 -8.484502 -3.668722 -4.425672 -4.341721 5.066456 -2.277234 9.820605 3.035655 11.868966 

Test program for pruned MLP ended!


TF output:
[[-5.244375  -8.484501  -3.6687226 -4.4256706 -4.3417206  5.06646
  -2.2772343  9.820606   3.0356545 11.868967 ]]
```

Then, you might find the resulting C/C++ program and functions the `out` directory. The main program is `out/main.c` and might modified to capture data from external sources and (cross) compiled for different platforms.

You might also modify these constants in the top of the example python script and use your own model to match the project requirements:



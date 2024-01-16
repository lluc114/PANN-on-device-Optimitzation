// mainh.h
#include <stdio.h>
#include <stdint.h> 

#define N_INPUTS           <N_INPUTS> // number of inputs
// #define N_ACTIVS_H1        <N_ACTIVS_H1> // number of neurons in hidden layer 1
// #define N_ACTIVS_H2        <N_ACTIVS_H2> // number of neurons in hidden layer 2
#define N_ACTIVS_H_TOTAL   <N_ACTIVS_H_TOTAL>
#define N_OUTPUTS          <N_OUTPUTS> // number of outputs

// #define N_WEIGHTS_H1       <N_WEIGHTS_H1>
// #define N_WEIGHTS_H2       <N_WEIGHTS_H2>
// #define N_PARAMETERS_H1    N_WEIGHTS_H1 + N_ACTIVS_H1 // maximum is (N_INPTS + 1) * N_ACTIVS_H1
// #define N_PARAMETERS_H2    N_WEIGHTS_H2 + N_ACTIVS_H2 
// #define N_PARAMETERS_OUT   (N_ACTIVS_H2 + 1) * N_OUTPUTS // set the maximum since readout layer is not pruned

// #define N_PARAMETERS       N_PARAMETERS_H1 + N_PARAMETERS_H2 + N_PARAMETERS_OUT // total number of parameters

// #define USE_BIAS_H1        <USE_BIAS_H1>
// #define USE_BIAS_H2        <USE_BIAS_H2>
// #define USE_BIAS_OUT       <USE_BIAS_OUT>

#define N_HIDDENS          <N_HIDDENS>

//#define DMEM_BASE          0
#define WAIT               {}

const uint16_t neuron_inputs[] = {<NEURON_INPUTS>};
const uint16_t input_indices[] = {<INPUT_INDICES>};

const uint16_t n_weights_H[] = {<N_WEIGHTS_H>};
const bool use_bias_H[] = {<USE_BIAS_H>};
const uint16_t n_activs_H[] = {<N_ACTIVS_H>};
const bool use_tanh = <USE_TANH>;

// memory reserved for weights
const <WEIGHTS_CTYPE> w[] = {
    <WEIGHTS_H>
};

// memory reserved for biases
const <BIAS_CTYPE> b[] = {
    <BIAS_H>
};

// memory reserved for activations
<ACTIVATIONS_CTYPE> a[N_INPUTS+N_ACTIVS_H_TOTAL+N_OUTPUTS];



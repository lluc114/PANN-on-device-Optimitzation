#include "main.h"
#include <stdio.h>
#include <stdint.h> 
#include <math.h>




// #define N_INPUTS           <N_INPUTS> // number of inputs
// // #define N_ACTIVS_H1        <N_ACTIVS_H1> // number of neurons in hidden layer 1
// // #define N_ACTIVS_H2        <N_ACTIVS_H2> // number of neurons in hidden layer 2
// #define N_ACTIVS_H_TOTAL   <N_ACTIVS_H_TOTAL>
// #define N_OUTPUTS          <N_OUTPUTS> // number of outputs

// // #define N_WEIGHTS_H1       <N_WEIGHTS_H1>
// // #define N_WEIGHTS_H2       <N_WEIGHTS_H2>
// // #define N_PARAMETERS_H1    N_WEIGHTS_H1 + N_ACTIVS_H1 // maximum is (N_INPTS + 1) * N_ACTIVS_H1
// // #define N_PARAMETERS_H2    N_WEIGHTS_H2 + N_ACTIVS_H2 
// // #define N_PARAMETERS_OUT   (N_ACTIVS_H2 + 1) * N_OUTPUTS // set the maximum since readout layer is not pruned

// // #define N_PARAMETERS       N_PARAMETERS_H1 + N_PARAMETERS_H2 + N_PARAMETERS_OUT // total number of parameters

// // #define USE_BIAS_H1        <USE_BIAS_H1>
// // #define USE_BIAS_H2        <USE_BIAS_H2>
// // #define USE_BIAS_OUT       <USE_BIAS_OUT>

// #define N_HIDDENS          <N_HIDDENS>

// //#define DMEM_BASE          0
// #define WAIT               {}

// const uint16_t neuron_inputs[] = {<NEURON_INPUTS>};
// const uint16_t input_indices[] = {<INPUT_INDICES>};

// const uint16_t n_weights_H[] = {<N_WEIGHTS_H>};
// const bool use_bias_H[] = {<USE_BIAS_H>};
// const uint16_t n_activs_H[] = {<N_ACTIVS_H>};
// const bool use_tanh = <USE_TANH>;

// // memory reserved for weights
// const <WEIGHTS_CTYPE> w[] = {
//     <WEIGHTS_H>
// };

// // memory reserved for biases
// const <BIAS_CTYPE> b[] = {
//     <BIAS_H>
// };

// // memory reserved for activations
// float a[N_INPUTS+N_ACTIVS_H_TOTAL+N_OUTPUTS];

void compute_hidden(
    size_t activation_ofst,
    size_t bias_ofst,
    size_t out_activation_ofst,
    size_t weight_ofst,
    size_t n_inputs_ofst,
    uint8_t n_H
){
    float a_tmp = 0; // holds current activation value
    for (size_t i = 0; i < n_activs_H[n_H]; i++) // i is the hidden neuron index --> iterate for each hidden neuron
    {
        // dot products
        if (use_bias_H[n_H]){
            a_tmp = b[i+bias_ofst]; // initialize using the bias value
        }else{
            a_tmp = 0;
        }

        for (size_t j = 0; j < neuron_inputs[i+n_inputs_ofst]; j++)
        {
            a_tmp += w[j+weight_ofst] * a[input_indices[j+weight_ofst]+activation_ofst];
        }

        if(use_tanh){
            // tanh activations
            a[i+out_activation_ofst] = tanhf32(a_tmp);
        }else{
            // relu activations
            if (a_tmp > 0) {
                a[i+out_activation_ofst] = a_tmp;
            } else {
                a[i+out_activation_ofst] = 0;
            }
        }

        // update weight offset
        weight_ofst += neuron_inputs[i+n_inputs_ofst];
    }
}

// NN inference code
void mlp_multiple_hidden_layer_pruned() {

    float a_tmp = 0; // holds current activation value

    // ------------------------------------------------------------------------
    // inference code starts HERE, DO NOT MODIFY
    // ------------------------------------------------------------------------
    /*
        HIDDEN LAYER COMPUTATIONS
    */
    // initialize array offsets
    size_t activation_ofst = 0;
    size_t out_activation_ofst = N_INPUTS;
    size_t weight_ofst = 0;
    size_t bias_ofst = 0;
    size_t n_inputs_ofst = 0;

    // compute hidden layer 1 activations
    for (size_t i = 0; i < n_activs_H[0]; i++) // i is the hidden neuron index --> iterate for each hidden neuron
    {
        // dot products
        if(use_bias_H[0]){
            a_tmp = b[i+bias_ofst]; // initialize using the bias value
        }else{
            a_tmp = 0;
        }
            
        for (size_t j = 0; j < neuron_inputs[i+n_inputs_ofst]; j++)
        {
            a_tmp += w[j+weight_ofst] * a[input_indices[j+weight_ofst]+activation_ofst];
        }

        if(use_tanh){
            // tanh activations
            a[i+out_activation_ofst] = tanhf32(a_tmp);
        }else{
            // relu activations
            if (a_tmp > 0) {
                a[i+out_activation_ofst] = a_tmp;
            } else {
                a[i+out_activation_ofst] = 0;
            }
        }
        
        // update weight offset
        weight_ofst += neuron_inputs[i+n_inputs_ofst];
    }
     
    // Hidden layers
    //out_activation_ofst = N_INPUTS;
    activation_ofst = N_INPUTS;
    out_activation_ofst = N_INPUTS + n_activs_H[0];
    bias_ofst = n_activs_H[0]*use_bias_H[0];
    n_inputs_ofst  = n_activs_H[0];
    for (uint8_t n_H = 1; n_H < N_HIDDENS - 1; n_H++)
    {
        // compute hidden layer activations
        // activation_ofst = N_INPUTS;
        // bias_ofst = N_ACTIVS_H1*USE_BIAS_H1;
        // out_activation_ofst = N_INPUTS + N_ACTIVS_H1;
        // weight_ofst = N_WEIGHTS_H1;
        // n_inputs_ofst = N_ACTIVS_H1;

        compute_hidden(activation_ofst, bias_ofst, out_activation_ofst, weight_ofst, n_inputs_ofst, n_H);
                
        activation_ofst += n_activs_H[n_H - 1];
        out_activation_ofst += n_activs_H[n_H];
        bias_ofst += n_activs_H[n_H]*use_bias_H[n_H];
        n_inputs_ofst  += n_activs_H[n_H];
        weight_ofst += n_weights_H[n_H];
    }

    /*
        OUTPUT LAYER COMPUTATIONS
    */
    // modify memory offsets
    // activation_ofst = N_INPUTS + N_ACTIVS_H1;
    // out_activation_ofst = N_INPUTS + N_ACTIVS_H1 + N_ACTIVS_H2;
    // weight_ofst = N_WEIGHTS_H1 + N_WEIGHTS_H2;
    // bias_ofst = N_ACTIVS_H1*USE_BIAS_H1 + N_ACTIVS_H2*USE_BIAS_H2;

    // compute outputs
    printf("C output:\n");
    for (size_t i = 0; i < N_OUTPUTS; i++)
    {
        // dot products
        if(use_bias_H[N_HIDDENS - 1] == 1){
            a_tmp = b[i+bias_ofst]; // initialize using the bias value
        }else{
            a_tmp = 0;
        }

        for (size_t j = 0; j < n_activs_H[N_HIDDENS - 2]; j++)
        {
            a_tmp += w[j+weight_ofst] * a[j+activation_ofst];
        }
        
        // no activation is applied to the output
        a[i+out_activation_ofst] = a_tmp;
        // update weight offset
        weight_ofst += n_activs_H[N_HIDDENS - 2];

        printf("%f ", a[i+out_activation_ofst]);
    }
    printf("\n");

    // ------------------------------------------------------------------------
    // end of inference code
    // ------------------------------------------------------------------------


}

int main() {

    printf("\nRunning test program for pruned MLP...\n\n");

    // some example input
    float x[] = {0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8181818181818p-7,0x1.0101010101010p-8,0x0.0p+0,0x0.0p+0,0x1.c1c1c1c1c1c1cp-6,0x0.0p+0,0x1.2929292929293p-3,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0101010101010p-8,0x1.0101010101010p-7,0x0.0p+0,0x1.b1b1b1b1b1b1bp-4,0x1.5151515151515p-2,0x1.6161616161616p-5,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.ddddddddddddep-2,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0101010101010p-8,0x0.0p+0,0x0.0p+0,0x1.6161616161616p-2,0x1.1f1f1f1f1f1f2p-1,0x1.b9b9b9b9b9b9cp-2,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.6161616161616p-4,0x1.7575757575757p-2,0x1.a9a9a9a9a9a9bp-2,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0101010101010p-6,0x0.0p+0,0x1.a9a9a9a9a9a9bp-3,0x1.0303030303030p-1,0x1.e1e1e1e1e1e1ep-2,0x1.2727272727272p-1,0x1.5f5f5f5f5f5f6p-1,0x1.3b3b3b3b3b3b4p-1,0x1.4d4d4d4d4d4d5p-1,0x1.0f0f0f0f0f0f1p-1,0x1.3535353535353p-1,0x1.5151515151515p-1,0x1.1919191919192p-1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0101010101010p-7,0x0.0p+0,0x1.6161616161616p-5,0x1.1313131313131p-1,0x1.0505050505050p-1,0x1.0101010101010p-1,0x1.4141414141414p-1,0x1.6161616161616p-1,0x1.3f3f3f3f3f3f4p-1,0x1.4f4f4f4f4f4f5p-1,0x1.6565656565656p-1,0x1.2b2b2b2b2b2b3p-1,0x1.2f2f2f2f2f2f3p-1,0x1.2121212121212p-1,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0101010101010p-8,0x0.0p+0,0x1.0101010101010p-7,0x1.0101010101010p-8,0x0.0p+0,0x1.8181818181818p-7,0x0.0p+0,0x0.0p+0,0x1.cdcdcdcdcdcddp-2,0x1.c9c9c9c9c9c9dp-2,0x1.a9a9a9a9a9a9bp-2,0x1.1313131313131p-1,0x1.5151515151515p-1,0x1.3333333333333p-1,0x1.3939393939394p-1,0x1.4b4b4b4b4b4b5p-1,0x1.4f4f4f4f4f4f5p-1,0x1.1f1f1f1f1f1f2p-1,0x1.3b3b3b3b3b3b4p-1,0x1.3d3d3d3d3d3d4p-1,0x1.6161616161616p-5,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0101010101010p-8,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8181818181818p-7,0x0.0p+0,0x0.0p+0,0x1.6565656565656p-2,0x1.1717171717171p-1,0x1.6969696969697p-2,0x1.7979797979798p-2,0x1.3333333333333p-1,0x1.2b2b2b2b2b2b3p-1,0x1.0707070707070p-1,0x1.2f2f2f2f2f2f3p-1,0x1.5353535353535p-1,0x1.5959595959596p-1,0x1.1f1f1f1f1f1f2p-1,0x1.3f3f3f3f3f3f4p-1,0x1.5353535353535p-1,0x1.8181818181818p-3,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0101010101010p-7,0x1.0101010101010p-6,0x1.0101010101010p-8,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8989898989899p-2,0x1.1111111111111p-1,0x1.b9b9b9b9b9b9cp-2,0x1.b5b5b5b5b5b5bp-2,0x1.b9b9b9b9b9b9cp-2,0x1.4545454545454p-1,0x1.0f0f0f0f0f0f1p-1,0x1.2121212121212p-1,0x1.2b2b2b2b2b2b3p-1,0x1.3f3f3f3f3f3f4p-1,0x1.4f4f4f4f4f4f5p-1,0x1.2121212121212p-1,0x1.3d3d3d3d3d3d4p-1,0x1.5353535353535p-1,0x1.ddddddddddddep-2,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0101010101010p-7,0x1.0101010101010p-7,0x1.0101010101010p-8,0x1.0101010101010p-7,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.a1a1a1a1a1a1ap-4,0x1.b1b1b1b1b1b1bp-2,0x1.d5d5d5d5d5d5dp-2,0x1.8d8d8d8d8d8d9p-2,0x1.bdbdbdbdbdbdcp-2,0x1.d5d5d5d5d5d5dp-2,0x1.1111111111111p-1,0x1.3939393939394p-1,0x1.0d0d0d0d0d0d1p-1,0x1.3535353535353p-1,0x1.3535353535353p-1,0x1.3939393939394p-1,0x1.4141414141414p-1,0x1.1b1b1b1b1b1b2p-1,0x1.2727272727272p-1,0x1.3939393939394p-1,0x1.6565656565656p-1,0x0.0p+0,0x1.8181818181818p-7,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.5151515151515p-4,0x1.a9a9a9a9a9a9bp-3,0x1.7171717171717p-2,0x1.d5d5d5d5d5d5dp-2,0x1.bdbdbdbdbdbdcp-2,0x1.9d9d9d9d9d9dap-2,0x1.cdcdcdcdcdcddp-2,0x1.0303030303030p-1,0x1.0d0d0d0d0d0d1p-1,0x1.1f1f1f1f1f1f2p-1,0x1.3535353535353p-1,0x1.4b4b4b4b4b4b5p-1,0x1.5555555555555p-1,0x1.3535353535353p-1,0x1.2f2f2f2f2f2f3p-1,0x1.3535353535353p-1,0x1.1f1f1f1f1f1f2p-1,0x1.1515151515151p-1,0x1.2d2d2d2d2d2d3p-1,0x1.4b4b4b4b4b4b5p-1,0x1.5959595959596p-3,0x0.0p+0,0x0.0p+0,0x1.7171717171717p-4,0x1.b1b1b1b1b1b1bp-3,0x1.0505050505050p-2,0x1.3131313131313p-2,0x1.5555555555555p-2,0x1.d9d9d9d9d9d9ep-2,0x1.0101010101010p-1,0x1.ededededededfp-2,0x1.bdbdbdbdbdbdcp-2,0x1.c5c5c5c5c5c5cp-2,0x1.d9d9d9d9d9d9ep-2,0x1.fdfdfdfdfdfe0p-2,0x1.f5f5f5f5f5f5fp-2,0x1.1717171717171p-1,0x1.0b0b0b0b0b0b1p-1,0x1.1111111111111p-1,0x1.4141414141414p-1,0x1.1919191919192p-1,0x1.3737373737373p-1,0x1.4343434343434p-1,0x1.2121212121212p-1,0x1.3737373737373p-1,0x1.5959595959596p-1,0x1.4343434343434p-1,0x1.7b7b7b7b7b7b8p-1,0x1.f1f1f1f1f1f1fp-3,0x0.0p+0,0x1.1111111111111p-2,0x1.7979797979798p-2,0x1.6969696969697p-2,0x1.bdbdbdbdbdbdcp-2,0x1.c9c9c9c9c9c9dp-2,0x1.bdbdbdbdbdbdcp-2,0x1.c9c9c9c9c9c9dp-2,0x1.cdcdcdcdcdcddp-2,0x1.fdfdfdfdfdfe0p-2,0x1.0f0f0f0f0f0f1p-1,0x1.1111111111111p-1,0x1.1f1f1f1f1f1f2p-1,0x1.f9f9f9f9f9fa0p-2,0x1.fdfdfdfdfdfe0p-2,0x1.2f2f2f2f2f2f3p-1,0x1.3535353535353p-1,0x1.1f1f1f1f1f1f2p-1,0x1.2929292929293p-1,0x1.f5f5f5f5f5f5fp-2,0x1.4545454545454p-1,0x1.4545454545454p-1,0x1.2121212121212p-1,0x1.1515151515151p-1,0x1.3333333333333p-1,0x1.4545454545454p-1,0x1.8989898989899p-1,0x1.d1d1d1d1d1d1dp-3,0x1.1919191919192p-2,0x1.5353535353535p-1,0x1.0303030303030p-1,0x1.a1a1a1a1a1a1ap-2,0x1.8989898989899p-2,0x1.9191919191919p-2,0x1.7979797979798p-2,0x1.8585858585858p-2,0x1.8989898989899p-2,0x1.999999999999ap-2,0x1.b1b1b1b1b1b1bp-2,0x1.a9a9a9a9a9a9bp-2,0x1.ddddddddddddep-2,0x1.e1e1e1e1e1e1ep-2,0x1.0303030303030p-1,0x1.2b2b2b2b2b2b3p-1,0x1.3939393939394p-1,0x1.4f4f4f4f4f4f5p-1,0x1.7d7d7d7d7d7d8p-1,0x1.7d7d7d7d7d7d8p-1,0x1.8989898989899p-1,0x1.8d8d8d8d8d8d9p-1,0x1.8d8d8d8d8d8d9p-1,0x1.7777777777777p-1,0x1.8b8b8b8b8b8b9p-1,0x1.7b7b7b7b7b7b8p-1,0x1.7171717171717p-1,0x1.2121212121212p-3,0x1.0101010101010p-4,0x1.f9f9f9f9f9fa0p-2,0x1.5757575757575p-1,0x1.7979797979798p-1,0x1.7979797979798p-1,0x1.7171717171717p-1,0x1.5757575757575p-1,0x1.3333333333333p-1,0x1.0f0f0f0f0f0f1p-1,0x1.e1e1e1e1e1e1ep-2,0x1.f9f9f9f9f9fa0p-2,0x1.fdfdfdfdfdfe0p-2,0x1.2525252525252p-1,0x1.7373737373737p-1,0x1.8787878787878p-1,0x1.a3a3a3a3a3a3ap-1,0x1.a1a1a1a1a1a1ap-1,0x1.0000000000000p+0,0x1.a3a3a3a3a3a3ap-1,0x1.6363636363636p-1,0x1.ebebebebebebfp-1,0x1.f9f9f9f9f9fa0p-1,0x1.f7f7f7f7f7f7fp-1,0x1.f7f7f7f7f7f7fp-1,0x1.efefefefefeffp-1,0x1.b9b9b9b9b9b9cp-1,0x1.9d9d9d9d9d9dap-1,0x1.8989898989899p-3,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.8181818181818p-5,0x1.0d0d0d0d0d0d1p-2,0x1.a9a9a9a9a9a9bp-2,0x1.4949494949495p-1,0x1.7373737373737p-1,0x1.8f8f8f8f8f8f9p-1,0x1.a5a5a5a5a5a5ap-1,0x1.a7a7a7a7a7a7ap-1,0x1.a5a5a5a5a5a5ap-1,0x1.a1a1a1a1a1a1ap-1,0x1.7d7d7d7d7d7d8p-1,0x1.2d2d2d2d2d2d3p-1,0x1.4949494949495p-2,0x1.0101010101010p-5,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.6565656565656p-1,0x1.a1a1a1a1a1a1ap-1,0x1.7979797979798p-1,0x1.5f5f5f5f5f5f6p-1,0x1.4545454545454p-1,0x1.3d3d3d3d3d3d4p-1,0x1.2f2f2f2f2f2f3p-1,0x1.6161616161616p-5,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0};

    // replace this loop with your preprocessing code
    for (size_t i = 0; i < N_INPUTS; i++)
    {
        a[i] = x[i];
    }

    // NN inference
    mlp_multiple_hidden_layer_pruned();

    printf("\nTest program for pruned MLP ended!\n\n");
}
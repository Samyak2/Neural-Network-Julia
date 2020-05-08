# Neural-Network-Julia

A neural network implementation in Julia

# Features

 - Construct a neural network with any number of layers
 - Only binary classification (for now)
 - Support for training on GPU

# Requirements

 - Julia 1.3
 - TimerOutputs
 - IJulia (for jupyter notebooks given in the `Examples` directory)
 
For GPU support, the following are required (make sure that their tests are passing):
 - CuArrays (tested with v1.7.3)
 - CUDAnative (tested with v2.10.2)
 - CUDAdrv (optional, tested with v6.0.0)

# Usage

Check out the [examples](./Examples) for complete end-to-end training and testing examples.

1. As of now, this has not been made into a Julia package. So you will have to include the `NeuralNetwork.jl` file to use it (also, there will be some JIT overhead during the first use). Download the files in this repo and import the functions by using (make sure to use the complete relative or absolute instead of just `"NeuralNetwork.jl"`):
    ```julia
    include("NeuralNetwork.jl")
    ```

2. Load your training and testing data in the form of `Float32` arrays (the outputs Y can be `Int` also).

3. Use the `neural_network_dense` function to train the Neural Network. See the help section for more info - `?neural_network_dense` should provide the docs. Here is an example usage:
    ```julia
    parameters, activations = neural_network_dense(train_X, train_Y, [12288, 10, 1], 1000, 0.001)
    ```
    
    `[12288, 10, 1]` is the number of neurons in each layer, `1000` is the number of steps to train for and `0.001` is the learning rate.

4. Get predictions and accuracy using the `predict` function. Again, check out the help page for more details - `?predict`. Example usage:
    ```julia
    predicts, accuracy = predict(test_X, test_Y, parameters, activations)
    ```
5. (Optional) To see the time taken for each step in the Neural Network, use
    ```julia
    print_timer(to)
    ```

## GPU Training:

 - `include("NeuralNetworkGPU.jl")`
 - Import the necessary packages - `using CuArrays, CUDAnative, CUDAdrv`
 - Convert the training data to `CuArray`s:

    ```julia
    train_X = CuArray(train_X)
    train_Y = CuArray(train_Y)
    ```
 - Use the same `neural_network_dense` and `predict` functions
 - That's it! (All the optmization for GPU is internal, take a look at [NeuralNetworkGPU.jl](NeuralNetworkGPU.jl))

# TODO

 - [ ] Multi-class classification
 - [ ] Custom GPU kernels for activation functions on GPU

# Acknowledgements

 - [Deeplearning.ai](https://www.deeplearning.ai/)

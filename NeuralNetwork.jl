# using LinearAlgebra
using TimerOutputs

const to = TimerOutput()

include("activations.jl")

"""
    initialize_parameters(layer_dims::Array{Int})::Dict{String, Array{Float64}}

Create and initialize a parameters dict to store parameters of a L-layer feedforward neural network.
The first argument to this function is an array representing number of units in each layer.
The second argument is a placeholder to infer the type of Arrays to initialize - either a normal array or a GPU based CuArray.

Parameters Wi (weights) and bi (biases) are initialized for every layer other than the input layer.
Weights are intialized randomly using Xavier initialization method (to normalize them) and
biases are initialized to zeros.

# Examples
```jldoctest
julia> parameters = initialize_parameters([5 10 1], [])
Dict{String,Array{Float32,N} where N} with 4 entries:
  "W2" => Float32[0.374419 -0.0859404 … -0.203984 -0.303371]
  "W1" => Float32[0.217445 -0.593327 … -0.199775 0.211567; -0.324463 0.611785 … -0.595275 0.668538; … ; -0.105528 0.425086 … -0.706531 0.53281; 0.225039 -0.08…
  "b2" => Float32[0.0]
  "b1" => Float32[0.0; 0.0; … ; 0.0; 0.0]

julia> for (key, value) in parameters
           println(key, " ", size(value))
       end
W2 (1, 10)
W1 (10, 5)
b2 (1, 1)
b1 (10, 1)
```
"""
function initialize_parameters(layer_dims::Array{Int}, Y::Array)::Dict{Integer, Tuple}
    parameters = Dict{Integer, Tuple}()
    for i = 2:length(layer_dims)
        # randomly initialize weights and reduce their magnitude (leads to small weights, prevents gradient explosion)
        @timeit to "Init weights" weights = randn(Float32, layer_dims[i], layer_dims[i-1]) / Float32(sqrt(layer_dims[i-1])) # Xavier initialization
        # initialize biases to zero
        @timeit to "Init biases" biases = zeros(Float32, layer_dims[i], 1)
        parameters[i-1] = (weights, biases)
    end
    return parameters
end

function forward_prop(X::Array{Float32}, parameters::Dict{Integer, Tuple}, activations::Tuple, num_layers)::Array{Float32}#, Dict{String, Array{Float32}}}
    # num_layers = length(parameters)  # number of layers in the network

    # caches = Dict{String, Array{Float32}}()  # init dict to store Z and A values for backprop

    # A0 is X (input)
    # caches[string("A", 0)] = X

    Ai = X

    for i = 1:num_layers
        # Z = W * A_prev + b
        # @timeit to "calculate Zi" begin
        # Wi = parameters[i][1]
            # Ai_prev = caches[string("A", i-1)]
            # Zi = zeros(Float32, size(Wi)[1], size(Ai_prev)[2])
            # mul!(Zi, Wi, Ai_prev)
            println("Size of W $i is $(size(parameters[i][1]))")
            Zi = parameters[i][1] * Ai
            println("Size of Zi is $(size(Zi))")
            Zi = Zi .+ parameters[i][2]
        # end
        # A = activation(Z)
        # @timeit to "calculate Ai" 
        Ai = activations[i].(Zi)
        # store A and Z for use in backprop
        # caches[string("Z", i)] = Zi
        # caches[string("A", i)] = Ai
    end

    return Ai
end

"""
Simple log binary loss
"""
function cost_binary(Y::Array{Float32}, Ŷ::Array{Float32})::Float32
    # @assert length(Y) == length(Ŷ)
    m = length(Y)

    cost = - sum(Y .* log.(Ŷ) .+ (1 .- Y) .* log.(1 .- Ŷ)) / m
    return cost
end

function backward_prop(Y::Array{Float32},
                       Ŷ::Array{Float32},
                       parameters::Dict{Integer, Array{Float32}},
                       caches::Dict{String, Array{Float32}},
                       layer_dims::Array{Int},
                       activations::Tuple)::Dict{String, Array{Float32}}

    num_layers = length(layer_dims)
    @assert length(Y) == length(Ŷ) # verify that predictions and targets have same length
    m = size(Y)[2]

    # formula to get initial gradient
    @timeit to "Initial dA" dA = (.- Y ./ Ŷ .+ (1 .- Y) ./ (1 .- Ŷ))
    if all(isnan.(dA))
        println("dA was NaN!")
        dA = randn(Float32) # TODO: Remove this hack
    end
    # println("dA: ", dA)

    # initialize dict to store gradients
    grads = Dict{String, Array{Float32}}()

    for l in num_layers-1:-1:1
        @timeit to "calculate dZ" dZ = dA .* activations[l].(caches[string("Z", l)])  # dZ = dA * activation(Z)
        @timeit to "calculate dw" grads[string("dw", l)] = 1/m .* (dZ * transpose(caches[string("A", l-1)]))  # dW = 1/m (dZ * A(l-1)')
        @timeit to "calculate db" grads[string("db", l)] = 1/m .* sum(dZ, dims=2)
        @timeit to "calculate dA" dA = transpose(parameters[string("W", l)]) * dZ  # dA = Wl' * dZ
    end

    return grads
end

function update_parameters(parameters::Dict{String, Array{Float32}}, grads::Dict{String, Array{Float32}}, layer_dims::Array{Int}, learning_rate::Number)::Dict{String, Array{Float32}}
    num_layers = length(layer_dims)
    for l = 1:num_layers-1
        # Simple gradient descent
        @timeit to "update dW" parameters[string("W", l)] -= learning_rate .* grads[string("dw", l)]
        @timeit to "update db" parameters[string("b", l)] -= learning_rate .* grads[string("db", l)]
    end
    return parameters
end

"""
Helper functions to get backward activations from activations
"""
function get_back_activations(activations)
    activations_back = []
    for activation in activations
        push!(activations_back, @eval ($(Symbol("$activation", "_back"))))
    end
    activations_back = Tuple(activations_back)
    return activations_back
end

# Prevent Rank 0 array and use Float32 for better consistency
function reshape_Y(Y::Array)
    Y = convert(Array{Float32, ndims(Y)}, Y)
    Y = reshape(Y, 1, :)
    return Y
end

"""
    neural_network_dense(X, Y, layer_dims::Array{Int}, num_iterations::Int, learning_rate::Number; activations=Nothing, print_stats=false, parameters=nothing, resume=false, checkpoint_steps=100)

Build and train a dense neural network for binary classification. Performs batch gradient descent for optimization and uses binary logistic loss for cost function.

Also supports resuming training from previously trained parameters.

Parameters:

- `X`: training inputs (the first value in size of this should be the same as the first value in layer_dims)

- `Y`: training outputs (for binary classification, first value of size should be 1)

- `layer_dims`: Number of neurons in each layer. First layer size should be equal to the number of features in input. Output layer should be 1 in case of binary classification

- `num_iterations`: Number of iterations to train for.

- `learning_rate`: for gradient descent optimizer

- `activations` (optional): A tuple of activation functions for every layer other than the input layer (default: (relu, relu...., sigmoid))

- `print_stats` (optional): Whether to print mean and variance of parameters for every `checkpoint_steps` steps (Statistics package must be imported to use this) (default: false)

- `checkpoint_steps` (optional): the loss (and stats, if print_stats is true) is printed every `checkpoint_steps` steps (default: 100)

- `resume` (optional): whether to resume training by using the given parameters (if parameters are not given, they are initialized again) (default: false)

- `parameters` (optional): parameters for resuming training from checkpoint. Used only if resume is true (default: nothing)

Returns:
- `parameters`: trained parameters
- `activations`: activations used in training (for passing them to `predict` function)
"""
function neural_network_dense(X, Y,
                              layer_dims::Array{Int},
                              num_iterations::Int,
                              learning_rate::Number;
                              activations=Nothing,
                              print_stats=false,
                              parameters=nothing,
                              resume=false,
                              checkpoint_steps=100)
    @timeit to "calculate num layers" num_layers = length(layer_dims) # calculate number of layers

    @timeit to "reshape Y" Y = reshape_Y(Y)
    @assert ndims(Y) == 2

    @timeit to "get back activations" begin
        # if activations are not given, assume that all hidden layers have relu and output layer has sigmoid
        if activations === Nothing
            activations = Array{Function}(undef, num_layers-1)
            for i = 1:num_layers-2
                activations[i] = relu
            end
            activations[num_layers-1] = sigmoid
        end
        activations = Tuple(activations)
        activations_back = get_back_activations(activations)
    end

    @timeit to "Initialize params" begin
        # Check if training has to resume or start over form beginning
        init_params = false
        if !resume
            init_params=true
        elseif (resume && parameters==nothing)
            println("Cannot resume without parameters, pass parameters=parameters to resume training. Reinitializing parameters")
            init_params=true
        end

        # initialize params if it has to
        if init_params
            parameters = initialize_parameters(layer_dims, Y)
        end
    end

    if print_stats
        for i in eachindex(parameters)
            println("\tInitial Mean of parameter ", i, " is ", mean(parameters[i]))
            println("\tInitial Variance of parameter ", i, " is ", var(parameters[i]))
        end
    end

    # pass through the whole dataset num_iterations times
    @timeit to "Actual training" begin
        for iteration = 1:num_iterations
            @timeit to "forward prop" Ŷ, caches = forward_prop(X, parameters, activations)
            @timeit to "backprop" grads = backward_prop(Y, Ŷ, parameters, caches, layer_dims, activations_back)
            @timeit to "update params" parameters = update_parameters(parameters, grads, layer_dims, learning_rate)

            @timeit to "print stats" begin
                # print stats every few steps
                if iteration % checkpoint_steps == 0
                    cost = cost_binary(Y, Ŷ)
                    println("Cost at iteration $iteration is $cost")
                    if print_stats
                        for i in eachindex(parameters)
                            println("\tMean of parameter ", i, " is ", mean(parameters[i]))
                            println("\tVariance of parameter ", i, " is ", var(parameters[i]))
                        end
                    end
                end
            end
        end
    end

    return parameters, activations
end

"""
    predict(X, Y, parameters, activations::Tuple)

Predict using the trained parameters and calculate the accuracy.

Parameters:

- `X`: testing inputs
- `Y`: outputs to check with
- `parameters`: trained parameters which is taken from the output of `neural_network_dense`
- `activations`: also given by the outputs of `neural_network_dense`

Returns:

- `predicts`: predictions from the NN
- `accuracy`: accuracy of predictions
"""
function predict(X, Y, parameters, activations::Tuple)
    m = size(X)[2]
    n = length(parameters)
    predicts = zeros((1, m))

    # Copy Y to CPU
    if Y != nothing
        @timeit to "copy to CPU" Y = Array(Y)
        Y = reshape_Y(Y)
    end

    @timeit to "forward prop" probas, caches = forward_prop(X, parameters, activations)
    @timeit to "calculate probas" begin
        probas = Array(probas)

        for i = 1 : m
            if probas[1, i] > 0.5f0
                predicts[1, i] = 1
            else
                predicts[1, i] = 0
            end
        end
    end

    accuracy = nothing
    if Y != nothing
        @timeit to "calculate accuracy" accuracy = sum(predicts .== Y) / m
        println("Accuracy is ", accuracy*100, "%")
    end

    return predicts, accuracy
end

include("activations.jl")

"""
    initialize_parameters(layer_dims::Array{Int})::Dict{String, Array{Float64}}

Create and initialize a parameters dict to store parameters of a L-layer feedforward neural network.
The single argument to this function is an array representing number of units in each layer.

Parameters Wi (weights) and bi (biases) are initialized for every layer other than the input layer.
Weights are intialized randomly and biases are initialized to zeros.

# Examples
```jldoctest
julia> parameters = initialize_parameters([5 10 1])
Dict{String,Array{Float64,N} where N} with 4 entries:
  "W2" => [-0.00305799 0.0119367 … -0.00322732 0.00213387]
  "W1" => [-0.00599823 -0.0130222 … 0.00449422 0.00915462; 0.0145317 0.0131843 … -0.00216658 0.0102101; … ; -0.00749814 0.00917309 … 0.00354458 -0.00476792; 0.00293553 0.0156417 … -0.00641187 0.0160924]
  "b2" => [0.0]
  "b1" => [0.0; 0.0; … ; 0.0; 0.0]

julia> for (key, value) in parameters
           println(key, " ", size(value))
       end
W2 (1, 10)
W1 (10, 5)
b2 (1, 1)
b1 (10, 1)
```
"""
function initialize_parameters(layer_dims::Array{Int})::Dict{String, Array{Float32}}
    parameters = Dict{String, Array{Float32}}()
    for i = 2:length(layer_dims)
        parameters[string("W", i-1)] = randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters[string("b", i-1)] = zeros(layer_dims[i], 1)
    end
    return parameters
end

function forward_prop(X::Array{Float32}, parameters::Dict{String, Array{Float32}}, layer_dims::Array{Int}, activations::Tuple)::Tuple{Array{Float32}, Dict{String, Array{Float32}}}
    num_layers = length(layer_dims)

    caches = Dict{String, Array{Float64}}()

    caches[string("A", 0)] = X

    Ai = Nothing
    for i = 1:num_layers-1
        Zi = parameters[string("W", i)] * caches[string("A", i-1)] .+ parameters[string("b", i)]
        Ai = activations[i].(Zi)
        caches[string("Z", i)] = Zi
        caches[string("A", i)] = Ai
    end

    return Ai, caches
end

function cost_binary(Y::Array{Float32}, Ŷ::Array{Float32})::Float32
    @assert length(Y) == length(Ŷ)
    m = length(Y)

    cost = - sum(Y .* log.(Ŷ) .+ (1 .- Y) .* log.(1 .- Ŷ)) / m
    # println("cost is ", string(cost, Y[1:30], Ŷ[1:30]))
    return cost
end

function backward_prop(Y::Array{Float32}, Ŷ::Array{Float32}, parameters::Dict{String, Array{Float32}}, caches::Dict{String, Array{Float32}}, layer_dims::Array{Int}, activations::Tuple)::Dict{String, Array{Float32}}

    num_layers = length(layer_dims)
    @assert length(Y) == length(Ŷ)
    m = size(Y)[2]
    # println("Number of examples(m) ", m)

    # activations_orig = activations
    # activations = []
    # for activation in activations_orig
    #     push!(activations, @eval ($(Symbol("$activation", "_back"))))
    # end
    # println(activations)

    dA = sum(.- Y ./ Ŷ .+ (1 .- Y) ./ (1 .- Ŷ)) / m
    if isnan(dA)
        println("dA was NaN!")
        dA = randn(Float32)
    end
    # println("dA: ", dA)

    grads = Dict{String, Array{Float32}}()

    for l in num_layers-1:-1:1
        dZ = dA .* activations[l].(caches[string("Z", l)])
        grads[string("dw", l)] = 1/m .* (dZ * transpose(caches[string("A", l-1)]))
        grads[string("db", l)] = 1/m .* sum(dZ, dims=2)
        dA = transpose(parameters[string("W", l)]) * dZ
    end

    return grads
end

function update_parameters(parameters::Dict{String, Array{Float32}}, grads::Dict{String, Array{Float32}}, layer_dims::Array{Int}, learning_rate::Number)::Dict{String, Array{Float32}}
    num_layers = length(layer_dims)
    for l = 1:num_layers-1
        parameters[string("W", l)] -= learning_rate .* grads[string("dw", l)]
        parameters[string("b", l)] -= learning_rate .* grads[string("db", l)]
        # if l > 1
        #     println("dw ", l, grads[string("dw", l)])
        #     println("db ", l, grads[string("db", l)])
        # end
    end
    return parameters
end

function neural_network_dense(X, Y, layer_dims::Array{Int}, num_iterations::Int, learning_rate::Number, activations=Nothing)
    num_layers = length(layer_dims) # calculate number of layers

    Y = convert(Array{Float32, 1}, Y)
    Y = reshape(Y, 1, :)

    # if activations are not given, assume that all hidden layers have relu and output layer has sigmoid
    if activations === Nothing
        activations = Array{Function}(undef, num_layers-1)
        for i = 1:num_layers-2
            activations[i] = relu
        end
        activations[num_layers-1] = sigmoid
    end
    activations = Tuple(activations)
    println(activations)
    activations_back = []
    for activation in activations
        push!(activations_back, @eval ($(Symbol("$activation", "_back"))))
    end
    activations_back = Tuple(activations_back)
    println(activations_back)

    parameters = initialize_parameters(layer_dims)

    for iteration = 1:num_iterations
        Ŷ, caches = forward_prop(X, parameters, layer_dims, activations)
        grads = backward_prop(Y, Ŷ, parameters, caches, layer_dims, activations_back)
        parameters = update_parameters(parameters, grads, layer_dims, learning_rate)

        if iteration % 1 == 0
            cost = cost_binary(Y, Ŷ)
            println("Cost at iteration $iteration is $cost")
        end
    end

    return parameters
end

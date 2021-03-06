# n_x = 5
# n_h = 4
# n_y = 1

# W1 = randn(n_h, n_x)
# println(W1)

import Random
Random.seed!(210)

# function initialize_parameters(layer_dims)
#     parameters = Dict{String, Array{Float64}}()
#     for i = 2:length(layer_dims)
#         parameters[string("W", i-1)] = randn(layer_dims[i], layer_dims[i-1]) * 0.01
#         parameters[string("b", i-1)] = zeros(layer_dims[i], 1)
#     end
#     return parameters
# end

include("NeuralNetwork.jl")

parameters = initialize_parameters([5 10 1])

function printdict(dict)
    for (key, value) in dict
        println(key, " ", size(value))
    end
end
function printdict_full(dict)
    for (key, value) in dict
        println(key, " ", (value))
    end
end
printdict(parameters)

# using LinearAlgebra

# function sigmoid(X)
#     return 1 ./ (1 .+ exp.(-X))
# end
# function sigmoid(x)
#     return 1 / (1 + exp(-x))
# end

# function relu(x)
#     if x>0
#         return x
#     else
#         return 0
#     end
# end

# function forward_prop(X::Array{Float64}, parameters::Dict{String, Array{Float64}}, layer_dims::Array{Int}, activations=Nothing)
#     num_layers = length(layer_dims)

#     if activations === Nothing
#         activations = Dict()
#         for i = 1:num_layers-1
#             # println("relu")
#             activations[i] = relu
#         end
#         # println(sigmoid)
#         activations[num_layers] = sigmoid
#     end
#     # println(activations)
#     caches = Dict()

#     Z1 = parameters[string("W", 1)] * X .+ parameters[string("b", 1)]
#     A1 = activations[1].(Z1)
#     caches[string("Z", 1)] = Z1
#     caches[string("A", 1)] = A1

#     Ai = Nothing
#     for i = 2:num_layers-1
#         Zi = parameters[string("W", i)] * caches[string("A", i-1)] .+ parameters[string("b", i)]
#         Ai = activations[i].(Zi)
#         caches[string("Z", i)] = Zi
#         caches[string("A", i)] = Ai
#     end

#     return Ai, caches
# end


X = randn(5, 100)
a2, caches = forward_prop(X, parameters, [5 10 1], (relu, sigmoid))
# a2, caches = forward_prop(X, parameters, [5 10 1])
println(a2)
# println(caches)

# function cost_binary(Y, Ŷ)
#     @assert length(Y) == length(Ŷ)
#     m = length(Y)

#     cost = - sum(Y .* log.(Ŷ) .+ (1 .- Y) .* log.(1 .- Ŷ)) / m
# end

Y = [0.777085  0.248755  0.660713  0.37982  0.978839  0.182363  0.07204  0.855448  0.37393  0.40736]
@assert cost_binary(Y, Y) == 0.4982477807320199

# function sigmoid_back(x)
#     sig = sigmoid(x)
#     return sig*(1-sig)
# end

# function relu_back(x)
#     if x>0
#         return 1
#     else
#         return 0
#     end
# end

# function backward_prop(Y::Array{Float64}, Ŷ::Array{Float64}, parameters::Dict{String, Array{Float64}}, caches::Dict{String, Array{Float64}}, layer_dims::Array{Int}, activations=Nothing)

#     num_layers = length(layer_dims)
#     @assert length(Y) == length(Ŷ)
#     m = size(Y)[2]

#     if activations === Nothing
#         activations = Dict()
#         for i = 1:num_layers-1
#             activations[i] = relu_back
#         end
#         activations[num_layers] = sigmoid_back
#     else
#         activations_orig = activations
#         activations = []
#         for activation in activations_orig
#             push!(activations, @eval ($(Symbol("$activation", "_back"))))
#         end
#     end
#     # println(activations)

#     dA = sum(.- Y ./ Ŷ .+ (1 .- Y) ./ (1 .- Ŷ))

#     grads = Dict{String, Array{Float64}}()

#     for l in num_layers-1:-1:1
#         dZ = dA .* activations[l].(caches[string("Z", l)])
#         grads[string("dw", l)] = 1/m .* (dZ * transpose(caches[string("A", l-1)]))
#         grads[string("db", l)] = 1/m .* sum(dZ, dims=2)
#         dA = transpose(parameters[string("W", l)]) * dZ
#     end

#     return grads
# end

# grads = backward_prop(randn(1, 100), a2, parameters, caches, [5 10 1])
# printdict(grads)
# @assert size(grads["dw1"]) == size(parameters["W1"])
grads = backward_prop(randn(1, 100), a2, parameters, caches, [5 10 1], [relu, sigmoid])
printdict(grads)
@assert size(grads["dw1"]) == size(parameters["W1"])

# function update_parameters(parameters::Dict{String, Array{Float64}}, grads::Dict{String, Array{Float64}}, layer_dims::Array{Int}, learning_rate::Float64)
#     num_layers = length(layer_dims)
#     for l = 1:num_layers-1
#         parameters[string("W", l)] += learning_rate .* grads[string("dw", l)]
#         parameters[string("b", l)] += learning_rate .* grads[string("db", l)]
#     end
#     return parameters
# end

printdict_full(parameters)
update_parameters(parameters, grads, [5 10 1], 0.01)
printdict_full(parameters)

# function neural_network_dense(X, Y, layer_dims::Array{Int}, num_iterations::Int, learning_rate::Float64, activations=Nothing)
#     num_layers = length(layer_dims) # calculate number of layers
#
#     # if activations are not given, assume that all hidden layers have relu and output layer has sigmoid
#     if activations === Nothing
#         activations = Array{Function}(undef, num_layers-1)
#         for i = 1:num_layers-2
#             activations[i] = relu
#         end
#         activations[num_layers-1] = sigmoid
#     end
#     activations = Tuple(activations)
#     println(activations)
#     activations_back = []
#     for activation in activations
#         push!(activations_back, @eval ($(Symbol("$activation", "_back"))))
#     end
#     activations_back = Tuple(activations_back)
#     println(activations_back)
#
#     parameters = initialize_parameters(layer_dims)
#
#     for iteration = 1:num_iterations
#         Ŷ, caches = forward_prop(X, parameters, layer_dims, activations)
#         grads = backward_prop(Y, Ŷ, parameters, caches, layer_dims, activations_back)
#         parameters = update_parameters(parameters, grads, layer_dims, learning_rate)
#
#         if iteration % 1 == 0
#             cost = cost_binary(Y, Ŷ)
#             println("Cost at iteration $iteration is $cost")
#         end
#     end
#
#     return parameters
# end

X = randn(10, 100)
Y = randn(1, 100)
layer_dims = [10 10 1]
neural_network_dense(X, Y, layer_dims, 100, 0.05)

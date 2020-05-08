using Zygote
X = [0 0 1 1; 0 1 0 1]
X = Array{Float32, 2}(X)

Y = [0, 1, 1, 0]
num_features = 2
num_hidden_layer_neurons = 10

layer_dims = [num_features, num_hidden_layer_neurons, 1]

include("../NeuralNetwork.jl")

num_layers = length(layer_dims)

parameters = initialize_parameters(layer_dims, Y)

activations = Nothing
activations = Array{Function}(undef, num_layers-1)
for i = 1:num_layers-2
   activations[i] = relu
end
activations[num_layers-1] = sigmoid

activations = Tuple(activations)
activations_back = get_back_activations(activations)

Y = reshape_Y(Y)

gradient((parameters) -> cost_binary(Y, forward_prop(X, parameters, activations)), parameters)


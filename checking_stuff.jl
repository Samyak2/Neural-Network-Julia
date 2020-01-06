# n_x = 5
# n_h = 4
# n_y = 1

# W1 = randn(n_h, n_x)
# println(W1)

function initialize_parameters(layer_dims)
    parameters = Dict()
    for i = 2:length(layer_dims)
        parameters[string("W", i-1)] = randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters[string("b", i-1)] = zeros(layer_dims[i], 1)
    end
    return parameters
end

a = initialize_parameters([5 10 1])

for (key, value) in a
    println(key, " ",size(value))
end

# using LinearAlgebra

function forward_prop(X, parameters, layer_dims)
    caches = Dict()

    Z1 = parameters[string("W", 1)] * X + parameters[string("b", 1)]
    caches[string("Z", 1)] = Z1
    caches[string("A", 1)] = relu(caches[])

    for i = 2:length(layer_dims)
        caches[string("Z", i)] = parameters[string("W", i)]*parameters[string("W", i)]
    end
end

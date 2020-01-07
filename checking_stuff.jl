# n_x = 5
# n_h = 4
# n_y = 1

# W1 = randn(n_h, n_x)
# println(W1)

function initialize_parameters(layer_dims)
    parameters = Dict{String, Array{Float64}}()
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

# function sigmoid(X)
#     return 1 ./ (1 .+ exp.(-X))
# end
function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function relu(x)
    if x>0
        return x
    else
        return 0
    end
end

function forward_prop(X::Array{Float64}, parameters::Dict{String, Array{Float64}}, layer_dims::Array{Int}, activations::Vector{Function}=Nothing)
    num_layers = length(layer_dims)

    if activations === Nothing
        activations = Dict()
        for i = 1:num_layers-1
            println("relu")
            activations[string(i)] = relu
        end
        println(sigmoid)
        activations[string(num_layers)] = sigmoid
    end
    println(activations)
    caches = Dict()

    Z1 = parameters[string("W", 1)] * X + parameters[string("b", 1)]
    A1 = activations[1].(Z1)
    caches[string("Z", 1)] = Z1
    caches[string("A", 1)] = A1

    Ai = Nothing
    for i = 2:num_layers-1
        Zi = parameters[string("W", i)] * caches[string("A", i-1)] + parameters[string("b", i)]
        Ai = activations[i].(Zi)
        caches[string("Z", i)] = Zi
        caches[string("A", i)] = Ai
    end

    return Ai, caches
end


X = randn(5, 1)
a2 = forward_prop(X, a, [5 10 1], [relu, relu, sigmoid])
println(a2)

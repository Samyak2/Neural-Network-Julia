include("activations.jl")

function initialize_parameters(layer_dims)
    parameters = Dict{String, Array{Float64}}()
    for i = 2:length(layer_dims)
        parameters[string("W", i-1)] = randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters[string("b", i-1)] = zeros(layer_dims[i], 1)
    end
    return parameters
end

function forward_prop(X::Array{Float64}, parameters::Dict{String, Array{Float64}}, layer_dims::Array{Int}, activations=Nothing)
    num_layers = length(layer_dims)

    if activations === Nothing
        activations = Dict()
        for i = 1:num_layers-1
            # println("relu")
            activations[i] = relu
        end
        # println(sigmoid)
        activations[num_layers] = sigmoid
    end
    # println(activations)
    caches = Dict()

    Z1 = parameters[string("W", 1)] * X .+ parameters[string("b", 1)]
    A1 = activations[1].(Z1)
    caches[string("Z", 1)] = Z1
    caches[string("A", 1)] = A1

    Ai = Nothing
    for i = 2:num_layers-1
        Zi = parameters[string("W", i)] * caches[string("A", i-1)] .+ parameters[string("b", i)]
        Ai = activations[i].(Zi)
        caches[string("Z", i)] = Zi
        caches[string("A", i)] = Ai
    end

    return Ai, caches
end

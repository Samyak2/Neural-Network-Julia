include("activations.jl")

function initialize_parameters(layer_dims::Array{Int})::Dict{String, Array{Float64}}
    parameters = Dict{String, Array{Float64}}()
    for i = 2:length(layer_dims)
        parameters[string("W", i-1)] = randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters[string("b", i-1)] = zeros(layer_dims[i], 1)
    end
    return parameters
end

function forward_prop(X::Array{Float64}, parameters::Dict{String, Array{Float64}}, layer_dims::Array{Int}, activations=Nothing)::Tuple{Array{Float64}, Dict{String, Array{Float64}}}
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

function cost_binary(Y::Array{Float64}, Ŷ::Array{Float64})::Float64
    @assert length(Y) == length(Ŷ)
    m = length(Y)

    cost = - sum(Y .* log.(Ŷ) .+ (1 .- Y) .* log.(1 .- Ŷ)) / m
end

function backward_prop(Y::Array{Float64}, Ŷ::Array{Float64}, parameters::Dict{String, Array{Float64}}, caches::Dict{String, Array{Float64}}, layer_dims::Array{Int}, activations=Nothing)::Dict{String, Array{Float64}}

    num_layers = length(layer_dims)
    @assert length(Y) == length(Ŷ)
    m = size(Y)[2]

    activations_orig = activations
    activations = []
    for activation in activations_orig
        push!(activations, @eval ($(Symbol("$activation", "_back"))))
    end
    # println(activations)

    dA = sum(.- Y ./ Ŷ .+ (1 .- Y) ./ (1 .- Ŷ))

    grads = Dict{String, Array{Float64}}()

    for l in num_layers-1:-1:1
        dZ = dA .* activations[l].(caches[string("Z", l)])
        grads[string("dw", l)] = 1/m .* (dZ * transpose(caches[string("A", l-1)]))
        grads[string("db", l)] = 1/m .* sum(dZ, dims=2)
        dA = transpose(parameters[string("W", l)]) * dZ
    end

    return grads
end

function update_parameters(parameters::Dict{String, Array{Float64}}, grads::Dict{String, Array{Float64}}, layer_dims::Array{Int}, learning_rate::Float64)::Dict{String, Array{Float64}}
    num_layers = length(layer_dims)
    for l = 1:num_layers-1
        parameters[string("W", l)] += learning_rate .* grads[string("dw", l)]
        parameters[string("b", l)] += learning_rate .* grads[string("db", l)]
    end
    return parameters
end

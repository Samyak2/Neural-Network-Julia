include("activationsGPU.jl")
include("NeuralNetwork.jl")
using CuArrays, Statistics

"""
    initialize_parameters(layer_dims::CuArray{Int})::Dict{String, CuArray{Float64}}

Create and initialize a parameters dict to store parameters of a L-layer feedforward neural network.
The first argument to this function is an array representing number of units in each layer.
The second argument is a placeholder to infer the type of Arrays to initialize - either a normal array or a GPU based CuArray.

Parameters Wi (weights) and bi (biases) are initialized for every layer other than the input layer.
Weights are intialized randomly using Xavier initialization method (to normalize them) and
biases are initialized to zeros.

# Examples
```jldoctest
julia> parameters = initialize_parameters([5 10 1], CuArray([0]))
Dict{String,CuArray{Float32,N,P} where P where N} with 4 entries:
  "W2" => Float32[-0.00542962 0.00721382 … 0.185659 -0.355868]
  "W1" => Float32[0.679642 -0.365945 … 0.145359 -0.47302; 0.200724 -0.339901 … 1.11603 0.551288; … ; -0.93578 -0.0143422 … 0.433826 -0.425189; -0.70129 -0.077…
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
function initialize_parameters(layer_dims::Array{Int}, Y::CuArray)::Dict{String, CuArray{Float32}}
    parameters = Dict{String, CuArray{Float32}}()
    for i = 2:length(layer_dims)
        parameters[string("W", i-1)] = CuArrays.randn(layer_dims[i], layer_dims[i-1]) ./ sqrt(layer_dims[i-1]) # Xavier initialization
        parameters[string("b", i-1)] = CuArrays.zeros(layer_dims[i], 1)
    end
    return parameters
end

function forward_prop(X::CuArray{Float32}, parameters::Dict{String, CuArray{Float32}}, activations::Tuple)::Tuple{CuArray{Float32}, Dict{String, CuArray{Float32}}}
    num_layers = length(parameters) ÷ 2

    caches = Dict{String, CuArray{Float32}}()

    caches[string("A", 0)] = X

    Ai = Nothing
    for i = 1:num_layers
        Zi = parameters[string("W", i)] * caches[string("A", i-1)] .+ parameters[string("b", i)] # * is equivivalent to np.dot
        Ai = activations[i].(Zi)
        caches[string("Z", i)] = Zi
        caches[string("A", i)] = Ai
    end

    return Ai, caches
end

function cost_binary(Y::CuArray{Float32}, Ŷ::CuArray{Float32})::Float32
    @assert length(Y) == length(Ŷ)
    m = length(Y)

    cost = - sum(Y .* log.(Ŷ) .+ (1 .- Y) .* log.(1 .- Ŷ)) / m
    # println("cost is ", string(cost, Y[1:30], Ŷ[1:30]))
    return cost
end

function backward_prop(Y::CuArray{Float32}, Ŷ::CuArray{Float32}, parameters::Dict{String, CuArray{Float32}}, caches::Dict{String, CuArray{Float32}}, layer_dims::Array{Int}, activations::Tuple)::Dict{String, CuArray{Float32}}

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

    dA = (.- Y ./ Ŷ .+ (1 .- Y) ./ (1 .- Ŷ))
    if all(isnan.(dA))
        println("dA was NaN!")
        dA = randn(Float32)
    end
    # println("dA: ", dA)

    grads = Dict{String, CuArray{Float32}}()

    for l in num_layers-1:-1:1
        dZ = dA .* activations[l].(caches[string("Z", l)])
        grads[string("dw", l)] = 1/m .* (dZ * transpose(caches[string("A", l-1)]))
        grads[string("db", l)] = 1/m .* sum(dZ, dims=2)
        dA = transpose(parameters[string("W", l)]) * dZ
    end

    return grads
end



function update_parameters_old(parameters::Dict{String, CuArray{Float32}}, grads::Dict{String, CuArray{Float32}}, layer_dims::Array{Int}, learning_rate::Number)::Dict{String, CuArray{Float32}}
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

function update_params_kernel!(params, lr, grads, m, n)
    tx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    ty = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if tx <= m && ty <= n
        @inbounds params[tx, ty] = params[tx, ty] - lr * grads[tx, ty]
    end
    return nothing
end

function get_threads_blocks(m, n)
    nthreads = 16
    threads = (min(nthreads, m), min(nthreads, n))
    blocks = (cld(m, threads[1]), cld(n, threads[2]))
    return threads, blocks
end

function update_parameters(parameters::Dict{String, CuArray{Float32}}, grads::Dict{String, CuArray{Float32}}, layer_dims::Array{Int}, learning_rate::Number)::Dict{String, CuArray{Float32}}
    num_layers = length(layer_dims)
    for l = 1:num_layers-1
        m,n = size(parameters[string("W", l)])
        threads, blocks = get_threads_blocks(m, n)
        @cuda blocks=blocks threads=threads update_params_kernel!(parameters[string("W", l)], learning_rate, grads[string("dw", l)], m, n)
        
        m,n = size(parameters[string("b", l)])
        threads, blocks = get_threads_blocks(m, n)
        @cuda blocks=blocks threads=threads update_params_kernel!(parameters[string("b", l)], learning_rate, grads[string("db", l)], m, n)
    end
    synchronize()
    return parameters
end

function reshape_Y(Y::CuArray)
    Y = convert(CuArray{Float32, ndims(Y)}, Y)
    Y = reshape(Y, 1, :)
    return Y
end

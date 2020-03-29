using CuArrays

# TODO: custom kernel for activations on GPU

function sigmoid(x::Float32)::Float32
    return 1 / (1 + CuArrays.exp(-x))
end

function sigmoid_back(x::Float32)::Float32
    sig = sigmoid(x)
    return sig*(1-sig)
end


function relu(x::Float32)::Float32
    if x>0
        return x
    else
        return 0
    end
end

function relu_back(x::Float32)::Float32
    if x>0
        return 1
    else
        return 0
    end
end

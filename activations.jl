function sigmoid(x::Float64)::Float64
    return 1 / (1 + exp(-x))
end

function sigmoid_back(x::Float64)::Float64
    sig = sigmoid(x)
    return sig*(1-sig)
end


function relu(x::Float64)::Float64
    if x>0
        return x
    else
        return 0
    end
end

function relu_back(x::Float64)::Float64
    if x>0
        return 1
    else
        return 0
    end
end

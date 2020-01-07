function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function sigmoid_back(x)
    sig = sigmoid(x)
    return sig*(1-sig)
end


function relu(x)
    if x>0
        return x
    else
        return 0
    end
end

function relu_back(x)
    if x>0
        return 1
    else
        return 0
    end
end
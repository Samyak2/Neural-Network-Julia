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

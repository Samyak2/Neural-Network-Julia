# this function makes sure that the number of examples in each class are the same
function equalize_inputs(train_X::Array{Float32,2}, train_Y::Array{Int,2}, num_classes)
    train_X_len = size(train_X)[2]
    train_Y_len = size(train_Y)[2]
    @assert train_X_len == train_Y_len
    len = train_X_len
    
    class_counter = zeros(Int, num_classes)
    for i in 1:len
        class_counter[train_Y[1, i]+1] += 1
    end
    println("Total entries for each class: ", class_counter)
    max_per_class = minimum(class_counter)
    
    new_train_X = Array{Float32, 2}(undef, size(train_X)[1], min(train_X_len, max_per_class*num_classes))
    new_train_Y = Array{Float32, 2}(undef, size(train_Y)[1], min(train_Y_len, max_per_class*num_classes))
    
    counters = zeros(Int, num_classes)
    iter = 1
    new_counter = 1
    while (iter < len) && any(counters .< max_per_class) # Exits when all of the counters are more than the max limit
#         println(counters, new_counter)
        if counters[train_Y[1, iter]+1] < max_per_class
            counters[train_Y[1, iter]+1] += 1
            new_train_X[:, new_counter] = train_X[:, iter]
            new_train_Y[:, new_counter] = train_Y[:, iter]
            new_counter += 1
        end
        iter += 1
    end
    println("Counters: ", counters)
    println("Total entries added: ", new_counter)
    println("Total entries checked: ", iter)
    return new_train_X, new_train_Y
end

# Example usage:
# new_train_X, new_train_Y = equalize_inputs(train_X, train_Y, 2)
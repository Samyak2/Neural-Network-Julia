using CSV
using DataFrames
using JLD
using Images
cd("Examples/Cactus")
input_file = "train.csv"
df = CSV.read(input_file)

first(df, 10)
describe(df)
df = sort(df, [:id])
images = df[!, :id]
train_Y = df[!, :has_cactus]
train_data = load("dataset.jld")
train_data

# d = train_data["test_X_orig"]
# size(d)[4]
# vcat(d[:, :, :, 1]...)
# for i = 0 : size(d)[4]
#     d[:, :, :, i] =
# end
# d2 = (reshape(d, :, size(d)[4]))
# d2[:, 1]

function reshape_image_array(image_array::Array{Float32,4})
    return reshape(image_array, :, size(image_array)[4])
end
function reshape_image_array_reverse(image_array::Array{Float32,2})
    return reshape(image_array, 32, 32, 3, :)
end

test_X = reshape_image_array(train_data["test_X_orig"])
train_X = reshape_image_array(train_data["train_X_orig"])
train_Y = reshape(train_Y, (1, size(train_Y)[1]))
# test_Y = reshape(test_Y, (1, size(test_Y)[1]))

include("../../NeuralNetwork.jl")
# neural_network_dense(train_X, train_Y, [3072, 1, 1], 0.05)
parameters, activations = neural_network_dense(train_X, train_Y, [3072, 10, 10, 1], 100, 0.1)
predicts, accuracy = predict(train_X, train_Y, parameters, activations)

a = reshape_image_array_reverse(train_X[:, [1,7]])
a2 = a[:,:,:,2]
a3 = permuteddimsview(a2, [3, 1, 2])
a3 = Array{Float32}(a3)
colorview(RGB, a3)

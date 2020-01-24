using CSV
using DataFrames
using JLD

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

test_X = reshape_image_array(train_data["test_X_orig"])
train_X = reshape_image_array(train_data["train_X_orig"])

include("../../NeuralNetwork.jl")
neural_network_dense(train_X, train_Y, [3072, 1, 1], 0.01)

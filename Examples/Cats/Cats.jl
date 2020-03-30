using HDF5

TRAIN_DATA_PATH="datasets/train_catvnoncat.h5"
TEST_DATA_PATH="datasets/test_catvnoncat.h5"

train_data_x_orig = h5read(TRAIN_DATA_PATH, "train_set_x")
train_data_y_orig = h5read(TRAIN_DATA_PATH, "train_set_y")

test_data_x_orig = h5read(TEST_DATA_PATH, "test_set_x")
test_data_y_orig = h5read(TEST_DATA_PATH, "test_set_y")

classes = h5read(TEST_DATA_PATH, "list_classes")

train_data_y_orig = reshape(train_data_y_orig, (1, size(train_data_y_orig)[1]))

test_data_y_orig = reshape(test_data_y_orig, (1, size(test_data_y_orig)[1]))

function reshape_image_array(image_array::Array{UInt8,4})
    return reshape(image_array, :, size(image_array)[4])
end
function reshape_image_array_reverse(image_array::Array{UInt8,2})
    return reshape(image_array, 64, 64, 3, :)
end

test_data_x = reshape_image_array(test_data_x_orig) ./ 255
test_data_x = Array{Float32, 2}(test_data_x)

train_data_x = reshape_image_array(train_data_x_orig) ./ 255
train_data_x = Array{Float32, 2}(train_data_x)

using Statistics, BenchmarkTools

include("../../NeuralNetwork.jl")

layer_dims = [12288, 7, 1]

# @btime neural_network_dense($train_data_x, $train_data_y_orig, $layer_dims, 100, 0.0075)

parameters, activations = neural_network_dense(train_data_x, train_data_y_orig, layer_dims, 1000, 0.0075)

predicts, accuracy = predict(train_data_x, train_data_y_orig, parameters, activations)

predicts, accuracy = predict(test_data_x, test_data_y_orig, parameters, activations)



using FileIO, Images, JLD
cd("Examples/Cactus")
# img = load("train/0a1b6731bda8f1b6a807fffc743f8d22.jpg")
# size(img)
# arr = channelview(img)
# size(arr)
#
# arr
#
# arr2 = permuteddimsview(arr, [2, 3, 1])

TRAIN_IMAGES_DIR = "train/"
TEST_IMAGES_DIR = "test/"

function images_in_dir(DIR::String)
    images = String[]
    for file in readdir(DIR)
        if endswith(file, ".jpg")
            push!(images, joinpath(DIR, file))
        end
    end
    return images
end

function load_image(path::String)
    img = load(path)
    arr = channelview(img)
    return (permuteddimsview(arr, [2, 3, 1]))
end

# arr2 = load_image("train/0a1b6731bda8f1b6a807fffc743f8d22.jpg")
# arr2
# arr3 = Array(arr2)
# arr3

train_images = images_in_dir(TRAIN_IMAGES_DIR)
test_images = images_in_dir(TEST_IMAGES_DIR)

# image_files = train_images
# train_X_orig = Array{Float32}(undef, 32, 32, 3, 0)
# train_X_orig = cat(train_X_orig, load_image(train_images[5]), dims=4)
# train_X_orig
# for image_file in image_files
#     global train_X_orig
#     train_X_orig = cat(train_X_orig, load_image(image_file), dims=4)
# end

function load_all_images(paths_vector)
    images_vector = Array{Float32}(undef, 32, 32, 3, 0)
    i = 0
    for image_file in paths_vector
        images_vector = cat(images_vector, load_image(image_file), dims=4)
        i += 1
        if i%100==0
            println(string("Processing image no. ", i))
        end
    end
    return images_vector
end

# train_X_orig = load_all_images(train_images)
# test_X_orig = load_all_images(test_images)
#
# test_X_orig
# save("test_X_orig.jld", "test_X_orig", test_X_orig)
#
# test_X = copy(test_X_orig)
#
# @benchmark load_image($test_images[8])
# @benchmark load_image($test_images[8])
# img2 = load_image(test_images[8])
# cat(test_X, img2, dims=4)
# @benchmark cat($test_X, $img2, dims=4)
# @benchmark cat($randn(100, 10, 10, 10), $randn(100, 10, 10, 10), dims=4)
# @benchmark cat(test_X, load_image(test_images[10]), dims=4)

# paths_vector = test_images
function load_images_parallel(paths_vector, num_bins)
    len = length(paths_vector) # Number of images
    bin_size = floor(UInt, len/num_bins) # Size of each bin
    all_bins = Vector{String}[] # To store image filenames in each bin
    # Create bins of images filenames
    for bin=1:num_bins-1
        paths_bin = paths_vector[(bin-1)*bin_size+1 : bin*bin_size]
        push!(all_bins, paths_bin)
    end
    # Add last bin with all extra images
    push!(all_bins, paths_vector[(num_bins-1)*bin_size+1 : len])
    bins = Vector{Array{Float32, 4}}(undef, num_bins) # To store bins of actual image arrays
    # Load each bin - multithreaded
    Threads.@threads for i = 1:num_bins
        bins[i] = load_all_images(all_bins[i])
    end
    # single array to store all images
    images_array = Array{Float32}(undef, 32, 32, 3, 0)
    # Concatenate all bins into single array
    for bin in bins
        images_array = cat(images_array, bin, dims=4)
    end
    return images_array
end
# test_X_orig_2 = load_images_parallel(test_images, 5)

# @benchmark load_images_parallel($test_images[1:100], 5)
# @benchmark load_images_parallel($test_images[1:100], 20)
# @benchmark load_all_images($test_images[1:100])

test_images = sort(test_images)
train_images = sort(train_images)
test_X_orig = load_images_parallel(test_images, 10)
train_X_orig = load_images_parallel(train_images, 100)
save("dataset.jld", "test_X_orig", test_X_orig, "train_X_orig", train_X_orig)

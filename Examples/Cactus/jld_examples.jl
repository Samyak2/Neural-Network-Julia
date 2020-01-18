using JLD, BenchmarkTools

text_X_orig = load("test_X_orig.jld", "test_X_orig")

@benchmark load("test_X_orig.jld", "test_X_orig")

d = load("dataset.jld")
d

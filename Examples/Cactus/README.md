# Aerial Cactus Identification example

[https://www.kaggle.com/c/aerial-cactus-identification](Kaggle Link)

## Part 1: Loading data

The dataset contains two zip files - `test.zip` and `train.zip`, and one CSV file `train.csv`. Unzip the two zip files to get two directories `train` and `test`.

### Installing requirements

 - Open Julia REPL and press the `]` key to get the [Pkg prompt](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html).
 - Type
    ```Julia
    add Images
    add FileIO
    add JLD
    add IJulia
    ```

    This might take a while.
 - Exit the Pkg prompt by pressing backspace or Ctrl+C.
 - Try importing the packages you installed to verify if they are installed correctly. When you import a package for the first time, it will pre-compile them which might take a minute or two.
    ```julia
    using Images
    using FileIO
    using JLD
    ```

# Usage

Check out the complete example in `Cactus.ipynb` (make sure to select the Julia 1.3 kernel in jupyter instead of the default python kernel)

module data_preprocessing
export load_and_preprocess_mnist

using MLDatasets
using Flux
using DataFrames
using Statistics


function onehotencode(df::DataFrame, col::Symbol)
    """
    Performs one-hot encoding on a categorical column in a DataFrame.

    # Arguments
    - `df`: A DataFrame containing categorical columns.
    - `col`: The column (as a Symbol) to be one-hot encoded.

    # Returns
    - A DataFrame with the categorical column replaced by one-hot encoded columns.
    """
    categories = unique(df[!, col])
    
    for category in categories
        new_col = Symbol(string(col, "_", category))  
        df[!, new_col] = (df[!, col] .== category) .* 1
    end
    
    # Drop the original categorical column
    select!(df, Not(col))
    return df
end

function onehotencode(df::DataFrame, cols::Vector{Symbol})
    """
    Applies one-hot encoding to multiple categorical columns in a DataFrame.

    # Arguments
    - `df`: A DataFrame containing categorical columns.
    - `cols`: A vector of column names (Symbols) to be one-hot encoded.

    # Returns
    - A DataFrame with specified columns one-hot encoded.
    """
    for col in cols
        df = onehotencode(df, col)
    end
    return df
end

function normalize_columns(df::DataFrame, cols::Vector{Symbol})
    """
    Normalizes specified numeric columns in a DataFrame using min-max scaling.

    # Arguments
    - `df`: A DataFrame containing numerical columns.
    - `cols`: A vector of column names (Symbols) to be normalized.

    # Returns
    - A DataFrame with normalized values in the specified columns.
    """
    for col in cols
        min_val = minimum(df[!, col])
        max_val = maximum(df[!, col])
        df[!, col] .= (df[!, col] .- min_val) ./ (max_val - min_val)  
    end
    return df
end

function load_and_preprocess_mnist()
    """
    Loads and preprocesses the MNIST dataset for training and testing.

    # Returns
    - `train_x`: Flattened training images of shape `(784, num_train_samples)`.
    - `train_y`: One-hot encoded labels for training data.
    - `test_x`: Flattened test images of shape `(784, num_test_samples)`.
    - `test_y`: One-hot encoded labels for test data.
    """
    train_x, train_y = MNIST(:train)[:]
    test_x, test_y = MNIST(:test)[:]

    train_x = Flux.flatten(train_x)
    test_x = Flux.flatten(test_x)
    
    # One-hot encode labels
    train_y = Flux.onehotbatch(train_y, 0:9)
    test_y = Flux.onehotbatch(test_y, 0:9)
    
    return train_x, train_y, test_x, test_y
end

function load_data(train_path::String, test_path::String)
    """
    Loads dataset from CSV files into DataFrames.

    # Arguments
    - `train_path`: File path to the training dataset.
    - `test_path`: File path to the test dataset.

    # Returns
    - `train_data`: A DataFrame containing the training dataset.
    - `test_data`: A DataFrame containing the test dataset.
    """
    train_data = CSV.read(train_path, DataFrame; header=true)
    test_data = CSV.read(test_path, DataFrame; header=true)

    return train_data, test_data
end

end
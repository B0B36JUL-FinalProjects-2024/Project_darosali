module data_preprocessing
export load_and_preprocess_mnist

using MLDatasets
using Flux
using DataFrames
using Statistics


function onehotencode(df::DataFrame, col::Symbol)

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
    for col in cols
        df = onehotencode(df, col)
    end
    return df
end

function normalize_columns(df::DataFrame, cols::Vector{Symbol})
    for col in cols
        min_val = minimum(df[!, col])
        max_val = maximum(df[!, col])
        df[!, col] .= (df[!, col] .- min_val) ./ (max_val - min_val)  
    end
    return df
end

function load_and_preprocess_mnist()
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
    train_data = CSV.read(train_path, DataFrame; header=true)
    test_data = CSV.read(test_path, DataFrame; header=true)

    return train_data, test_data
end

end
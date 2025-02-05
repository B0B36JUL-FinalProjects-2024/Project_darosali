module data_preprocessing
export load_data, load_and_preprocess_mnist

using MLDatasets
using Flux
using CSV
using DataFrames
using PooledArrays
using CategoricalArrays
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

function normalize_columns!(df::DataFrame, cols::Vector{Symbol})
    for col in cols
        # x_norm = (x - mean) / std
        df[!, col] = (df[!, col] .- mean(df[!, col])) ./ std(df[!, col])
    end
    return df
end

function load_and_preprocess_mnist()
    train_x, train_y = MNIST(:train)[:]
    test_x, test_y = MNIST(:test)[:]
    
    # Flatten and normalize images
    train_x = Flux.flatten(train_x) #./ 255.0f0
    test_x = Flux.flatten(test_x)#./ 255.0f0
    
    # One-hot encode labels
    train_y = Flux.onehotbatch(train_y, 0:9)
    test_y = Flux.onehotbatch(test_y, 0:9)
    
    return train_x, train_y, test_x, test_y
end

function load_data(train_path::String, test_path::String)
    train_data = CSV.read(train_path, DataFrame; header=false)
    test_data = CSV.read(test_path, DataFrame; header=false, skipto=2)

    return train_data, test_data
end

# Function to handle missing values
function handle_missing!(df::DataFrame, cols::Vector{Symbol})
    for col in names(df)
        if eltype(df[!, col]) <: AbstractString
            df[!, col] = categorical(replace(df[!, col], " ?" => missing))
        end
    end
    
    for col in cols
        df[!, col] = coalesce.(df[!, col], "Unknown")  # Fill missing with "Unknown"
    end
    return df
end

function prepare_adult()
    train_data = CSV.read("data/adult/adult.data", DataFrame; header=false)
    test_data = CSV.read("data/adult/adult.test", DataFrame; header=false)
    column_names = [
        :age, :workclass, :fnlwgt, :education, :education_num, :marital_status,
        :occupation, :relationship, :race, :sex, :capital_gain, :capital_loss,
        :hours_per_week, :native_country, :income
    ]
    rename!(train_data, column_names)
    rename!(test_data, column_names)
end

function preprocess_adult()
end

# column_names = [
#     :age, :workclass, :fnlwgt, :education, :education_num, :marital_status,
#     :occupation, :relationship, :race, :sex, :capital_gain, :capital_loss,
#     :hours_per_week, :native_country, :income
# ]
# train_data, test_data = load_data("data/adult/adult.data", "data/adult/adult.test")
# rename!(train_data, column_names)
# rename!(test_data, column_names)
# train_data = handle_missing!(train_data, [:workclass, :occupation, :native_country])
# test_data = handle_missing!(test_data, [:workclass, :occupation, :native_country])
# println(describe(train_data, :nmissing))
# categorical_features = [
#     :workclass, :education, :marital_status, :occupation, :relationship,
#     :race, :sex, :native_country
# ]
# train_data = onehotencode(train_data, categorical_features)
# test_data = onehotencode(test_data, categorical_features)
# missing_cols = setdiff(names(train_data), names(test_data))
# for col in missing_cols
#     test_data[!, col] .= 0  # Add column with zeros
# end
# numerical_cols = [
#     :age, :fnlwgt, :education_num, :capital_gain,
#     :capital_loss, :hours_per_week
# ]
# train_data = normalize_columns!(train_data, numerical_cols)
# test_data = normalize_columns!(test_data, numerical_cols)
# test_data = test_data[:, names(train_data)] 
# train_data[!, :income] = train_data[!, :income] .== " >50K"
# test_data[!, :income] = test_data[!, :income] .== " >50K."
# test_data
end
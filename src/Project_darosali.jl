using MLDatasets
using Flux
using CSV
using DataFrames
using PooledArrays
using CategoricalArrays
using MLJBase
using Statistics


function load_data(train_path::String, test_path::String)
    train_data = CSV.read(train_path, DataFrame; header=true)
    test_data = CSV.read(test_path, DataFrame; header=true)

    # rename!(train_data, column_names)
    # rename!(test_data, column_names)

    return train_data, test_data
end

train_data, test_data = load_data("data/adult/train_processed.csv", "data/adult/test_processed.csv")

target = :income
X_train = Matrix(train_data[:, Not(target)])
y_train = train_data[:, target]
X_test = Matrix(test_data[:, Not(target)])
y_test = test_data[:, target]

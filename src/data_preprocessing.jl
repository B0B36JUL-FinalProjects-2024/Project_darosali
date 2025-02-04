using MLDatasets
using CSV
using DataFrames


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
        # Normalize the column: (x - mean) / std
        df[!, col] = (df[!, col] .- mean(df[!, col])) ./ std(df[!, col])
    end
    return df
end


function preprocess_MNIST()
    train_x, train_y = MNIST(:train)[:]
    test_x, test_y = MNIST(:test)[:]

    train_x = Flux.flatten(train_x) ./ 255.0f0
    test_x = Flux.flatten(test_x) ./ 255.0f0
    println(size(test_y))
    println(size(train_x))
    train_y = Flux.onehotbatch(train_y, 0:9)
    test_y = Flux.onehotbatch(test_y, 0:9)
    return train_x, train_y, test_x, test_y
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

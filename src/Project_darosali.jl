using MLDatasets
using Flux
using CSV
using DataFrames
using PooledArrays
using CategoricalArrays
using MLJBase
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
        # Normalize the column: (x - mean) / std
        df[!, col] = (df[!, col] .- mean(df[!, col])) ./ std(df[!, col])
    end
    return df
end

train_x, train_y = MNIST(:train)[:]
test_x, test_y = MNIST(:test)[:]
train_x = Flux.flatten(train_x) ./ 255.0f0
test_x = Flux.flatten(test_x) ./ 255.0f0
println(size(test_y))
println(size(train_x))
train_y = Flux.onehotbatch(train_y, 0:9)
test_y = Flux.onehotbatch(test_y, 0:9)

train_data = CSV.read("data/adult/adult.data", DataFrame; header=false)
test_data = CSV.read("data/adult/adult.test", DataFrame; header=false, skipto=2)
column_names = [
        :age, :workclass, :fnlwgt, :education, :education_num, :marital_status,
        :occupation, :relationship, :race, :sex, :capital_gain, :capital_loss,
        :hours_per_week, :native_country, :income
    ]
rename!(train_data, column_names)
rename!(test_data, column_names)

# Convert categorical columns (String31) to String, then replace " ?" with `missing`
for col in names(test_data)
    if eltype(test_data[!, col]) <: AbstractString  # Only process string columns
        test_data[!, col] = categorical(replace(test_data[!, col], " ?" => missing))  # Convert to categorical
    end
end

for col in names(train_data)
    if eltype(train_data[!, col]) <: AbstractString  # Only process string columns
        train_data[!, col] = categorical(replace(train_data[!, col], " ?" => missing))  # Convert to categorical
    end
end

for col in [:workclass, :occupation, :native_country]
    train_data[!, col] = coalesce.(train_data[!, col], "Unknown")
    test_data[!, col] = coalesce.(test_data[!, col], "Unknown")
end
test_data
describe(test_data, :nmissing)

println(any(col -> any(==(" ?"), col), eachcol(test_data)))

categorical_features = [
    :workclass, :education, :marital_status, :occupation, :relationship,
    :race, :sex, :native_country
]

train_data = onehotencode(train_data, categorical_features)
test_data = onehotencode(test_data, categorical_features)
numerical_cols = [
    :age, :fnlwgt, :education_num, :capital_gain,
    :capital_loss, :hours_per_week
]
train_data = normalize_columns!(train_data, numerical_cols)
test_data = normalize_columns!(test_data, numerical_cols)


sum(test_data[!, :income])
train_data[!, :income] = train_data[!, :income] .== " >50K"
test_data[!, :income] = test_data[!, :income] .== " >50K."

CSV.write("data/adult/train_processed.csv", train_data)
CSV.write("data/adult/test_processed.csv", test_data)

target = :income
X_train = Matrix(train_data[:, Not(target)])
y_train = train_data[:, target]
X_test = Matrix(test_data[:, Not(target)])
y_test = test_data[:, target]
using Random
using Flux
using CSV
using DataFrames
using JLD2

include("train_NN.jl")
include("utils.jl")
include("constants.jl")
include("inferrer.jl")

# ========================
# 1️ Parse command-line arguments
# ========================
if length(ARGS) < 2
    println("Usage: julia TX_inferrer.jl <file_dir> <es_file_name>")
    exit(1)
end

scRNA_seq_file_name = ARGS[1]
result_file_name = ARGS[2]

# ========================
# 2️ Load scRNA-seq data and perform inference
# ========================
gene_exp = DataFrame(CSV.File(scRNA_seq_file_name))
estimated = DeepTX_inferrer(gene_exp)

# ========================
# 3️ Save inference results
# ========================
CSV.write(joinpath(RESULT_DIR, result_file_name), estimated)
println("✅ Inference finished! Results saved to: ", joinpath(RESULT_DIR, result_file_name))

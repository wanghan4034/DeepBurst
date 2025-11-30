using Catalyst
using Random
using BlackBoxOptim
using Flux
using CSV
using DataFrames
using JLD2
using Sobol
include("train_NN.jl")
include("utils.jl")
include("constants.jl")
 
d = 5
intensity = 1
logranges = [  1.0 15.0
                0.1 10.0
                1.0 15.0
                0.01 10.0
                0.1  400.0
                1.0 1.0
             ]

prior = Product(Uniform.(logranges[1:d, 1], logranges[1:d, 2]))
number_col=5
@load joinpath(MODELWEIGHT_DIR, "$MODEL_TYPE.jld2") model
function hellinger2(aa, bb)
    ret = 0.0
    for (a, b) in zip(aa, bb)
        ret += (sqrt(a) - sqrt(b))^2
    end
    ret
end

function loss_hellinger_map(x::AbstractVector, model, hist_yobs, tt = tt)
    bufs = zeros(Threads.nthreads())
    Threads.@threads for i = 1:length(tt)
        ps = x
        # pred = pred_pdf(model, ps, 0:length(hist_yobs))
        pred = pred_pdf_infe(model, ps, 0:length(hist_yobs))
        bufs[Threads.threadid()] += hellinger2(pred, hist_yobs)
    end
    sum(bufs)
end

function inference_parameters(gene_exp_data,model)
    op4 = names(gene_exp_data)
    print(length(op4))
    estimates = Vector{Float64}[]
    for j in eachindex(op4)
        gene_exp = gene_exp_data[:, j]
        hist_yobs = convertCountsToProb(gene_exp)
        tt = [0, 0]
        yobs = []
        Threads.@threads for i in [1]
            opt_result = bboptimize(
                p -> loss_hellinger_map(p, model, hist_yobs, tt);
                SearchRange = [tuple(logranges[i, :]...) for i = 1:d],
                TraceMode = :silent,
            )
            push!(estimates, best_candidate(opt_result))
            println(best_candidate(opt_result))
        end
    end
    estimates
end

function calculate_bs_bf(estimates)
    burst_freq = 1 ./ (estimates[:,1] ./ estimates[:,2] .+ estimates[:,3] ./ estimates[:,4])
    burst_size = estimates[:,5] .* (estimates[:,1] ./ estimates[:,2])
    mean_val = burst_freq.*burst_size
    burst_freq,burst_size,mean_val
end

function save_csv(estimates,file_path,gene_name_arr)
    estimates_df = DataFrame(estimates, :auto)
    estimates_df = DataFrame(Matrix(estimates_df)', :auto);
    estimates_df.gene_name = gene_name_arr
    CSV.write(file_path, estimates_df)
end

function matrix_to_dataframe(estimates,gene_name_arr)
    estimates_df = DataFrame(estimates, :auto)
    estimates_df = DataFrame(Matrix(estimates_df)', :auto);
    estimates_df.gene_name = gene_name_arr
    estimates_df
end

function loss(param, hist_yobs,model)
    pred = pred_pdf_infe(model, param, 0:length(hist_yobs))
    return hellinger2(pred, hist_yobs)
end

function TX_inferrer(hist_yobs,model;param=[], η=0.0001, 
    max_epoch=2, patience=5, min_delta=1e-4,sample_id=1)
    logranges = [
        1.0 15.0;
        0.1 10.0;
        1.0 15.0;
        0.01 10.0;
        0.1 400.0
    ]
    
    best_loss = Inf
    wait = 0
    temp_loss = Inf
    if length(param)==0
        prior = Product(Uniform.(logranges[1:d, 1], logranges[1:d, 2]))
        param = rand(prior)
    end
    for epoch in 1:max_epoch
        grads = gradient(p -> loss(p, hist_yobs, model), param)
        grad_values = grads[1]
        # grad_values .= map(x -> x > 0 ? clamp(x, 0.005, 10.0) : clamp(x, -10.0, -0.005), grad_values)

        new_param = param .- η .* grad_values
        for i in 1:length(param)
            new_param[i] = clamp(new_param[i], logranges[i,1], logranges[i,2])
        end
        param .= new_param
        pred = pred_pdf_infe(model, param, 0:length(hist_yobs))
        temp_loss = hellinger2(pred, hist_yobs)

        # Early stopping
        # if temp_loss + min_delta < best_loss
        #     best_loss = temp_loss
        #     wait = 0
        # else
        #     wait += 1
        #     if wait >= patience
        #         println("Sample $sample_id | Early stopping triggered at epoch $epoch with best loss $best_loss")
        #         break
        #     end
        # end
    end
    pred = pred_pdf_infe(model, param, 0:length(hist_yobs))
    temp_loss = hellinger2(pred, hist_yobs)
    return param, temp_loss
end

function TX_inferrer_fine_tune(estimates_BB,gene_exp_data)
    estimates = []
    losses = []
    for i in 1:length(estimates_BB)
        hist_yobs = convertCountsToProb(gene_exp_data[!,i])
        blackBox_param = deepcopy(estimates_BB[i])
        param, temp_loss = TX_inferrer(hist_yobs, model; param=blackBox_param)
        push!(estimates, param)
        push!(losses, temp_loss)
    end 
    estimates 
end

function DeepTX_inferrer(gene_exp; fine_tune_flag=true)
    
    estimated_BB = inference_parameters(gene_exp,model)
    # estimated = TX_inferrer_fine_tune(estimated_BB,gene_exp)
    if fine_tune_flag
        estimated = TX_inferrer_fine_tune(estimated_BB, gene_exp)
    else
        estimated = estimated_BB
    end
    estimated = matrix_to_dataframe(estimated,names(gene_exp))
    mean_true = mean.(eachcol(gene_exp))
    var_true = var.(eachcol(gene_exp))
    burst_freq,burst_size,mean_es = calculate_bs_bf(estimated)
    estimated.bf=burst_freq
    estimated.bs=burst_size
    estimated.mean_es = mean_es
    estimated.mean_true = mean_true
    estimated.var_true = var_true
    return(estimated)
end
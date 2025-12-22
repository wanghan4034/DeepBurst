# finetune dim size
echo "Job1"
for eid in E003 E116 E118
do
    for fold in 0 1 2 3
    do  
        echo "experiment $eid $fold"
        python benchmarks/DeepBurst_para_tune/train.py --config configs/default_para_tune_dim256.yaml --meta extra/datasets/processed/v1/meta_datasets/meta_data_$eid.csv  --npy-dir /GPUFS/sysu_jjzhang_3/wanghan/DeepBurst/extra/datasets/processed/v1 --fold $fold  -o benchmarks/DeepBurst_para_tune/checkpoints/$eid.$fold.bs_bf_para.tune.dim256.model.pt --exp-id 2 --binsizes 500 > logs/$eid.$fold.bs_bf_para.tune.dim256.train.log 2>&1
        sleep 120
    done
done 


# Fine-tune binsize 
echo "Job1"
for eid in E003 E116 E118
do
    for fold in 0 1 2 3
    for binsize in 100 200 500 1000 2000
    do  
        echo "experiment $eid $fold"
        python benchmarks/DeepBurst_para_tune/train.py --config configs/default_para_tune_dim256.yaml --meta extra/datasets/processed/v1/meta_datasets/meta_data_$eid.csv  --npy-dir /GPUFS/sysu_jjzhang_3/wanghan/DeepBurst/extra/datasets/processed/v1 --fold $fold  -o benchmarks/DeepBurst_para_tune/checkpoints/$eid.$fold.bs_bf_para.tune.dim256.model.pt --exp-id 2 --binsizes 500 > logs/$eid.$fold.bs_bf_para.tune.dim256.train.log 2>&1
        sleep 120
    done
done 


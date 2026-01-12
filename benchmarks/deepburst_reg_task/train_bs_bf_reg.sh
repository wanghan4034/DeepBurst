# different deley reg
echo "Job2: different deley reg"
for eid in E003 E116 E118
do
    for fold in 0 1 2 3
    do  
        for delay in 0.2 0.4 0.6 0.8 1
        do
            echo "experiment $eid $fold $delay"
            python benchmarks/burstformer_reg_task/train.py --config configs/default.yaml --meta extra/datasets/processed/v1/meta_datasets/meta_data_${eid}_delay_${delay}.csv  --npy-dir extra/datasets/processed/v1 --fold $fold  -o benchmarks/burstformer_reg_task/checkpoints/$eid.$fold.bs_bf_para.reg.delay_${delay}.model.pt --exp-id 2 --binsizes 500 --targets bs bf > logs/$eid.$fold.bs_bf_para.reg.delay_${delay}.train.log 2>&1
            sleep 20
        done
    done
done 


echo "Job3: DeepTX different deley reg"
for eid in E003 E116 E118
do
    for fold in 0 1 2 3
    do  
        for delay in 0.2 0.4 0.6 0.8 1
        do
            echo "experiment $eid $fold $delay"
            python benchmarks/burstformer_reg_task/train.py --config configs/default.yaml --meta extra/datasets/processed/v1/meta_datasets/meta_data_${eid}_deeptx_delay_${delay}.csv  --npy-dir extra/datasets/processed/v1 --fold $fold  -o benchmarks/burstformer_reg_task/checkpoints/$eid.$fold.bs_bf_para.reg.deeptx.delay_${delay}.model.pt --exp-id 2 --binsizes 500 --targets bs bf > logs/$eid.$fold.bs_bf_para.reg.deeptx.delay_${delay}.train.log 2>&1
            sleep 20
        done
    done
done 
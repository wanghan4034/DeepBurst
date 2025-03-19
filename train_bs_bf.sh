export LD_LIBRARY_PATH=/GPUFS/sysu_jjzhang_3/.conda/envs/chromoformer/lib:$LD_LIBRARY_PATH
export WANDB_MODE=offline


# cell-type specific model
echo "Job1"
for eid in E003 E116 E118
do
    for fold in 0 1 2 3
    do  
        echo "experiment $eid $fold"
        python benchmark/Promoterformer/train.py --config benchmark/Promoterformer/configs/default.yaml --meta extra/datasets/processed/v1/meta_datasets/meta_data_$eid.csv  --npy-dir extra/datasets/processed/v1 --fold $fold  -o benchmark/Promoterformer/checkpoints/$eid.$fold.No_feature_bin.bs_bf_para.model.pt --exp-id 2 --binsizes 500 > benchmark/Promoterformer/logs/$eid.$fold.No_feature_bin.bs_bf_para.train.log 2>&1
        sleep 120
    done
done 


# One Feature ablation only histone marks
echo "Job2"
for eid in E003 E116 E118
do
    for mark in "H3K4me3 H3K9me3 H3K27me3 H3K36me3 H3K27ac H3K9ac" "H3K4me1 H3K9me3 H3K27me3 H3K36me3 H3K27ac H3K9ac" "H3K4me1 H3K4me3 H3K27me3 H3K36me3 H3K27ac H3K9ac" "H3K4me1 H3K4me3 H3K9me3 H3K36me3 H3K27ac H3K9ac" "H3K4me1 H3K4me3 H3K9me3 H3K27me3 H3K27ac H3K9ac" "H3K4me1 H3K4me3 H3K9me3 H3K27me3 H3K36me3 H3K9ac" "H3K4me1 H3K4me3 H3K9me3 H3K27me3 H3K36me3 H3K27ac"
    do
        for fold in 0 1 2 3
        do  
            echo "experiment $eid $mark $fold"
            tag=`echo $mark | tr ' ' '_'`
            echo "experiment tag: $tag"
            python benchmark/Promoterformer/train.py --config benchmark/Promoterformer/configs/default.yaml --meta extra/datasets/processed/v1/meta_datasets/meta_data_$eid.csv  --npy-dir extra/datasets/processed/v1 --fold $fold  -o benchmark/Promoterformer/checkpoints/$eid.remove_$tag.$fold.No_feature_bin.bs_bf_para.model.pt --exp-id 2 --binsizes 500 --remove_marks $mark > benchmark/Promoterformer/logs/$eid.remove_$tag.$fold.No_feature_bin.bs_bf_para.train.log 2>&1
            sleep 120
        done
    done 
done 
sleep 120

# One Feature drop ablation
echo "Job3"
for eid in E003 E116 E118
do
    for mark in H3K4me1 H3K4me3 H3K9me3 H3K27me3 H3K36me3 H3K27ac H3K9ac
    do
        for fold in 0 1 2 3
        do  
            echo "experiment $eid $mark $fold"
            python benchmark/Promoterformer/train.py --config benchmark/Promoterformer/configs/default.yaml --meta extra/datasets/processed/v1/meta_datasets/meta_data_$eid.csv  --npy-dir extra/datasets/processed/v1 --fold $fold  -o benchmark/Promoterformer/checkpoints/$eid.remove_$mark.$fold.No_feature_bin.bs_bf_para.model.pt --exp-id 2 --binsizes 500 --remove_marks $mark > benchmark/Promoterformer/logs/$eid.remove_$mark.$fold.No_feature_bin.bs_bf_para.train.log 2>&1
            sleep 120
        done
    done 
done 



echo "Job4"
for fold in 0 1 2 3
do  
    echo "experiment agnostic model $fold"
    python benchmark/Promoterformer/train.py --config benchmark/Promoterformer/configs/default.yaml --meta extra/datasets/processed/v1/train.csv  --npy-dir extra/datasets/processed/v1 --fold $fold  -o benchmark/Promoterformer/checkpoints/agnostic.$fold.No_feature_bin.bs_bf_para.model.pt --exp-id 2 --binsizes 500 > benchmark/Promoterformer/logs/agnostic.$fold.No_feature_bin.bs_bf_para.train.log 2>&1
    sleep 120
done

## distance 
echo "Job5"
for eid in E003 E116 E118
do
    for distance in 1000 2000 5000 10000 20000 40000
    do
        for fold in 0 1 2 3
        do  
            echo "experiment $eid $fold $distance"
            python benchmark/Promoterformer/train.py --config benchmark/Promoterformer/configs/default.yaml --meta extra/datasets/processed/v1/meta_datasets/meta_data_$eid.csv  --npy-dir extra/datasets/processed/v1 --fold $fold  -o benchmark/Promoterformer/checkpoints/$eid.$fold.No_feature_bin.$distance.bs_bf_para.model.pt --exp-id 2 --binsizes 500 --w_prom $distance --w_max $distance > benchmark/Promoterformer/logs/$eid.$fold.No_feature_bin.$distance.bs_bf_para.train.log 2>&1
            sleep 120
        done
    done 
done 



## binsize 
echo "Job6"
for eid in E003 E116 E118
do
    for binsize in 100 200 300 400 500 1000 2000 
    do
        for fold in 0 1 2 3
        do  
            echo "experiment $eid $fold"
            python benchmark/Promoterformer/train.py --config benchmark/Promoterformer/configs/default.yaml --meta extra/datasets/processed/v1/meta_datasets/meta_data_$eid.csv  --npy-dir extra/datasets/processed/v1 --fold $fold  -o benchmark/Promoterformer/checkpoints/$eid.$fold.binsize_$binsize.No_feature_bin.cv.bs_bf_para.model.pt --exp-id 2 --targets cv_label --binsizes $binsize > benchmark/Promoterformer/logs/$eid.$fold.binsize_$binsize.No_feature_bin.cv.bs_bf_para.train.log 2>&1
            sleep 120
        done
    done 
done 


# Active & repressive model valid 
echo "Job7"
for eid in E003 E116 E118
do
    for mark in  'H3K27ac H3K9ac H3K4me1 H3K4me3' 'H3K9me3 H3K27me3 H3K36me3'
    do
        for fold in 0 1 2 3
        do  
            echo "experiment $eid $mark $fold"
            tag=`echo $mark | tr ' ' '_'`
            echo "experiment tag: $tag"
            python benchmark/Promoterformer/train.py --config benchmark/Promoterformer/configs/default.yaml --meta extra/datasets/processed/v1/meta_datasets/meta_data_$eid.csv  --npy-dir extra/datasets/processed/v1 --fold $fold  -o benchmark/Promoterformer/checkpoints/$eid.$tag.$fold.No_feature_bin.bs_bf_para.model.pt --exp-id 2 --binsizes 500 --remove_marks $mark > benchmark/Promoterformer/logs/$eid.$tag.$fold.No_feature_bin.bs_bf_para.train.log 2>&1
            sleep 120
        done
    done 
done 
exp_id = 'burstformer'
confs = ['1']

eids = [
    'E003',
    'E116',
    'E118',
]
folds = [
    '1', 
    '2', 
    '3', 
    '4',
]

# Temporary override.
eids = ['E003']

ALL = []
ALL.append(
    expand(
        'ckpts/{eid}/{exp_id}-{eid}-conf{conf}-fold{fold}.pt',
        eid=eids, exp_id=[exp_id], conf=confs, fold=folds,
    )
)

rule all:
    input: ALL

nohup python benchmark/Promoterformer/train.py  --config benchmark/Promoterformer/configs/default.yaml --meta extra/datasets/processed/v1/meta_datasets/meta_data_E116.csv  --npy-dir extra/datasets/processed/v1 --fold 0  -o benchmark/Promoterformer/checkpoints/E116.lack_H3K4me1.model.pt --exp-id 2 --binsizes 500 --add_feature_bin --remove_marks H3K4me1 > benchmark/Promoterformer/logs/E116.lack_H3K4me1.train.log 2>&1 &

rule train:
    output:
        'benchmark/Promoterformer/checkpoints/{exp_id}-{eid}-conf{conf}-fold{fold}.pt'
    shell:
        'python benchmark/Promoterformer/train.py '
        '--config benchmark/Promoterformer/configs/default.yaml '
        '--meta extra/datasets/processed/v1/meta_datasets/meta_data_{eid}.csv '
        '--npy-dir extra/datasets/processed/v1 '
        '--remove_marks H3K4me1 '
        '--exp-id {wildcards.exp_id} '
        '--fold {wildcards.fold} ' 
        '--eid {wildcards.eid} '
        '--output {output} '


use rule train as train_1 with:
    output:
        'benchmark/Promoterformer/checkpoints/{exp_id}-{eid}-conf{conf}-fold{fold,[1234]}.pt'
    params:
        gpu = 1
    resources:
        gpu1 = 1

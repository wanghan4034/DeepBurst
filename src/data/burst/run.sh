nohup python src/data/burst/extract_umi_counts.py --cell_type H1 > logs/E003.extract_umi_counts.log 2>&1 &
python src/data/burst/data_convert.py --eid E003 --gene_id2neighbors extra/datasets/processed/v1/E003/gene_id2neighbors_E003.csv -o extra/datasets/processed/v1/meta_datasets 

nohup python src/data/burst/extract_umi_counts.py --cell_type gm12878 > logs/E116.extract_umi_counts.log 2>&1 &
python src/data/burst/data_convert.py --eid E116 --gene_id2neighbors extra/datasets/processed/v1/E116/gene_id2neighbors_E116.csv -o extra/datasets/processed/v1/meta_datasets 

nohup python src/data/burst/extract_umi_counts.py --cell_type HepG2 > logs/E118.extract_umi_counts.log 2>&1 &
python src/data/burst/data_convert.py --eid E118 --gene_id2neighbors extra/datasets/processed/v1/E118/gene_id2neighbors_E118.csv -o extra/datasets/processed/v1/meta_datasets 
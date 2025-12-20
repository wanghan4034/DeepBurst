nohup python src/data_preprocessing/label_generation/txburst/txburst_infer.py --cell_type H1 > logs/E003.txburst_infer.log 2>&1 &
python src/data_preprocessing/label_generation/convert_to_label.py --eid E003 --gene_id2neighbors extra/datasets/processed/v1/E003/gene_id2neighbors_E003.csv -o extra/datasets/processed/v1/meta_datasets 

nohup python src/data_preprocessing/label_generation/txburst/txburst_infer.py --cell_type gm12878 > logs/E116.txburst_infer.log 2>&1 &
python src/data_preprocessing/label_generation/convert_to_label.py --eid E116 --gene_id2neighbors extra/datasets/processed/v1/E116/gene_id2neighbors_E116.csv -o extra/datasets/processed/v1/meta_datasets 

nohup python src/data_preprocessing/label_generation/txburst/txburst_infer.py --cell_type HepG2 > logs/E118.txburst_infer.log 2>&1 &
python src/data_preprocessing/label_generation/convert_to_label.py --eid E118 --gene_id2neighbors extra/datasets/processed/v1/E118/gene_id2neighbors_E118.csv -o extra/datasets/processed/v1/meta_datasets 
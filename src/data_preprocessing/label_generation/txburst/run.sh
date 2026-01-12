for delay in 1.0
do
    python src/data_preprocessing/label_generation/txburst/txburst_infer.py --cell_type H1 -o extra/datasets/burst/processed_v2 --delay $delay --with_cell_size > logs/E003.txburst_infer.log 2>&1
    python src/data_preprocessing/label_generation/convert_to_label.py --eid E003 --delay $delay --gene_id2neighbors extra/datasets/processed/v1/E003/gene_id2neighbors_E003.csv --processed_dir extra/datasets/burst/processed_v2 -o extra/datasets/processed/v2/meta_datasets  --with_cell_size 
done

for delay in 1.0
do
    python src/data_preprocessing/label_generation/txburst/txburst_infer.py --cell_type HepG2 -o extra/datasets/burst/processed_v2 --delay $delay   --with_cell_size > logs/E118.txburst_infer.log 2>&1
    python src/data_preprocessing/label_generation/convert_to_label.py --eid E118 --delay $delay --gene_id2neighbors extra/datasets/processed/v1/E118/gene_id2neighbors_E118.csv --processed_dir extra/datasets/burst/processed_v2 -o extra/datasets/processed/v2/meta_datasets  --with_cell_size
done

# for delay in 1.0
# do
#     python src/data_preprocessing/label_generation/txburst/txburst_infer.py --cell_type gm12878 -o extra/datasets/burst/processed_v2 --delay $delay  --with_cell_size > logs/E116.txburst_infer.log 2>&1
#     python src/data_preprocessing/label_generation/convert_to_label.py --eid E116 --delay $delay --gene_id2neighbors extra/datasets/processed/v1/E116/gene_id2neighbors_E116.csv --processed_dir extra/datasets/burst/processed_v2 -o extra/datasets/processed/v2/meta_datasets  --with_cell_size
# done


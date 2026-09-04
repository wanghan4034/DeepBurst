delay=1.0
python benchmarks/revised_v2/reviewer3/convert_to_label_replace_median.py --eid E116 --delay $delay --gene_id2neighbors extra/datasets/processed/v1/E116/gene_id2neighbors_E116.csv --processed_dir extra/datasets/burst/processed_v2 -o extra/datasets/processed/v2/meta_datasets  --with_cell_size


delay=1.0
python benchmarks/revised_v2/reviewer3/convert_to_label_replace_median.py --eid E003 --delay $delay --gene_id2neighbors extra/datasets/processed/v1/E003/gene_id2neighbors_E003.csv --processed_dir extra/datasets/burst/processed_v2 -o extra/datasets/processed/v2/meta_datasets  --with_cell_size


delay=1.0
python benchmarks/revised_v2/reviewer3/convert_to_label_replace_median.py --eid E118 --delay $delay --gene_id2neighbors extra/datasets/processed/v1/E118/gene_id2neighbors_E118.csv --processed_dir extra/datasets/burst/processed_v2 -o extra/datasets/processed/v2/meta_datasets  --with_cell_size


delay=1.0
python benchmarks/revised_v2/reviewer3/convert_to_label_remove_middle.py --eid E116 --delay $delay --gene_id2neighbors extra/datasets/processed/v1/E116/gene_id2neighbors_E116.csv --processed_dir extra/datasets/burst/processed_v2 -o extra/datasets/processed/v2/meta_datasets  --with_cell_size


delay=1.0
python benchmarks/revised_v2/reviewer3/convert_to_label_remove_middle.py --eid E003 --delay $delay --gene_id2neighbors extra/datasets/processed/v1/E003/gene_id2neighbors_E003.csv --processed_dir extra/datasets/burst/processed_v2 -o extra/datasets/processed/v2/meta_datasets  --with_cell_size


delay=1.0
python benchmarks/revised_v2/reviewer3/convert_to_label_remove_middle.py --eid E118 --delay $delay --gene_id2neighbors extra/datasets/processed/v1/E118/gene_id2neighbors_E118.csv --processed_dir extra/datasets/burst/processed_v2 -o extra/datasets/processed/v2/meta_datasets  --with_cell_size

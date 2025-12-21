# Prepare features
nohup python src/data_preprocessing/histone_marks/data_process.py --eid E003 --gene extra/datasets/genomic/hg19/genes.bed  --epi_dir extra/datasets/epigenetic/hg19 -o  extra/datasets/processed/v2 > logs/E003_data_process.log 2>&1 &
nohup python src/data_preprocessing/histone_marks/data_process.py --eid E118 --gene extra/datasets/genomic/hg19/genes.bed  --epi_dir extra/datasets/epigenetic/hg19 -o extra/datasets/processed/v2 > logs/E118_data_process.log 2>&1 &
nohup python src/data_preprocessing/histone_marks/data_process.py --eid E116 --gene extra/datasets/genomic/hg19/genes.bed  --epi_dir extra/datasets/epigenetic/hg19 -o  extra/datasets/processed/v2 > logs/E116_data_process.log 2>&1 &
# nohup python src/data_preprocessing/histone_marks/data_process.py --eid E123 --gene extra/datasets/genomic/hg19/genes.bed  --epi_dir extra/datasets/epigenetic/hg19 -o  extra/datasets/processed/v2 > logs/E123_data_process.log 2>&1 &
# nohup python src/data_preprocessing/histone_marks/data_process.py --eid E125 --gene extra/datasets/genomic/hg19/genes.bed  --epi_dir extra/datasets/epigenetic/hg19 -o  extra/datasets/processed/v2 > logs/E125_data_process.log 2>&1 &

# Prepare labels
bash src/data/burst/run.sh
import pandas as pd 

gene_id2neighbors = pd.read_csv('extra/datasets/meta_datasets/gene_id2neighbors_E003.csv',names=['gene_id','eid','chrom','start','end','strand','neighbors','scores'])
gene_id2neighbors = gene_id2neighbors[['gene_id','neighbors','scores']]


meta_dataset = pd.read_csv('extra/datasets/meta_datasets/meta_data_E003.csv')
meta_dataset = meta_dataset.drop(['neighbors','scores'],axis=1)

data = pd.merge(meta_dataset,gene_id2neighbors,how='inner',on='gene_id')

data.to_csv('extra/datasets/meta_datasets/meta_data_E003.csv',index=False)
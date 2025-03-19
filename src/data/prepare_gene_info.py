import pandas as pd 

refseq_annotations = pd.read_csv('extra/datasets/annotations/hg19/refseq_annotation_hg19.tsv',sep='\t')
refseq_annotations['ref_seq_id'] = refseq_annotations.apply(lambda row:row['name'].split('.')[0],axis=1)
refseq_annotations['transcript_start_site'] = refseq_annotations.apply(lambda row:row['txStart'] if row['strand']=='+' else row['txEnd'],axis=1)

refseq_annotations_1 = refseq_annotations[['ref_seq_id','chrom','transcript_start_site','strand']]

ncbi_annotations = pd.read_csv('extra/datasets/annotations/hg19/ncbi_annotations_hg19.txt',sep='\t')
ncbi_annotations_1 = ncbi_annotations[ncbi_annotations['RefSeq mRNA ID'].str.startswith('NM_')].drop(['Gene stable ID version'],axis=1)
ncbi_annotations_1.columns = ['gene_id','ncbi_chrom','ncbi_tss','ncbi_strand','gc_content','cds_length','ref_seq_id']

data = pd.merge(ncbi_annotations_1,refseq_annotations_1,how='inner',on='ref_seq_id')

data = data[['ncbi_tss','transcript_start_site']]
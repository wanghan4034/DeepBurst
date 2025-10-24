import os
import re
import uuid
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import gtfparse
from typing import List, Dict,Union, Iterator
from src.data.features.constants import GENOMIC_FASTA_PATH
from src.data.features.basic import GenomicSite,  GenomicSlice, CREsInfo, GeneInfo, TranscriptInfo, TranscriptStartSite, FragData
from src.utils.logs import get_logger
logger = get_logger()

TRANSCRIPT = 'transcript'
EXON = 'exon'

def get_genomic_slice(chrom:str, start:int, end:int, score:int = 0,strand:str = '+')->GenomicSlice:
    start = GenomicSite(chrom=chrom,location=start, strand=strand)
    end = GenomicSite(chrom=chrom,location=end, strand=strand)
    genomic_slice = GenomicSlice(start=start, end=end, score=score)
    return genomic_slice


def get_gene_infos(genes_bed_path:str, frag_data: FragData = None)->Dict[str, GeneInfo]:
    logger.info("Start loading gene infos")

    genes_data = pd.read_csv(genes_bed_path,sep='\t',names=['chrom','start','end','gene_id','score','strand'])

    gene_infos = {}

    for record in tqdm(genes_data.to_records()):
        gene_id=record['gene_id']
        gemomic_base = GenomicSite(chrom=record['chrom'],location=record['start'],strand=record['strand'])
        transcript_start_site = TranscriptStartSite(gemomic_base)
        if frag_data:
            frag_sample = frag_data.search_frag_by_site(gemomic_base)
            transcript_start_site.add_frag(frag_sample)
        gene_info = GeneInfo(gene_id,transcript_start_site=transcript_start_site)
        gene_infos[gene_id] = gene_info

    return gene_infos


def parse_annotation(annotation_path)->Dict[str, GeneInfo]:
    # use the positions of exon to estimate the tss positions
    annotations = gtfparse.read_gtf(annotation_path).to_pandas()
    gene_infos = {}
    def _parse_line(record):
        chrom = record['seqname']
        start = GenomicSite(chrom=chrom, location=int(record['start']),strand=record['strand'])
        end = GenomicSite(chrom=chrom, location=int(record['end']),strand=record['strand'])
        gene_id = record['gene_id']
        transcript_id = record['transcript_id']
        score = record['score']
        genomic_slice = GenomicSlice(start=start, end=end, score=score)
        info = None
        if record['feature'] == TRANSCRIPT:
            info = TranscriptInfo(gene_id, transcript_id, genomic_slice)

        return info

    for record in tqdm(annotations.to_records()):
        if record['feature'] == TRANSCRIPT:
            gene_id = record['gene_id']
            info = _parse_line(record)
            if gene_id not in gene_infos:
                gene_infos[gene_id] = GeneInfo(gene_id,[info])
            else:
                gene_infos[gene_id].add_transcript(info)

    return gene_infos


def get_cres_infos(cres_path:str ='./extra/datasets/CREs/GRCh19-cCREs.bed')-> Dict[str,CREsInfo]:
    cres_infos = {}
    data = pd.read_csv(cres_path,sep='\t',header=None)[[0,1,2,3,4]].dropna(how='any')
    columns = ['chrom','start','end','cre_type','cre_id']
    data.columns = columns
    logger.info("Start loading cres infos")
    for row in tqdm(data.to_records()):
        start = GenomicSite(chrom=row['chrom'], location=row['start'])
        end = GenomicSite(chrom=row['chrom'], location=row['end'])
        genomic_slice = GenomicSlice(start,end)
        cres_info = CREsInfo(row['cre_id'],row['cre_type'],genomic_slice)
        if row['cre_id'] not in cres_infos:
            cres_infos[row['cre_id']] = cres_info

    return cres_infos


def region_search(regions: np.array ,location:int): 
    """
    regions: CREs start, end positions
    location: tss location
    """
    def _min_distance(region, location):
        return min(abs(region-location))

    distances = np.array([_min_distance(region,location) for region in regions])
    return distances


def get_dna_sequences(genomic_slices: Union[GenomicSlice,List[GenomicSlice]], genomic_fa = GENOMIC_FASTA_PATH)->List[str]:
    def _check_line(flag,target):
        if flag == target:
            raise ValueError('File is corrupted')
        return target
    
    if isinstance(genomic_slices,GenomicSlice):
        genomic_slices = [genomic_slices]

    slice_ids = range(len(genomic_slices))

    results = {}
    random_uuid = uuid.uuid4()
    temp_bed = f'./.{random_uuid}.bed'
    with open(temp_bed, 'w') as w:
        for gene_id, region in zip(slice_ids,genomic_slices):
            w.write(f'{region.chrom}\t{region.start.location}\t{region.end.location}\t{gene_id}\t{region.score}\t{region.strand}\n')

    with os.popen(f'bedtools getfasta -s -name -fi {genomic_fa} -bed {temp_bed}') as r:
        flag = 1
        for line in r.readlines():
            if line.startswith('>'):
                flag = _check_line(flag, -1)
                gene_id = re.findall(r'>(.*?)::chr',line.strip())[0]
            else:
                flag = _check_line(flag, 1)
                if not results.get(gene_id):
                    results[gene_id] = line.strip().upper()

    os.popen(f'rm {temp_bed}')
    return list(results.values())



class GenomicInfo:
    def __init__(self, annotation_path, frag_data: 'FragData' = None):
        self.gene_infos = get_gene_infos(annotation_path, frag_data)
        self.frag_data = frag_data

    def get_gene_info(self,gene_id:str):
        return self.gene_infos[gene_id]
    
    def add_cres_infos(self,cres_info_path):
        self.cres_infos = get_cres_infos(cres_info_path)


if __name__ == '__main__':
    genes_bed_path = 'extra/datasets/genomic/genes.bed'
    frag_data_path = 'extra/datasets/pcHi-C/LI11.txt'
    frag_data = FragData(frag_data_path)
    genomic_info = GenomicInfo(genes_bed_path,frag_data)
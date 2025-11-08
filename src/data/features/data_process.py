
import os
import numpy as np
import pickle
import math
import argparse
from tqdm import tqdm
from copy import deepcopy
from typing import Dict
from src.data.features.basic import  FragData
from src.data.features.genomics import GenomicInfo, get_dna_sequences
from src.data.features.epigenetic import BigWigInfo
from src.data.features.genomic_utils import genomic_site_region_scale
from src.utils.logs import get_logger
from src.data.features.constants import GENOMIC_FASTA_PATH,  SCALED_WINDOW_SIZE , BASE_MAP, MARKS
logger = get_logger()

BIN_SIZE = 500
SCORE_THRESHOLD = 1.5
MAX_BINS = 2 * SCALED_WINDOW_SIZE / BIN_SIZE

parser = argparse.ArgumentParser()
parser.add_argument('--gene', required=True, help='Gene bed file with transcript start site.')
parser.add_argument('--eid', required=True, help='Cell Type EID.')
parser.add_argument('--hic', required=False, help='Hic Fragment data, which should be txt format.')
parser.add_argument('--sequence', required=False, help='add DNA sequence, which should be txt format.')
parser.add_argument('--epi_dir', required=True, help='Epigenetic data dir.')
parser.add_argument('--max_neighbors', type=int, default=8, help='Input tissue.')
parser.add_argument('-o','--output', required=True, help='Input tissue.')

args = parser.parse_args()

eid = args.eid
genes_bed_path = args.gene
frag_data_path = args.hic
epigenetic_dir = args.epi_dir
saved_dir = args.output
max_neighbors = args.max_neighbors


saved_dir = os.path.join(saved_dir,eid)
npy_saved_dir = os.path.join(saved_dir,f'regions')
meta_data_path = os.path.join(saved_dir,f'gene_id2neighbors_{eid}.csv')


logger.warning(f"eid: {eid}")
logger.warning(f"genes_bed_path: {genes_bed_path}")
logger.warning(f"hic_data_path: {frag_data_path}")
logger.warning(f"epigenetic_dir: {epigenetic_dir}")
logger.warning(f"saved_dir: {saved_dir}")
logger.warning(f"npy_saved_dir: {npy_saved_dir}")
logger.warning(f"meta_data_path: {meta_data_path}")

if not os.path.exists(saved_dir):
    os.makedirs(saved_dir, exist_ok=True)

if not os.path.exists(npy_saved_dir):
    os.makedirs(npy_saved_dir, exist_ok=True)


if frag_data_path:
    frag_data = FragData(frag_data_path)
else:
    frag_data = None


genomic_info = GenomicInfo(genes_bed_path,frag_data)

with open(os.path.join(saved_dir,f'{eid}_genomic_info.pkl'),'wb') as w:
    pickle.dump(genomic_info,w)
    logger.info(f"The genomic_info object has been dumpped in {eid}_genomic_info.pkl")

signals: Dict[str, BigWigInfo] = {}
for mark in MARKS:
    bigwig_file_path = os.path.join(epigenetic_dir,f'{eid}-{mark}.bw')
    signals[mark] = BigWigInfo(bigwig_file_path)

logger.info('Loaded marks info')

def parse_dna_sequence(s: str):
    data = np.zeros(shape=(len(BASE_MAP),len(s)), dtype=np.float32)
    for idx, base in enumerate(s):
        index = BASE_MAP.get(base)
        if index != None:
            data[index,idx] = 1
    return data


region_ids = []
with open(meta_data_path,'w+') as w:
    for gene_id, gene_info in tqdm(genomic_info.gene_infos.items()):
        line = []
        regions = {}
        strand = gene_info.strand
        chrom = gene_info.chrom
        start = gene_info.location
        end = gene_info.location + 1

        transcript_start_site = gene_info.transcript_start_site
        genomic_site = transcript_start_site.genomic_site
        genomic_slice = genomic_site_region_scale(genomic_site)
        line += [gene_id,eid,chrom,str(start),str(end),strand]

        if gene_id  not in regions:
            regions[gene_id] = genomic_slice

        neighbor_list = []
        score_list = []
        for neighbor in transcript_start_site.topk_interactive_fragsamples(max_neighbors):
            neighbor_id = neighbor.neighbor_id
            score = neighbor.score
            if score < SCORE_THRESHOLD:
                continue

            neighbor_list.append(neighbor_id)
            score_list.append(str(score))
            genomic_slice = deepcopy(neighbor.neighbor.genomic_slice)  # Important , must use DEEPCOPY !
            genomic_slice.strand = strand
            if neighbor_id  not in regions:
                regions[neighbor_id] = genomic_slice

        if neighbor_list:
            line += [';'.join(neighbor_list),';'.join(score_list)]

        line = ','.join(line)
        w.writelines(line+'\n')


        sample = {}

        for region_id, item in regions.items():
            if region_id not in region_ids:
                region_ids.append(region_id)
            else:
                continue

            marks_data = []
            for mark in MARKS:
                marks_data += signals[mark].get_values_by_location(item)
            
            marks_data = np.array(marks_data)
            if args.sequence:
                sequences = get_dna_sequences([item],GENOMIC_FASTA_PATH)
                sequence_data = parse_dna_sequence(sequences[0])
                data = np.concatenate((marks_data,sequence_data), axis=0)
            else:
                data = marks_data
            out = os.path.join(npy_saved_dir, f'{region_id}.npy')
            np.save(out, data)

with open(os.path.join(saved_dir, f'{eid}_region_ids.pkl'),'wb') as f:
    pickle.dump(region_ids,f)
    logger.info(f"Saved region_ids, total regions {len(region_ids)} ")
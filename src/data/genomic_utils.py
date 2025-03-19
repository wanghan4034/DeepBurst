from src.data.basic import GenomicSite, GenomicSlice, FragSample
from src.data.constants import FASTA_SIZE, SCALED_WINDOW_SIZE , GENOMIC_SLICE , START_PADDING_SIZE, END_PADDING_SIZE
from typing import Dict, Union



def genomic_site_region_scale(genomic_site: 'GenomicSite') -> GenomicSlice:
    start = genomic_site - SCALED_WINDOW_SIZE
    start_padding_size = abs(start.location) if start.location < 0 else 0
    if start_padding_size:
        start.location = 0
    
    end = genomic_site + SCALED_WINDOW_SIZE
    end_padding_size = end.location - FASTA_SIZE[end.chrom] if FASTA_SIZE[end.chrom] < end.location else 0
    if end_padding_size:
        end.location =  FASTA_SIZE[end.chrom]

    return GenomicSlice(start = start, end = end)


def fragsample_region_scale(fragsampe: 'FragSample') -> GenomicSlice:
    middle_site =  (fragsampe.start + fragsampe.end) // 2    
    return genomic_site_region_scale(middle_site)
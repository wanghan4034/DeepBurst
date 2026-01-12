import numpy as np
import pandas as pd
from typing import List, Dict, Union, Any
from tqdm import tqdm 
from src.data_preprocessing.histone_marks.utils import gemonic_slice_split, generate_frags_linking_key, genomic_slice_concat



class GenomicSite:
    def __init__(self, chrom:str = None, location: int = None, strand: str = None, value: Any = 0):
        self._chrom = chrom
        self._location = location
        self._strand = strand
        self._value = value

    @property
    def chrom(self):
        return self._chrom
    
    @chrom.setter
    def chrom(self, x):
        self._chrom = x

    @property
    def location(self):
        return self._location
    
    @location.setter
    def location(self, x):
        self._location = x

    @property
    def strand(self):
        return self._strand
    
    @strand.setter
    def strand(self, x):
        self._strand = x

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, x):
        self._value = x

    def __repr__(self):
        return f"{self.chrom}\t{self.location}\t{self.value}\t{self.strand}"
    
    def __str__(self) -> str:
        return f"{self.chrom}\t{self.location}\t{self.strand}"
    
    def __add__(self, other: Union[int, 'GenomicSite']):
        if isinstance(other, int):
            return GenomicSite(self.chrom, self.location + other, self.strand, self.value)
        
        return GenomicSite(self.chrom, self.location + other.location, self.strand, self.value)

    def __sub__(self, other: Union[int, 'GenomicSite']):
        if isinstance(other, int):
            return GenomicSite(self.chrom, self.location - other, self.strand, self.value)

        return GenomicSite(self.chrom, self.location - other.location, self.strand, self.value)

    def __floordiv__(self, other: int):
        
        return GenomicSite(self.chrom, self.location // other, self.strand, self.value)
    
    def __le__(self, other: Union[int, 'GenomicSite']):
        if isinstance(other, int):
            return self.location <= other

        return self.location <= other.location

    def __ge__(self, other: Union[int, 'GenomicSite']):
        if isinstance(other, int):
            return self.location >= other

        return self.location >= other.location

    def __gt__(self, other: Union[int, 'GenomicSite']):
        if isinstance(other, int):
            return self.location > other

        return self.location > other.location


    def __lt__(self, other: Union[int, 'GenomicSite']):
        if isinstance(other, int):
            return self.location < other

        return self.location < other.location


class TranscriptStartSite(GenomicSite):
    def __init__(self, genomic_site: 'GenomicSite', frag_sample: 'FragSample' = None):
        self._genomic_site = genomic_site
        self._frag_sample = frag_sample
    
    @property
    def frag_sample(self):
        return self._frag_sample

    @property
    def genomic_site(self):
        return self._genomic_site

    @property
    def chrom(self):
        return self._genomic_site.chrom

    @property
    def location(self):
        return self._genomic_site.location

    @property
    def strand(self):
        return self._genomic_site.strand
    
    @property
    def interactive_fragsamples(self)->Dict[str, 'Neighbor']:
        return self._frag_sample.neighbors if self._frag_sample else {}

    @property
    def value(self):
        return self._genomic_site.value

    def add_frag(self, frag_sample: 'FragSample'):
        self._frag_sample = frag_sample

    def add_site(self, genomic_site: 'GenomicSite'):
        self._genomic_site = genomic_site

    def topk_interactive_fragsamples(self, topk)->List['Neighbor']:
        return self._frag_sample.get_topk_neighbors(topk) if self._frag_sample else []

    def __repr__(self):
        return f"{self.chrom}\t{self.location}\t{self.value}\t{self.strand}"
    
    def __str__(self) -> str:
        return f"{self.chrom}\t{self.location}\t{self.strand}"


class GenomicSlice:
    def __init__(self, start: GenomicSite = None, end: GenomicSite = None, score:int = 0, sequence:str=None):
        self._start = start
        self._end = end
        self._score = score
        self._sequence = sequence
    
    def __repr__(self) -> str:
        return f"{self.chrom}\t{self.start.location}\t{self.end.location}\t{self.strand}"
    
    @property
    def genomic_slice_id(self) -> str:
        return genomic_slice_concat(self.chrom, self.start.location,self.end.location)
    
    @property
    def chrom(self):
        return self._start.chrom

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def score(self):
        return self._score

    @property
    def strand(self):
        return self._start.strand
    
    @strand.setter
    def strand(self, x:str):
        self._start.strand = x 

    @property
    def sequence(self):
        return self._sequence
    
    @sequence.setter
    def sequence(self,sequence):
        if isinstance(sequence, str):
            self._sequence = sequence
        else:
            raise ValueError
    
    @sequence.deleter
    def sequence(self):
        self._sequence = None

    def is_overlap(self, other: Union[str,'GenomicSlice'],valid_strand:bool = False):
        if isinstance(other, str):
            chrom, start, end = gemonic_slice_split(other)
            start = GenomicSite(chrom, start)
            end = GenomicSite(chrom, end)
            strand = None
        else:
            chrom, start, end, strand = other.chrom, other.start, other.end, other.strand
        if valid_strand and self.strand != strand:
            return False
        if self.chrom != chrom:
            return False
        if self.end >=  end:
            if self.start > end:
                return False
        if self.end < end:
            if self.end < start:
                return False
        return True



class FragSample:
    def __init__(self, frag_id: str, genomic_slice:'GenomicSlice'):
        self.frag_id = frag_id
        self.genomic_slice = genomic_slice
        self.neighbors = {}

    def is_contain(self,genomic_site: 'GenomicSite'):
        flag = (genomic_site.chrom == self.chrom) and (genomic_site.location > self.start.location and genomic_site.location < self.end.location)
        return flag

    def add_neighbor(self, neighbor: 'Neighbor'):
        self.neighbors[neighbor.neighbor_id] = neighbor

    def get_topk_neighbors(self,topk)->List['Neighbor']:
        return sorted(self.neighbors.values(),key=lambda neighbor:neighbor.score, reverse=True)[:topk]

    @property
    def chrom(self)->str:
        return self.genomic_slice.chrom
    
    @property
    def start(self)->GenomicSite:
        return self.genomic_slice.start

    @property
    def end(self)->GenomicSite:
        return self.genomic_slice.end
    
    @property
    def strand(self)->str:
        return self.genomic_slice.strand

    @strand.setter
    def strand(self,x: str):
        self.strand = x

    def __repr__(self) -> str:
        return f"{self.frag_id}\tneigbor_nums:{len(self.neighbors)}"

class Neighbor:
    """A class contains the information of a pair of fragments in pcHi-C data
    """
    def __init__(self, master_fragsample: 'FragSample', neighbor_fragsample: 'FragSample', score: 'float') -> None:
        self._master = master_fragsample
        self._neighbor = neighbor_fragsample 
        self.score = score
    
    @property
    def master_id(self):
        return self._master.frag_id
    
    @property
    def neighbor_id(self):
        return self._neighbor.frag_id
    
    @property
    def master(self):
        return self._master

    @property
    def neighbor(self):
        return self._neighbor
     
    @property
    def frags_linking_id(self):
        return f"{generate_frags_linking_key(self.master_id, self.neighbor_id)}\t{self.score}"
    
    def __repr__(self) -> str:
        return f"{self.neighbor_id}\t{self.score}"


class FragData:
    def __init__(self, hic_data_path):
        self.frag_samples = self._serialize(hic_data_path)

    def _serialize(self,hic_data_path:str)->Dict[str, FragSample]:
        raw_data = pd.read_csv(hic_data_path,sep='\t')

        def _parse_frag(frag_id):
            chrom, start, end = gemonic_slice_split(frag_id)
            start = GenomicSite(chrom, start)
            end = GenomicSite(chrom, end)
            frag_sample = FragSample(frag_id, GenomicSlice(start, end))
            return frag_sample

        frag_samples:Dict[str, 'FragSample'] = {}

        for record in tqdm(raw_data.to_records(), total=len(raw_data)):
            frag_sample = _parse_frag(record['frag1'])
            neighbor_frag_sample = _parse_frag(record['frag2'])
            neighbor_sample = Neighbor(frag_sample,neighbor_frag_sample,score=record['dist_res'])
            if frag_sample.frag_id not in frag_samples:
                frag_sample.add_neighbor(neighbor_sample)
                frag_samples[frag_sample.frag_id] = frag_sample
            else:
                frag_samples[frag_sample.frag_id].add_neighbor(neighbor_sample)

            neighbor_sample_reversed = Neighbor(neighbor_frag_sample,frag_sample,score=record['dist_res'])
            if neighbor_frag_sample.frag_id not in frag_samples:
                neighbor_frag_sample.add_neighbor(neighbor_sample_reversed)
                frag_samples[neighbor_frag_sample.frag_id] = neighbor_frag_sample
            else:
                frag_samples[neighbor_frag_sample.frag_id].add_neighbor(neighbor_sample_reversed)

        return frag_samples

    def search_frag_by_site(self, site: Union[str,GenomicSite]) -> Union[None|FragSample]:
        if isinstance(site, str):
            chrom, location = site.split(':')
            site = GenomicSite(chrom, int(location))
        
        frag_samples = [frag_sample for frag_sample in self.frag_samples.values() if frag_sample.is_contain(site)]

        if len(frag_samples) == 0:
            return None
        
        frag_sample = frag_samples[0]

        return frag_sample

    
    def search_neighbors_by_site(self, site: Union[str,GenomicSite]):

        frag_sample = self.search_frag_by_site(site)
        return frag_sample.neighbors if frag_sample else frag_sample
    
    def __repr__(self) -> str:
        return f"frag_samples:{len(self.frag_samples)}"

class UTR5Info:
    def __init__(self) -> None:
        pass

class UTR3Info:
    def __init__(self) -> None:
        pass


class CDSInfo:
    def __init__(self,gene_id:str,transcript_id:str, genomic_slice:GenomicSlice):
        self.gene_id = gene_id
        self.transcript_id = transcript_id
        self.genomic_slice = genomic_slice


class ExonInfo:
    def __init__(self,gene_id:str,exon_id:str, exon_number:str,genomic_slice:GenomicSlice):
        self.gene_id = gene_id
        self.exon_id = exon_id
        self.exon_number = exon_number
        self.genomic_slice = genomic_slice
        

class TranscriptInfo:
    def __init__(self,gene_id:str,transcript_id:str, genomic_slice:GenomicSlice):
        self.gene_id = gene_id
        self.transcript_id = transcript_id
        self.genomic_slice = genomic_slice
    
    @property
    def strand(self):
        return self.genomic_slice.strand


class CREsInfo:
    def __init__(self, cres_id:str, cres_type:str, region:GenomicSlice) -> None:
        self.id = cres_id
        self.type = cres_type
        self.region = region

    def __repr__(self):
        return f"{self.id} {self.type}"

    def distance_from_target_location(self,genomic_node:GenomicSite):
        if genomic_node.chrom != self.region.chrom:
            return np.inf

        target_location = genomic_node.location
        start = self.region.start.location
        end = self.region.end.location
        region = np.array([start, end])
        return min(abs(region-target_location))
    
    def is_overlap(self,other: Union[str,'GenomicSlice'])->bool:
        return self.region.is_overlap(other)


class GeneInfo:
    def __init__(self,gene_id: str, transcript_start_site: TranscriptStartSite = None, transcript_infos: List[TranscriptInfo] = None, gene_name:str = None):
        self.gene_id = gene_id
        self.transcript_infos = transcript_infos
        self.gene_name = gene_name
        self._transcript_start_site = transcript_start_site

    def __repr__(self):
        return f"{self.gene_id}\t{self.chrom}\t{self.location}\t{self.strand}"
    
    @property
    def transcript_start_site(self) -> TranscriptStartSite:
        if not self._transcript_start_site:
            genomic_slices = [transcript_info.genomic_slice for transcript_info in self.transcript_infos]
            if self.strand == '+':
                idx = np.argmin([genomic_slice.start.location for genomic_slice in genomic_slices])
                return genomic_slices[idx].start

            idx = np.argmax([genomic_slice.end.location for genomic_slice in genomic_slices])

            self._transcript_start_site =  genomic_slices[idx].end

        return self._transcript_start_site

    @property
    def chrom(self):
        return self._transcript_start_site.chrom

    @property
    def location(self):
        return self._transcript_start_site.location

    @property
    def strand(self):
        return self._transcript_start_site.strand

    @property
    def interactive_fragsamples(self):
        if not self._transcript_start_site:
            raise ValueError('No transcript_start_site ')

        return self._transcript_start_site.interactive_fragsamples
    
    @property
    def region(self):
        raise NotImplementedError

    def add_transcript(self, transcript_info: TranscriptInfo):
        self.transcript_infos.append(transcript_info)
    
    def add_transcript_start_site(self, site: TranscriptStartSite):
        self._transcript_start_site = site

    def get_expression(self,):
        raise NotImplementedError
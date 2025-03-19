import numpy as np 
import math
MARKS = ['H3K4me1', 'H3K4me3', 'H3K9me3', 'H3K27me3', 'H3K36me3', 'H3K27ac', 'H3K9ac']
def get_base_distribution(sequence:str):
    """compute base distribution from  a DNA sequence

    Args:
        sequence (_type_): DNA sequence

    Returns:
        List: distribution
    """
    bases = ['A','T','C','G']
    frequency = {base:0 for base in bases}
    for base in sequence:
        if base in bases:
            frequency[base] += 1

    result = [frequency[base]/len(sequence) for base in bases]
    return result

def get_bin_signal(x: np.ndarray, bin_size: int):
    n_bins = np.ceil(len(x) / bin_size)
    x_bins = []
    for i in range(n_bins):
        x = np.mean(x[i*bin_size:(i+1)*bin_size])
        x_bins.append(x_bins)
    return x_bins


def gemonic_slice_split(s: str):
    chrom = s.split(':')[0]
    start = s.split(':')[1].split('-')[0]
    end = s.split(':')[1].split('-')[1]
    return chrom, int(start), int(end)

def generate_frags_linking_key(id1, id2):
    return f"{id1}->{id2}"

def genomic_slice_concat(chrom: str, start:int, end:int):
    return f"{chrom}:{start}-{end}"




def bin_and_padding(x:np.ndarray, bin_size=500, max_n_bins = 80, strand = '+'):
    """Given a 2D tensor x, make binned tensor by
    taking average values of `bin_size` consecutive values.
    Appropriately pad by
    left_pad = ceil((max_n_bins - n_bins) / 2)
    right_pad = floor((max_n_bins - n_bins) / 2)
    """
    # Binning.
    n_bins = math.ceil(x.shape[-1] / bin_size)
    if strand == '-':
        x = np.fliplr(x)

    x_binned = []
    for i in range(n_bins):
        marks = np.mean(x[:len(MARKS), i * bin_size : (i + 1) * bin_size], keepdims=True, axis= -1)
        seq = np.sum(x[len(MARKS):, i * bin_size : (i + 1) * bin_size], keepdims=True, axis= -1)
        b = np.concatenate((marks,seq), axis=0)
        b = np.log1p(b)
        x_binned.append(b)
    x_binned = np.concatenate(x_binned, axis=1)

    # Padding.
    left_pad = math.ceil((max_n_bins - n_bins) / 2)
    right_pad = math.floor((max_n_bins - n_bins) / 2)

    x_binned = np.concatenate(
        [
            np.zeros([x.shape[0], left_pad]),
            x_binned,
            np.zeros([x.shape[0], right_pad]),
        ],
        axis=1,
    )

    return x_binned, left_pad, n_bins, right_pad


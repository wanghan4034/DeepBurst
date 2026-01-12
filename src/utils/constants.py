import torch 

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MARKS = ["H3K4me1","H3K4me3","H3K9me3","H3K27me3","H3K36me3","H3K27ac","H3K9ac"]
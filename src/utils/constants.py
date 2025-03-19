import torch 

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

EMBEDDING_TRANSFORMER = "embedding_transformer"
REGULATION_TRANSFORMER = "regulation_transformer"

ATTENTION_MODULE = {
    0: EMBEDDING_TRANSFORMER,
    1: REGULATION_TRANSFORMER,
}

BURST_SIZE = 'bs_label'
BURST_FREQUENCY = 'bf_label'

TARGETS = [BURST_SIZE, BURST_FREQUENCY]
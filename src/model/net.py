import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.constants import DEVICE
from src.model.modules import Transformer

class EmbeddingTransformer(nn.Module):
    def __init__(
        self, n_feats, n_layers, n_heads, d_model, d_ff, pos_enc=True, activation=F.relu
    ):
        super(EmbeddingTransformer, self).__init__()

        self.d_model = d_model
        self.lin_proj = nn.Linear(n_feats, self.d_model, bias=False)

        self.transformer = Transformer(
            n_layers, n_heads, self.d_model, self.d_model, d_ff, activation, gate=False
        )
        self.pos_enc = pos_enc

    def _pos_enc(self, dim, max_len):
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, 1).unsqueeze(1)
        k = torch.exp(-np.log(10000) * torch.arange(0, dim, 2) / dim)
        pe[:, 0::2] = torch.sin(pos * k)
        pe[:, 1::2] = torch.cos(pos * k)
        return pe

    def forward(self, x, mask = None):
        """
        x : bsz x (max_n_interactions + 1) x max_n_bins x n_mark
        mask : bsz x (max_n_interactions + 1) x 1 x max_n_bins x max_n_bins
        """
        bsz, max_n_bins, n_mark = x.size(0), x.size(2), x.size(3)
        if mask != None:
            mask = mask.view(-1, 1, max_n_bins, max_n_bins)
        # --> mask : (bsz x (max_n_interactions + 1)) x 1 x max_n_bins x max_n_bins

        x = x.view(-1, max_n_bins, n_mark)
        # --> x: (bsz x (max_n_interactions + 1)) x max_n_bins x n_maxk
        x = self.lin_proj(x)
        # --> x: (bsz x (max_n_interactions + 1)) x max_n_bins x d_model

        # Embed and linearly project each genomic region independently.
        # Add positional embedding.
        if self.pos_enc:
            x = (
                x
                + self._pos_enc(dim=self.d_model, max_len=max_n_bins)[:, : self.d_model]
                .unsqueeze(0)
                .to(DEVICE)
            )
        # --> x : (bsz x (max_n_interactions + 1)) x max_n_bins x d_model
        x = self.transformer(x, mask)
        # --> x : (bsz x (max_n_interactions + 1)) x max_n_bins x d_model
        x = x.view(bsz, -1, max_n_bins, self.d_model)

        return x, x[:, :, max_n_bins // 2]



class ChromoformerClassifier(nn.Module):
    def __init__(
        self,
        n_feats_p=7,
        d_emb=128,
        d_head=128,
        embed_kws={
            "n_layers": 1,
            "n_heads": 2,
            "d_model": 128,
            "d_ff": 128,
        },
        binsizes=[2000, 500, 100],
        seed=42,
        targets = None,
    ):
        super(ChromoformerClassifier, self).__init__()
        torch.manual_seed(seed)

        # Update arguments for each transformer layer.
        embed_kws["n_feats"] = n_feats_p
        embed_kws["d_model"] = d_emb

        self.binsizes = binsizes
        self.embed = nn.ModuleDict(
            {str(binsize): EmbeddingTransformer(**embed_kws) for binsize in binsizes}
        )

        self.fc_head = nn.Sequential(
            nn.Linear(d_emb * len(binsizes), d_head),
            nn.ReLU(),
            nn.Linear(d_head, len(targets) * 2),
        )

    def forward(
        self,
        promoter_feats,
        promoter_pad_masks = None,
    ):

        promoter_embeddings_full, promoter_embeddings_tss = {}, {}
        for binsize in self.binsizes:
            layer = self.embed[str(binsize)]
            p_emb_full, p_emb = layer(
                promoter_feats, promoter_pad_masks
            )
            promoter_embeddings_full[binsize] = p_emb_full
            promoter_embeddings_tss[binsize] = p_emb


        x_in = {
            binsize: promoter_embeddings_tss[binsize]
            for binsize in self.binsizes
        }

        x = torch.cat([x_in[binsize][:, 0] for binsize in self.binsizes], axis=1)
        # burst_size, burst_frequency = torch.chunk(self.fc_head(x),len(targets),dim=-1)

        return  self.fc_head(x)

    def generate_transformer_inputs(self, x: torch.Tensor, mask: torch.Tensor, binsize:int):

        bsz, max_n_bins, n_mark = mask.size(0), mask.size(4), x.size(3)
        mask = mask.view(-1, 1, max_n_bins, max_n_bins)
        # --> mask : (bsz x (max_n_interactions + 1)) x 1 x max_n_bins x max_n_bins

        x = x.view(-1, max_n_bins, n_mark)
        # --> x: (bsz x (max_n_interactions + 1)) x max_n_bins x n_maxk
        x = self.embed[f"{binsize}"].lin_proj(x)
        # --> x: (bsz x (max_n_interactions + 1)) x max_n_bins x d_model

        # Embed and linearly project each genomic region independently.
        # Add positional embedding.
        if self.embed[f"{binsize}"].pos_enc:
            x = (
                x
                + self.embed[f"{binsize}"]._pos_enc(
                        dim=self.embed[f"{binsize}"].d_model, 
                        max_len=max_n_bins,
                    )[:, : self.embed[f"{binsize}"].d_model]
                .unsqueeze(0)
                .to(DEVICE)
            )
        
        return x, mask

    def get_attention_weights(self, binsize, x, mask, bias=None):
        _, attention_weights = self.embed[f"{binsize}"].transformer.layers[0].self_att(x, mask)

        attention_weights = attention_weights.cpu().detach().numpy()
        return attention_weights 


if __name__ == "__main__":

    targets = ['bf', 'bs']
    model = ChromoformerClassifier(targets=targets).to(DEVICE)    

    # Dummy data.
    bsz = 8
    i_max = 8

    x_p_2000, x_p_500, x_p_100 = (
        torch.randn([bsz, 1, 20, 7]),
        torch.randn([bsz, 1, 80, 7]),
        torch.randn([bsz, 1, 400, 7]),
    )
    x_pcre_2000, x_pcre_500, x_pcre_100 = (
        torch.randn([bsz, i_max, 20, 7]),
        torch.randn([bsz, i_max, 80, 7]),
        torch.randn([bsz, i_max, 400, 7]),
    )

    pad_mask_p_2000, pad_mask_p_500, pad_mask_p_100 = (
        torch.randn([bsz, 1, 1, 20, 20]).bool(),
        torch.randn([bsz, 1, 1, 80, 80]).bool(),
        torch.randn([bsz, 1, 1, 400, 400]).bool(),
    )
    pad_mask_pcre_2000, pad_mask_pcre_500, pad_mask_pcre_100 = (
        torch.randn([bsz, i_max, 1, 20, 20]).bool(),
        torch.randn([bsz, i_max, 1, 80, 80]).bool(),
        torch.randn([bsz, i_max, 1, 400, 400]).bool(),
    )

    interaction_mask_2000, interaction_mask_500, interaction_mask_100 = (
        torch.randn([bsz, 1, 1 + i_max, 1 + i_max]).bool(),
        torch.randn([bsz, 1, 1 + i_max, 1 + i_max]).bool(),
        torch.randn([bsz, 1, 1 + i_max, 1 + i_max]).bool(),
    )
    interaction_freq = torch.randn([bsz, 1 + i_max, 1 + i_max])

    x_p_2000, x_p_500, x_p_100 = x_p_2000.to(DEVICE), x_p_500.to(DEVICE), x_p_100.to(DEVICE)
    x_pcre_2000, x_pcre_500, x_pcre_100 = (
        x_pcre_2000.to(DEVICE),
        x_pcre_500.to(DEVICE),
        x_pcre_100.to(DEVICE),
    )

    pad_mask_p_2000, pad_mask_p_500, pad_mask_p_100 = (
        pad_mask_p_2000.to(DEVICE),
        pad_mask_p_500.to(DEVICE),
        pad_mask_p_100.to(DEVICE),
    )
    pad_mask_pcre_2000, pad_mask_pcre_500, pad_mask_pcre_100 = (
        pad_mask_pcre_2000.to(DEVICE),
        pad_mask_pcre_500.to(DEVICE),
        pad_mask_pcre_100.to(DEVICE),
    )

    interaction_mask_2000, interaction_mask_500, interaction_mask_100 = (
        interaction_mask_2000.to(DEVICE),
        interaction_mask_500.to(DEVICE),
        interaction_mask_100.to(DEVICE),
    )
    interaction_freq = interaction_freq.to(DEVICE)



    promoter_feats = {
        2000: x_p_2000,
        500: x_p_500,
        100: x_p_100,
    }
    promoter_pad_masks = {
        2000: pad_mask_p_2000,
        500: pad_mask_p_500,
        100: pad_mask_p_100,
    }
    pcre_feats = {
        2000: x_pcre_2000,
        500: x_pcre_500,
        100: x_pcre_100,
    }
    pcre_pad_masks = {
        2000: pad_mask_pcre_2000,
        500: pad_mask_pcre_500,
        100: pad_mask_pcre_100,
    }
    interaction_masks = {
        2000: interaction_mask_2000,
        500: interaction_mask_500,
        100: interaction_mask_100,
    }



    bs, bf = model(
        promoter_feats,
        promoter_pad_masks,
    )

    print(bs.sum())
    print(bs.shape)
    # -0.1900, [8, 1]

    # ckpt = torch.load("test.pt")
    # model_classifier.load_state_dict(ckpt["net"])

    # ckpt = torch.load(
    #     "checkpoints/burst.bf.E116.reg.model.pt"
    # )
    # model_regressor.load_state_dict(ckpt["net"])

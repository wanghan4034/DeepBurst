import gseapy as gp
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd 

eid = "E116"
remove_marks = "H3K36me3"
gene_ids = pd.read_csv(f"extra/datasets/benchmark/Promoterformer/results/{eid}_{remove_marks}_associated_genes.csv")
gene_id2names = pd.read_csv("extra/datasets/burst/raw_data/H1/features.tsv",sep="\t",names=['gene_id','gene_name','type'])
gene_names = pd.merge(gene_ids['gene_id'],gene_id2names[['gene_id','gene_name']],on='gene_id')

def plot_gene_enrich2(data,file_name,species,top_term=20):

    matplotlib.rcParams.update({'font.size': 8})
   
    data.Term = data.Term.str.split(" \(GO").str[0]
    try:
        ax = gp.dotplot(data,top_term=top_term, figsize=(6,6),size=5, cmap = plt.cm.autumn_r,size_scale=40)
        ax.grid(False)
        ax.tick_params(axis='x', labelsize=8)  # x轴刻度字体大小
        ax.tick_params(axis='y', labelsize=8)  # y轴刻度字体大小
        plt.show()
        plt.savefig(file_name,dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("the species {} has no significant pathway".format(species),e)
        return []
    

def get_go_enrich(gene_set_busrt):

    result = gp.enrichr(gene_set_busrt,
                        gene_sets='GO_Biological_Process_2021',
                        outdir=None)
    result_sig = result.res2d[result.res2d['Adjusted P-value']<0.05]

    # result2 = gp.enrichr(gene_set_mean,
    #                     gene_sets='GO_Biological_Process_2021',
    #                     outdir=None)
    # result2_sig = result2.res2d[result2.res2d['Adjusted P-value']<0.05]
    if result_sig.shape[0]>0:
        file_name = "figures/DE_genes/pathway.jpg"
        plot_gene_enrich2(result_sig,file_name,"human")

gene_set_busrt = list(gene_names['gene_name'])
get_go_enrich(gene_set_busrt)
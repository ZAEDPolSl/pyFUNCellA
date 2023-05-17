def remove_redundant_genesets(genesets, genes):
    incorrect = []
    for i, (gs_name, geneset_genes) in enumerate(genesets.items()):
        genes_in_ds = [gene in geneset_genes for gene in genes]
        N_gs = sum(genes_in_ds)  # number of genes in GS
        if N_gs == 0:
            incorrect.append(gs_name)
    filtered_genesets = {
        key: genesets[key] for key in genesets.keys() if key not in incorrect
    }
    return filtered_genesets

import math

from gsea import calculate_gene_zscores, rank_expressions


def get_ranks(data):
    print("Transforming the geneset...\n")
    transformed = calculate_gene_zscores(data)
    print("Getting the ranks...\n")
    ranks = rank_expressions(transformed)
    return ranks

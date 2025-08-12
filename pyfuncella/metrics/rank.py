from scipy.stats import rankdata


def rank_genes(data, descending=True, ordinal=False):
    # for each patient return the gene ranking
    if ordinal:
        return (-data).argsort(axis=0).argsort(axis=0) + 1
    if descending:
        return rankdata(-data, method="average", axis=0)
    return rankdata(data, method="average", axis=0)

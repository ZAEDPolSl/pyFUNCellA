from pyfuncella.preprocess.clean_geneset import remove_redundant_genesets


def test_remove_geneset():
    genesets = {"gs1": {"a", "b", "c"}, "gs2": {"v", "u", "m"}}
    expected_genesets = {"gs1": {"a", "b", "c"}}
    genes = ["a", "b", "d", "e"]
    res = remove_redundant_genesets(genesets, genes)
    assert res == expected_genesets


def test_remove_geneset_case_sensitive():
    genesets = {"gs1": {"a", "b", "c"}, "gs2": {"v", "u", "m"}}
    expected_genesets = {"gs1": {"a", "b", "c"}}
    genes = ["a", "b", "d", "V"]
    res = remove_redundant_genesets(genesets, genes)
    assert res == expected_genesets

import numpy as np
import pandas as pd
import pytest

from enrichment_auc.preprocess import entrez2genes


def test_drops_nan_genes():
    entrez = ["enttrez_1", "entrez_2", "entrez_3", "entrez_4"]
    d = {"p1": [1.0, 2.01, 5.0, 7.0], "p2": [1.001, 2.0, 6.0, 0.01]}
    d_expected = {"p1": [1.0, 5.0], "p2": [1.001, 6.0]}
    cells = pd.DataFrame(data=d, index=entrez)
    gene_name = "external_gene_name"
    genes_names = ["gen1", np.nan, "gen3", np.nan]
    genes = pd.DataFrame(data={gene_name: genes_names}, index=entrez)
    new_cells = entrez2genes.entrez2genes(genes, cells, gene_name)
    expected = pd.DataFrame(data=d_expected, index=["gen1", "gen3"])
    assert new_cells.equals(expected)


def test_drops_empty_genes():
    entrez = ["enttrez_1", "entrez_2", "entrez_3", "entrez_4"]
    d = {"p1": [1.0, 2.01, 5.0, 7.0], "p2": [1.001, 2.0, 6.0, 0.01]}
    d_expected = {"p1": [1.0, 5.0], "p2": [1.001, 6.0]}
    cells = pd.DataFrame(data=d, index=entrez)
    gene_name = "external_gene_name"
    genes_names = ["gen1", "", "gen3", ""]
    genes = pd.DataFrame(data={gene_name: genes_names}, index=entrez)
    new_cells = entrez2genes.entrez2genes(genes, cells, gene_name)
    expected = pd.DataFrame(data=d_expected, index=["gen1", "gen3"])
    assert new_cells.equals(expected)


def test_skips_unoccuring_genes():
    # tests if the genes not in cell expr are not inserted
    entrez = ["enttrez_1", "entrez_2", "entrez_3", "entrez_4"]
    d = {"p1": [1.0, 2.01, 5.0, 7.0], "p2": [1.001, 2.0, 6.0, 0.01]}
    cells = pd.DataFrame(data=d, index=entrez)

    gene_name = "external_gene_name"
    new_entrez = ["enttrez_1", "entrez_2", "entrez_3", "entrez_4", "entrez_5"]
    genes_names = ["gen1", "gen2", "gen3", "gen4", "gen5"]
    genes = pd.DataFrame(data={gene_name: genes_names}, index=new_entrez)

    new_cells = entrez2genes.entrez2genes(genes, cells, gene_name)
    expected = pd.DataFrame(data=d, index=["gen1", "gen2", "gen3", "gen4"])
    assert new_cells.equals(expected)


def test_removes_unoccuring_entrez():
    # tests if the expressions not in genes are removed
    entrez = ["enttrez_1", "entrez_2", "entrez_3", "entrez_4"]
    d = {"p1": [1.0, 2.01, 5.0, 7.0], "p2": [1.001, 2.0, 6.0, 5.0001]}
    cells = pd.DataFrame(data=d, index=entrez)

    gene_name = "external_gene_name"
    new_entrez = ["enttrez_1", "entrez_2", "entrez_3"]
    genes_names = ["gen1", "gen2", "gen3"]
    genes = pd.DataFrame(data={gene_name: genes_names}, index=new_entrez)

    new_cells = entrez2genes.entrez2genes(genes, cells, gene_name)

    d_expected = {"p1": [1.0, 2.01, 5.0], "p2": [1.001, 2.0, 6.0]}
    expected = pd.DataFrame(data=d_expected, index=["gen1", "gen2", "gen3"])
    assert new_cells.equals(expected)


def test_leaves_unocurring():
    entrez = ["enttrez_1", "entrez_2", "entrez_3", "entrez_4"]
    d = {"p1": [1.0, 2.01, 5.0, 7.0], "p2": [1.001, 2.0, 6.0, 5.0001]}
    cells = pd.DataFrame(data=d, index=entrez)

    gene_name = "external_gene_name"
    new_entrez = ["enttrez_1", "entrez_2", "entrez_3", "entrez_5"]
    genes_names = ["gen1", "gen2", "gen3", "gen5"]
    genes = pd.DataFrame(data={gene_name: genes_names}, index=new_entrez)

    new_cells = entrez2genes.entrez2genes(genes, cells, gene_name)

    d_expected = {"p1": [1.0, 2.01, 5.0], "p2": [1.001, 2.0, 6.0]}
    expected = pd.DataFrame(data=d_expected, index=["gen1", "gen2", "gen3"])
    assert new_cells.equals(expected)


def test_removes_duplicates():
    entrez = ["enttrez_1", "entrez_2", "entrez_3", "entrez_4"]
    d = {"p1": [1.0, 2.01, 5.0, 7.0], "p2": [1.001, 2.0, 6.0, 5.0]}
    cells = pd.DataFrame(data=d, index=entrez)

    gene_name = "external_gene_name"
    new_entrez = ["enttrez_1", "entrez_2", "entrez_3", "entrez_4"]
    genes_names = ["gen1", "gen2", "gen3", "gen3"]
    genes = pd.DataFrame(data={gene_name: genes_names}, index=new_entrez)

    new_cells = entrez2genes.entrez2genes(genes, cells, gene_name)

    d_expected = {"p1": [1.0, 2.01, 7.0], "p2": [1.001, 2.0, 5.0]}
    expected = pd.DataFrame(data=d_expected, index=["gen1", "gen2", "gen3"])
    assert new_cells.equals(expected)


def test_removes_many_duplicates():
    entrez = ["entrez_0", "enttrez_1", "entrez_2", "entrez_3", "entrez_4", "entrez_5"]
    d = {
        "p1": [0.0, 1.0, 2.01, 5.0, 7.0, 3.002],
        "p2": [0.001, 1.001, 2.0, 6.0, 5.0, 3],
    }
    cells = pd.DataFrame(data=d, index=entrez)

    gene_name = "external_gene_name"
    genes_names = ["gen3", "gen1", "gen2", "gen3", "gen3", "gen2"]
    genes = pd.DataFrame(data={gene_name: genes_names}, index=entrez)

    new_cells = entrez2genes.entrez2genes(genes, cells, gene_name)

    d_expected = {"p1": [1.0, 2.01, 7.0], "p2": [1.001, 2.0, 5.0]}
    expected = pd.DataFrame(data=d_expected, index=["gen1", "gen2", "gen3"])
    assert new_cells.equals(expected)


def test_warns_no_external():
    entrez = ["enttrez_1", "entrez_2", "entrez_3"]
    d = {"p1": [1.0, 2.01, 5.0], "p2": [1.001, 2.0, 6.0]}
    cells = pd.DataFrame(data=d, index=entrez)
    gene_name = ""
    genes_names = ["gen1", "gen2", "gen3"]
    some_col = [1, 2, 3]
    genes = pd.DataFrame(
        data={"external_gene_name": genes_names, "meta": some_col}, index=entrez
    )
    with pytest.warns(UserWarning):
        new_cells = entrez2genes.entrez2genes(genes, cells, gene_name)
    expected = pd.DataFrame(data=d, index=genes_names)
    assert new_cells.equals(expected)


def test_adds_index_only():
    # checks if cells are not appended with metadata during conversion
    entrez = ["enttrez_1", "entrez_2", "entrez_3"]
    d = {"p1": [1.0, 2.01, 5.0], "p2": [1.001, 2.0, 6.0]}
    cells = pd.DataFrame(data=d, index=entrez)
    gene_name = "external_gene_name"
    genes_names = ["gen1", "gen2", "gen3"]
    some_col = [1, 2, 3]
    genes = pd.DataFrame(data={gene_name: genes_names, "meta": some_col}, index=entrez)

    new_cells = entrez2genes.entrez2genes(genes, cells, gene_name)
    expected = pd.DataFrame(data=d, index=genes_names)
    assert new_cells.equals(expected)

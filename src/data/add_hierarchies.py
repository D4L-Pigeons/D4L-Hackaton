import sys

import anndata as ad
import numpy as np

from src.global_utils.paths import RAW_ANNDATA_HIERARCHY_PATH

default_hierarchies_mapping = {
    "cite-seq": {
        "CD14+ Mono": "Monocytes",
        "CD16+ Mono": "Monocytes",
        "CD4+ T activated": "T Cells",
        "CD4+ T naive": "T Cells",
        "CD8+ T naive": "T Cells",
        "CD8+ T CD57+ CD45RO+": "T Cells",
        "CD8+ T CD57+ CD45RA+": "T Cells",
        "CD8+ T TIGIT+ CD45RO+": "T Cells",
        "CD8+ T TIGIT+ CD45RA+": "T Cells",
        "CD8+ T CD49f+": "T Cells",
        "CD8+ T CD69+ CD45RO+": "T Cells",
        "CD8+ T CD69+ CD45RA+": "T Cells",
        "CD8+ T naive CD127+ CD26- CD101-": "T Cells",
        "CD4+ T activated integrinB7+": "T Cells",
        "CD4+ T CD314+ CD45RA+": "T Cells",
        "MAIT": "T Cells",
        "gdT CD158b+": "T Cells",
        "gdT TCRVD2+": "T Cells",
        "dnT": "T Cells",
        "T reg": "T Cells",
        "T prog cycling": "T Cells",
        "Naive CD20+ B IGKC+": "B Cells",
        "Naive CD20+ B IGKC-": "B Cells",
        "Transitional B": "B Cells",
        "B1 B IGKC+": "B Cells",
        "B1 B IGKC-": "B Cells",
        "Plasma cell IGKC+": "Plasma Cells",
        "Plasma cell IGKC-": "Plasma Cells",
        "Plasmablast IGKC+": "Plasma Cells",
        "Plasmablast IGKC-": "Plasma Cells",
        "NK": "NK Cells",
        "NK CD158e1+": "NK Cells",
        "cDC1": "Dendritic Cells",
        "cDC2": "Dendritic Cells",
        "pDC": "Dendritic Cells",
        "HSC": "Progenitors and Stem Cells",
        "Lymph prog": "Progenitors and Stem Cells",
        "G/M prog": "Progenitors and Stem Cells",
        "MK/E prog": "Progenitors and Stem Cells",
        "Reticulocyte": "Erythroid Lineage",
        "Erythroblast": "Erythroid Lineage",
        "Proerythroblast": "Erythroid Lineage",
        "Normoblast": "Erythroid Lineage",
        "ILC": "Innate Lymphoid Cells",
        "ILC1": "Innate Lymphoid Cells",
    }
}


def add_second_hierarchy(
    data, dataset_name, hierarchies_mapping: dict = default_hierarchies_mapping
):
    labels = data.obs["cell_type"]
    print(labels)
    data.obs["second_hierarchy"] = labels.map(hierarchies_mapping[dataset_name])
    print("Second hierarchy mapping:", data.obs["second_hierarchy"].head())

    data.write(filename=str(RAW_ANNDATA_HIERARCHY_PATH / dataset_name))
    return data


def main(
    dataset, dataset_name="cite-seq"
):  # TODO: for now set default. but remove later
    add_second_hierarchy(dataset, dataset_name)


if __name__ == "__main__":
    main()

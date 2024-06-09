import numpy as np
import anndata as ad

from utils.paths import RAW_ANNDATA_HIERARCHY_PATH

default_hierarchies_mapping = {
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


def add_second_hierarchy(
    _data, hierarchies_mapping: dict = default_hierarchies_mapping
):
    if RAW_ANNDATA_HIERARCHY_PATH.exists():
        return ad.read_h5ad(RAW_ANNDATA_HIERARCHY_PATH)

    labels = _data.obs["cell_type"]
    print(labels)
    _data.obs["second_hierarchy"] = labels.map(hierarchies_mapping)
    # print(np.unique(_data.obs["cell_type"].values))
    # print(np.unique(_data.obs["second_hierarchy"].values))
    print("Second hierarchy mapping:", _data.obs["second_hierarchy"].head())

    _data.write(filename=RAW_ANNDATA_HIERARCHY_PATH)
    return _data


def main():
    add_second_hierarchy(default_hierarchies_mapping)


if __name__ == "__main__":
    main()

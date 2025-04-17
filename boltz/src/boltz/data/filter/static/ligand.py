import numpy as np

from boltz.data import const
from boltz.data.types import Structure
from boltz.data.filter.static.filter import StaticFilter

LIGAND_EXCLUSION = {
    "144",
    "15P",
    "1PE",
    "2F2",
    "2JC",
    "3HR",
    "3SY",
    "7N5",
    "7PE",
    "9JE",
    "AAE",
    "ABA",
    "ACE",
    "ACN",
    "ACT",
    "ACY",
    "AZI",
    "BAM",
    "BCN",
    "BCT",
    "BDN",
    "BEN",
    "BME",
    "BO3",
    "BTB",
    "BTC",
    "BU1",
    "C8E",
    "CAD",
    "CAQ",
    "CBM",
    "CCN",
    "CIT",
    "CL",
    "CLR",
    "CM",
    "CMO",
    "CO3",
    "CPT",
    "CXS",
    "D10",
    "DEP",
    "DIO",
    "DMS",
    "DN",
    "DOD",
    "DOX",
    "EDO",
    "EEE",
    "EGL",
    "EOH",
    "EOX",
    "EPE",
    "ETF",
    "FCY",
    "FJO",
    "FLC",
    "FMT",
    "FW5",
    "GOL",
    "GSH",
    "GTT",
    "GYF",
    "HED",
    "IHP",
    "IHS",
    "IMD",
    "IOD",
    "IPA",
    "IPH",
    "LDA",
    "MB3",
    "MEG",
    "MES",
    "MLA",
    "MLI",
    "MOH",
    "MPD",
    "MRD",
    "MSE",
    "MYR",
    "N",
    "NA",
    "NH2",
    "NH4",
    "NHE",
    "NO3",
    "O4B",
    "OHE",
    "OLA",
    "OLC",
    "OMB",
    "OME",
    "OXA",
    "P6G",
    "PE3",
    "PE4",
    "PEG",
    "PEO",
    "PEP",
    "PG0",
    "PG4",
    "PGE",
    "PGR",
    "PLM",
    "PO4",
    "POL",
    "POP",
    "PVO",
    "SAR",
    "SCN",
    "SEO",
    "SEP",
    "SIN",
    "SO4",
    "SPD",
    "SPM",
    "SR",
    "STE",
    "STO",
    "STU",
    "TAR",
    "TBU",
    "TME",
    "TPO",
    "TRS",
    "UNK",
    "UNL",
    "UNX",
    "UPL",
    "URE",
}


class ExcludedLigands(StaticFilter):
    """Filter excluded ligands."""

    def filter(self, structure: Structure) -> np.ndarray:
        """Filter excluded ligands.

        Parameters
        ----------
        structure : Structure
            The structure to filter chains from.

        Returns
        -------
        np.ndarray
            The chains to keep, as a boolean mask.

        """
        valid = np.ones(len(structure.chains), dtype=bool)

        for i, chain in enumerate(structure.chains):
            if chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]:
                continue

            res_start = chain["res_idx"]
            res_end = res_start + chain["res_num"]
            residues = structure.residues[res_start:res_end]
            if any(res["name"] in LIGAND_EXCLUSION for res in residues):
                valid[i] = 0

        return valid

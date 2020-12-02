import os
import pytest

import numpy as np

from syntfbs.motifs import  Motifs
from syntfbs.data_generator import DataGenerator


def relative_to_this_folder(pathname):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, pathname)


@pytest.fixture
def jaspar_motifs():
    dirname = relative_to_this_folder(
        "../data/jaspar/JASPAR2020_CORE_non-redundant_pfms_jaspar")

    filename = os.path.join(dirname, "MA0001.2.jaspar")
    assert os.path.exists(filename), filename
    return Motifs.load_jaspar(filename)


@pytest.fixture
def gata_motifs():
    dirname = relative_to_this_folder(
        "../data/jaspar/JASPAR2020_CORE_non-redundant_pfms_jaspar")
    gata_motifs = [
        "MA0035.4.jaspar",
        "MA0036.3.jaspar",
        "MA0037.3.jaspar",
        "MA0140.2.jaspar",
        "MA0482.2.jaspar",
        "MA0766.2.jaspar",
        "MA1013.1.jaspar",
        "MA1014.1.jaspar",
        "MA1015.1.jaspar",
        "MA1016.1.jaspar",
        "MA1017.1.jaspar",
        "MA1018.1.jaspar",
        "MA1104.2.jaspar",
        "MA1323.1.jaspar",
        "MA1324.1.jaspar",
        "MA1325.1.jaspar",
        "MA1396.1.jaspar",    
    ]

    return [
        Motifs.load_jaspar(os.path.join(dirname, mfn))
        for mfn in gata_motifs
    ]



def test_generator(jaspar_motifs):

    generator = DataGenerator(
        [jaspar_motifs], filename="temp.h5",
        slice_length = 100, bin_length=50,
        chunk_size=50)


    # generator.generate_chunk(50)
    generator.generate(2)


def test_generator_multi(gata_motifs):

    generator = DataGenerator(
        gata_motifs, filename="temp.h5",
        slice_length = 100, bin_length=50,
        chunk_size=50)

    generator._select_motifs()


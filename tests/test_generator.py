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


def test_generator(jaspar_motifs):

    generator = DataGenerator(
        [jaspar_motifs], filename="temp.h5",
        slice_length = 100, bin_length=50,
        chunk_size=50)


    # generator.generate_chunk(50)
    generator.generate(2)


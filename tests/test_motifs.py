import os
import pytest

import numpy as np

from syntfbs.motifs import load_jaspar, \
    random_sequence_generator, \
    Motifs


def relative_to_this_folder(pathname):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, pathname)


@pytest.fixture
def jaspar_filename():
    dirname = relative_to_this_folder(
        "../data/jaspar/JASPAR2020_CORE_non-redundant_pfms_jaspar")

    filename = os.path.join(dirname, "MA0001.2.jaspar")
    assert os.path.exists(filename), filename
    return filename


def test_load_jaspar(jaspar_filename):

    motif = load_jaspar(jaspar_filename)
    assert motif is not None


def test_motif_experiments(jaspar_filename):

    motif = load_jaspar(jaspar_filename)
    assert motif is not None
    print(motif)

    # print(motif.background)
    # print(motif.pseudocounts)
    # print(motif.counts)
    pwm = motif.counts.normalize(pseudocounts=0.5)
    print(pwm)
    print(motif.pwm)

    # print(pwm.consensus)
    # print(pwm.anticonsensus)
    # print(pwm.degenerate_consensus)
    # print(pwm.degenerate_consensus)    
    # print(38*"=")
    # print(motif.counts)
    # print(motif.pwm)
    # print(motif.pssm)

    # print(motif.pwm)
    # print(dir(motif.pwm))

    # # for i in motif.pwm:
    # #     print(i)
    # #     print(motif.pwm[i])
    
    # for i in range(len(motif.pwm)):
    #     print(motif.pwm[i])

    # print(motif.pwm.length)
    # print(motif.pwm.values())


    # pwm = np.array([motif.pwm[let] for let in motif.pwm])
    # print(pwm.shape)
    # print(pwm)

    # print(pwm[:, 0])
    # print(pwm[:, 0].shape)
    # print(list(motif.pwm.keys()))
    # res = generate_sequence(pwm)

    # print(len(res))

    # pssm = motif.pssm
    # for p, s in pssm.search(res):
    #     print(p, s)


def test_motif_sequence_generator(jaspar_filename):
    motif = Motifs.load_jaspar(jaspar_filename)

    for count, seq in enumerate(motif.sequence_generator()):
        print(seq, motif.sequence_score(seq))

        if count >= 5:
            break

    print(50*"=")
    last = seq
    print("last:", last)

    for count, seq in enumerate(motif.mutation_generator(last)):
        print(seq, motif.sequence_score(seq))

        if count >= 5:
            break

    print(50*"=")

    generator = random_sequence_generator(len(motif))
    for count, seq in enumerate(generator):
        print(seq, motif.sequence_score(seq))

        if count >= 5:
            break


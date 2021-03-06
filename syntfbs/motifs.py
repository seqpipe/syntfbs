import random
import numpy as np

from Bio import motifs


ALPHABET = ["A", "C", "G", "T"]


def load_jaspar(filename):
    with open(filename, "rt") as handle:
        srf = motifs.parse(handle, "jaspar")
        assert srf is not None
        assert len(srf) == 1
        return srf[0]


def random_sequence_generator(length):
    b_pr = np.array([0.25, 0.25, 0.25, 0.25])

    while True:

        seq = np.random.choice(ALPHABET, length, p=b_pr)
        result = "".join(seq)
    
        yield result


def mutate(sequence, mutations=1):
    assert mutations >= 0

    res = list(sequence)

    while mutations > 0:
        nuc = random.choice(ALPHABET)
        pos = random.randint(0, len(sequence) - 1)
        res[pos] = nuc
        mutations -= 1

    return "".join(res)


def mutation_generator(seq, total_count=-1):
    res = list(seq)
    while total_count != 0:
        yield "".join(res)
        nuc = random.choice(ALPHABET)
        pos = random.randint(0, len(seq) - 1)
        res[pos] = nuc

        total_count -= 1



class Motifs:

    def __init__(self, motif):
        self.motif = motif
        self.pwm = self.motif2pwm(motif)

    def __len__(self):
        return len(self.motif)

    @staticmethod
    def motif2pwm(motif):
        pwm = motif.counts.normalize(pseudocounts=0.5)
        pwm = np.array([pwm[let] for let in ALPHABET])
        return pwm
    
    @staticmethod
    def load_jaspar(filename):
        with open(filename, "rt") as handle:
            srf = motifs.parse(handle, "jaspar")
            assert srf is not None
            assert len(srf) == 1
            return Motifs(srf[0])

    def sequence_score(self, seq):
        assert len(seq) == len(self.motif)

        b_pr = np.ones(len(seq), dtype=np.float) * 0.25
        s_pr = np.ones(len(seq), dtype=np.float)
        for column, nuc in enumerate(seq):
            nuc_index = ALPHABET.index(nuc)
            assert 0 <= nuc_index < 4
            s_pr[column] = self.pwm[nuc_index, column]
        
        score = s_pr / b_pr
        score = np.product(score)

        if not np.isfinite(score):
            print(self.motif, seq, score, b_pr, s_pr)

        return score

    def generate(self):
        rows, cols = self.pwm.shape
        assert rows == 4

        seq = []
        for i in range(cols):
            column = self.pwm[:, i]

            res = np.random.choice(ALPHABET, 1, p=column)
            seq.append(res[0])
        result = "".join(seq)
        return result

    def sequence_generator(self):

        while True:
            result = self.generate()
            yield result

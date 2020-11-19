import numpy as np


DEEPSEA_MAPPING = {
    "A": np.array([1, 0, 0, 0]).T,
    "G": np.array([0, 1, 0, 0]).T,
    "C": np.array([0, 0, 1, 0]).T,
    "T": np.array([0, 0, 0, 1]).T,
    "N": np.array([0, 0, 0, 0]).T,
}


SPLICE_AI_MAPPING = {
    "A": np.array([1, 0, 0, 0]).T,
    "C": np.array([0, 1, 0, 0]).T,
    "G": np.array([0, 0, 1, 0]).T,
    "T": np.array([0, 0, 0, 1]).T,
    "N": np.array([0, 0, 0, 0]).T,
}


DEFAULT_ONE_HOT_MAPPING = SPLICE_AI_MAPPING


def one_hot_encode(seq, mapping=DEFAULT_ONE_HOT_MAPPING):
    assert isinstance(seq, str), seq

    result = np.zeros((4, len(seq)),np.int8)
    for seq_index, nucleotide in enumerate(seq):
        if nucleotide not in set(["N", "A", "C", "G", "T"]):
            print("ERROR:", seq_index, nucleotide, seq)

        result[:, seq_index] = mapping.get(nucleotide, mapping["N"])
    return result


def one_hot_decode(seq, mapping=DEFAULT_ONE_HOT_MAPPING):
    result = (seq.shape[0] + 1) * ["N"]
    alphabet = list(mapping.keys())
    for seq_index in range(len(seq) + 1):
        one_hot_nucleotide = list(seq[:, seq_index].T)
        assert len(one_hot_nucleotide) == 4

        if 1 not in one_hot_nucleotide:
            result[seq_index] = "N"
        else:
            index = one_hot_nucleotide.index(1)
            result[seq_index] = alphabet[index]
    return "".join(result)

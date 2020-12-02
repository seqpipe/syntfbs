
import os
import random
import math
import h5py

import numpy as np
from numpy.core.fromnumeric import shape

from syntfbs.motifs import random_sequence_generator, mutation_generator, \
    mutate, Motifs

from syntfbs.encoding import one_hot_encode


class DataGenerator:

    def __init__(
        self, motifs, filename,
        slice_length=1000, bin_length=200, chunk_size=200):
    
        self.motifs = motifs
        self.filename = filename

        self.slice_length = slice_length
        self.bin_lenght = bin_length
        self.bin_start = (self.slice_length - self.bin_lenght) // 2
        self.bin_end = self.bin_start + self.bin_lenght

        self.chunk_size = chunk_size
        self.mutations = 10
        self.noise = 0.33

        self.data = []
        self.sequences = []
        self.targets = []

    def _build_data_volume(self):
        shape = (0, 4, self.slice_length)
        chunks = (self.chunk_size, 4, self.slice_length)

        maxshape = (None, shape[1], shape[2])
        data_volume = self.hdf5.create_dataset(
            "data", shape, chunks=chunks, maxshape=maxshape, dtype=np.int8,
            compression="gzip"
        )
        return data_volume

    def _write_data_volume(self, data_volume, data_chunk):
        start = data_volume.shape[0]
        print("data:", data_volume.shape, start, len(data_chunk))
        data_volume.resize(start + len(data_chunk), axis=0)

        data_array = np.array(
            [one_hot_encode(data) for data in data_chunk])
        data_volume[start:start + len(data_chunk)] = data_array

        return data_volume

    def _build_labels_volume(self):
        shape = (0, len(self.motifs))
        chunk = (self.chunk_size, len(self.motifs))

        maxshape = (None, shape[1])
        labels_volume = self.hdf5.create_dataset(
            "labels", shape, chunks=chunk, maxshape=maxshape,
            dtype=np.float64,
            compression='gzip'
        )
        return labels_volume

    def _write_labels_volume(self, labels_volume, labels_chunk):
        start = labels_volume.shape[0]
        print("labels:", labels_volume.shape, start, len(labels_chunk))
        labels_volume.resize(start + len(labels_chunk), axis=0)
        labels_volume[start:start+len(labels_chunk)] = np.array(labels_chunk)

    def _select_rundom_mutations(self):
        mutations = np.random.poisson(
            lam=math.floor(self.mutations/2) + 1)
        mutations = min(self.mutations, mutations)
        return mutations

    def _select_motifs(self):
        if len(self.motifs) == 1:
            return [{
                "motif": 0,
                "mutations": self._select_rundom_mutations()
            }]
        result = []
        for _ in range(math.floor(len(self.motifs) * self.noise)):
            motif_index = random.randint(0, len(self.motifs) - 1)
            result.append({
                "motif": motif_index,
                "mutations": self._select_rundom_mutations(),
            })
        return result

    def generate_chunk(self, total_count):
        
        data = []
        labels = []

        base_generator = random_sequence_generator(self.slice_length)
        
        while len(data) < total_count:
            # generate signal
            for _ in range(math.floor(self.mutations*len(self.motifs))):
                base_sequence = next(base_generator)
                result = base_sequence

                selected_motifs = self._select_motifs()
                scores = np.zeros(shape=len(self.motifs), dtype=np.float64)

                for desc in selected_motifs:
                    motif_index = desc["motif"]
                    motif = self.motifs[motif_index]
                    mutations = desc["mutations"]
                    motif_sequence = motif.generate()
                    motif_sequence = mutate(motif_sequence, mutations)
                    motif_score = motif.sequence_score(motif_sequence)
                    if motif_score < 50:
                        motif_score = 0.0
                    scores[motif_index] = motif_score

                    pos = random.randint(
                        self.bin_start, self.bin_end - len(motif) - 1)

                    result = f"{result[:pos]}{motif_sequence}" \
                        f"{result[(pos + len(motif)):]}"
                    assert len(base_sequence) == len(result)

                data.append(result)
                labels.append(scores)
                if len(data) >= total_count:
                    return data, labels

            # generate pure noise
            for _ in range(math.floor(self.noise * self.mutations)):
                sequence = next(base_generator)
                # print(sequence, "noise")

                data.append(sequence)
                labels.append(np.zeros(len(self.motifs), dtype=np.float64))

                if len(data) >= total_count:
                    return data, labels

        return data, labels


    def generate(self, chunks):
        self.hdf5 = h5py.File(self.filename, 'w')

        data_volume = self._build_data_volume()
        labels_volume = self._build_labels_volume()

        for i in range(chunks):
            print(80*"=")
            print("chunk=", i)

            data_chunk, labels_chunk = self.generate_chunk(self.chunk_size)
            print("data chunk generated...")
            self._write_data_volume(data_volume, data_chunk)
            self._write_labels_volume(labels_volume, labels_chunk)

        self.hdf5.close()



if __name__ == "__main__":
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

    fox_motifs = [
        "MA0030.1.jaspar",
        "MA0031.1.jaspar",
        "MA0032.2.jaspar",
        "MA0033.2.jaspar",
        "MA0040.1.jaspar",
        "MA0041.1.jaspar",
        "MA0042.2.jaspar",
        "MA0047.3.jaspar",
        "MA0148.4.jaspar",
        "MA0157.2.jaspar",
        "MA0479.1.jaspar",
        "MA0480.1.jaspar",
        "MA0481.3.jaspar",
        "MA0593.1.jaspar",
        "MA0613.1.jaspar",
        "MA0614.1.jaspar",
        "MA0845.1.jaspar",
        "MA0846.1.jaspar",
        "MA0847.2.jaspar",
        "MA0848.1.jaspar",
        "MA0849.1.jaspar",
        "MA0850.1.jaspar",
        "MA0851.1.jaspar",
        "MA0852.2.jaspar",
        "MA1103.2.jaspar",
        "MA1487.1.jaspar",
        "MA1489.1.jaspar",
        "MA1606.1.jaspar",
        "MA1607.1.jaspar",
        "MA1683.1.jaspar",
        "MA1684.1.jaspar",
    ]


    motif_filename = \
        "data/jaspar/JASPAR2020_CORE_non-redundant_pfms_jaspar/MA0035.4.jaspar"    
    data_dirname = "data/jaspar/JASPAR2020_CORE_non-redundant_pfms_jaspar/"

    motif = Motifs.load_jaspar(motif_filename)
    motifs = [motif]

    motifs = [
        Motifs.load_jaspar(os.path.join(data_dirname, mfn))
        for mfn in gata_motifs
    ]

    motifs = [
        Motifs.load_jaspar(os.path.join(data_dirname, mfn))
        for mfn in fox_motifs
    ]

    generator = DataGenerator(
        motifs,
        "FOX_train.h5",
        slice_length=1000, bin_length=1000,
        chunk_size=10_000)
    
    generator.generate(2)

    generator = DataGenerator(
        motifs,
        "FOX_test.h5",
        slice_length=1000, bin_length=1000,
        chunk_size=10_000)
    
    generator.generate(1)

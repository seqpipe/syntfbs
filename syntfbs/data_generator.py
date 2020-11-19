import random
import math
import h5py

import numpy as np

from syntfbs.motifs import random_sequence_generator, mutation_generator, \
    Motifs

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
        self.mutations = 4
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
            # compression="gzip"
        )
        return data_volume

    def _write_data_volume(self, data_volume, data_chunk):
        start = data_volume.shape[0]
        print("data:", data_volume.shape, start, len(data_chunk))
        data_volume.resize(start + len(data_chunk), axis=0)

        for index, sequence in enumerate(data_chunk):
            one_hot = one_hot_encode(sequence)
            data_volume[start + index, :, :] = one_hot
            if index % 1000 == 0:
                print(f"storing {index}/{len(data_chunk)}")

        return data_volume

    def _build_labels_volume(self):
        shape = (0, len(self.motifs))
        chunk = (self.chunk_size, len(self.motifs))

        maxshape = (None, shape[1])
        labels_volume = self.hdf5.create_dataset(
            "labels", shape, chunks=chunk, maxshape=maxshape,
            dtype=np.float16,
            # compression='gzip'
        )
        return labels_volume

    def _write_labels_volume(self, labels_volume, labels_chunk):
        start = labels_volume.shape[0]
        print("labels:", labels_volume.shape, start, len(labels_chunk))
        labels_volume.resize(start + len(labels_chunk), axis=0)
        for index, labels in enumerate(labels_chunk):
            labels_volume[start + index, :] = np.array(labels)


    def generate_chunk(self, total_count):
        
        data = []
        labels = []

        base_generator = random_sequence_generator(self.slice_length)
        
        while len(data) < total_count:
            for motif in self.motifs:
                motif_generator = motif.sequence_generator()

                motif_sequence = next(motif_generator)
                
                for mutation_sequence in mutation_generator(
                        motif_sequence, self.mutations):

                    score = motif.sequence_score(mutation_sequence)

                    pos = random.randint(
                        self.bin_start, self.bin_end - len(motif) - 1)

                    base_sequence = next(base_generator)
                    result = f"{base_sequence[:pos]}{motif_sequence}" \
                        f"{base_sequence[(pos + len(motif)):]}"
                    assert len(base_sequence) == len(result)

                    # print(result, score)
                    if score < 50:
                        score = 0.0

                    data.append(result)
                    labels.append([score])
                    if len(data) >= total_count:
                        return data, labels

                for _ in range(math.floor(self.noise * self.mutations)):
                    sequence = next(base_generator)
                    # print(sequence, "noise")

                    data.append(sequence)
                    labels.append([0.0])

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

            self._write_data_volume(data_volume, data_chunk)
            self._write_labels_volume(labels_volume, labels_chunk)

        self.hdf5.close()



if __name__ == "__main__":
    motif_filename = \
        "data/jaspar/JASPAR2020_CORE_non-redundant_pfms_jaspar/MA0035.4.jaspar"
    
    motif = Motifs.load_jaspar(motif_filename)

    # generator = DataGenerator(
    #     [motif],
    #     "MA0035_4_subst_train.h5",
    #     slice_length=1000, bin_length=100,
    #     chunk_size=10_000)
    
    # generator.generate(10)

    generator = DataGenerator(
        [motif],
        "MA0035_4_subst_test.h5",
        slice_length=1000, bin_length=100,
        chunk_size=10_000)
    
    generator.generate(1)

import h5py
import numpy as np


def main(filename):
    print("\n\n")
    print(80*"=")
    print("dataset:", filename)
    dataset = h5py.File(filename, "r")
    labels = dataset["labels"][:]
    print("shape:    ", labels.shape)
    print("")
    print("max label: ", np.max(labels))
    print("      >50: ", np.sum(labels > 50))
    print("    >1000: ", np.sum(labels > 1000))
    print("   >10000: ", np.sum(labels > 10_000))
    print("")
    print("columns >    50:", np.sum(np.sum(labels > 50, axis=1) > 0))
    print("columns >  1000:", np.sum(np.sum(labels > 1000, axis=1) > 0))
    print("columns > 10000:", np.sum(np.sum(labels > 10_000, axis=1) > 0))
    print(80*"=")

if __name__ == "__main__":

    filename = "FOX_train.h5"
    main(filename)

    filename = "MA0035_4_subst_train.h5"
    main(filename)

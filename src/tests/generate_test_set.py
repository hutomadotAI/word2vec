import os
import pickle

from pathlib import Path


SCRIPT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
DATASET_DIR = SCRIPT_PATH.parent / "datasets"

TEST_SIZE = 100


def main():
    datafile = DATASET_DIR / "glove.840B.300d.pkl"
    print("Loading file: {}".format(datafile))
    with datafile.open("rb") as filehandle:
        data = pickle.load(filehandle)
        number_items = len(data)
        test_items_skip = int(number_items / TEST_SIZE)

        count = 0
        test_data = {}
        print("Found {} items".format(number_items))
        for key, item_data in data.items():
            if count % test_items_skip == 0:
                print("*** Taking key {}".format(key))
                test_data[key] = item_data
            else:
                pass
            # print(count % test_items_skip)
            count += 1

    testfile = SCRIPT_PATH / "data_test_embedding.pkl"
    print("Writing file: {}".format(testfile))
    with testfile.open("wb") as filehandle:
        pickle.dump(test_data, filehandle)


if __name__ == "__main__":
    main()

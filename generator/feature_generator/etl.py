import os

import h5py
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from tqdm import tqdm


class ETL:
    def __init__(self, config, nlp_processor):
        """
        :param config:
        :param nlp_processor:
        """
        self.__dict__ = config
        self.nlp = nlp_processor
        self.vec_dim = 512
        self.tfhub = hub.load(self.google_vec_url)
        print("### TFhub Loaded ...")

    @staticmethod
    def write_nparray(path, arr):
        """
        :param path:
        :param arr:
        :return:
        """
        with h5py.File(path, "w") as hf:
            hf.create_dataset("d", data=arr)
        hf.close()

    @staticmethod
    def read_nparray(path):
        """
        :param path:
        :return:
        """
        arr = None
        with h5py.File(path, "r") as hf:
            arr = hf["d"][:]
        hf.close()
        return arr

    def data_and_xydict(self, df_file_paths: list, max_len):
        """
        :param df_file_paths:
        :param max_len:
        :return:
        """

        label_dict = {"I-CIT": 0, "B-CIT": 1}

        train_x0_path = os.path.join(self.output_dir, "train_x0.h5")
        train_y_path = os.path.join(self.output_dir, "train_y.h5")
        test_x0_path = os.path.join(self.output_dir, "test_x0.h5")
        test_y_path = os.path.join(self.output_dir, "test_y.h5")

        if (
                os.path.exists(train_x0_path)
                and os.path.exists(train_y_path)
                and os.path.exists(test_x0_path)
                and os.path.exists(test_y_path)
        ):

            train_x0 = ETL.read_nparray(train_x0_path)
            train_y = ETL.read_nparray(train_y_path)
            test_x0 = ETL.read_nparray(test_x0_path)
            test_y = ETL.read_nparray(test_y_path)

            return (
                {
                    "train": {"x0": train_x0, "y": train_y, },
                    "test": {"x0": test_x0, "y": test_y, },
                },
                label_dict,
            )
        else:

            all_labs = ["I-CIT", "B-CIT"]
            all_labs = list(all_labs)
            all_labs = {x: i for i, x in enumerate(all_labs)}
            dummy_x = [0.0] * self.vec_dim
            dummy_x = np.array(dummy_x)
            dummy_y = [0]
            dummy_y = np.array(dummy_y)

            train_x0 = []
            train_y = []

            test_x0 = []
            test_y = []

            count = 0

            print("### Data Processing Started ...")

            for file_path in df_file_paths:
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path, index_col=0)
                    cit_sec = df.x.tolist()
                    cit_lab = df.y.tolist()
                    cit_lab = [
                        [1] if x.startswith("B") else [0] for x in cit_lab
                    ]
                    tfhub_x = [np.array(x) for x in self.tfhub(cit_sec)]
                    if count >= self.train_count:
                        test_x0.append(tfhub_x)
                        test_y.append(cit_lab)
                    else:
                        train_x0.append(tfhub_x)
                        train_y.append(cit_lab)
                    count = count + 1

            print(f"### Post Padding with MaxSeqLen-> {max_len}")

            for i in tqdm(range(len(train_x0))):
                x0 = train_x0[i]
                y = train_y[i]
                rem = max_len - len(x0)
                for k in range(rem):
                    x0 = np.append(x0, [dummy_x], axis=0)
                    y = np.append(y, [dummy_y], axis=0)
                train_x0[i] = x0
                train_y[i] = y

            print("### Writing Data (Test) ...")

            train_x0 = np.array(train_x0)
            train_y = np.array(train_y)

            ETL.write_nparray(train_x0_path, train_x0)
            ETL.write_nparray(train_y_path, train_y)

            for i in tqdm(range(len(test_x0))):
                x0 = test_x0[i]
                y = test_y[i]
                rem = max_len - len(x0)
                for k in range(rem):
                    x0 = np.append(x0, [dummy_x], axis=0)
                    y = np.append(y, [dummy_y], axis=0)
                test_x0[i] = x0
                test_y[i] = y

            test_x0 = np.array(test_x0)
            test_y = np.array(test_y)

            ETL.write_nparray(test_x0_path, test_x0)
            ETL.write_nparray(test_y_path, test_y)

            return (
                {
                    "train": {"x0": train_x0, "y": train_y},
                    "test": {"x0": test_x0, "y": test_y},
                    "labels": all_labs,
                },
                all_labs,
            )

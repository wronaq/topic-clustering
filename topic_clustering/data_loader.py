from pathlib import Path


class DataLoader:
    def __init__(self, path):
        self.path = path

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[1] / "data/preprocessed"

    def load_data(self):
        path_to_data = DataLoader.data_dirname() / self.path
        with open(path_to_data, "r") as f:
            lines = f.readlines()
            sentences = []
            for line in lines:
                sentences.append(line.rstrip("\n"))
        return sentences

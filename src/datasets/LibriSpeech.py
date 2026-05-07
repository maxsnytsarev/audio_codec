from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class LibriSpeechDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(self, name="train", *args, **kwargs):
        index_path = ROOT_PATH / "data" / "LibriSpeech" / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        data_path = ROOT_PATH / "data" / "LibriSpeech" / name
        if not data_path.exists():
            raise FileNotFoundError(f"no data: {data_path}")
        data_path.mkdir(exist_ok=True, parents=True)

        big_dirs = [p for p in data_path.iterdir() if p.is_dir()]
        for i in tqdm(range(len(big_dirs))):
            cur_big_dir = big_dirs[i]
            cur_small_dirs = [p for p in cur_big_dir.iterdir() if p.is_dir()]
            for j in tqdm(range(len(cur_small_dirs))):
                cur_small_dir = cur_small_dirs[j]
                files = [f.name for f in cur_small_dir.iterdir() if f.is_file()]
                for file in files:
                    file_type = Path(file).suffix
                    if file_type == ".flac":
                        info = torchaudio.info(str(cur_small_dir / file))
                        sr = info.sample_rate
                        length = info.num_frames / info.sample_rate
                        assert sr == 16000
                        index.append(
                            {
                                "path": str(cur_small_dir / file),
                                "length": length,
                            }
                        )
        write_json(index, str(data_path / "index.json"))
        return index

    def load_object(self, path):
        data_object, sr = torchaudio.load(str(path))
        assert sr == 16000
        assert data_object.shape[0] == 1
        return data_object

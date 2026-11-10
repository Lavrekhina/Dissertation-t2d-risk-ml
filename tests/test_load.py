import pandas as pd
from t2d.load import find_file


def test_find_file_handles_txt_suffix(tmp_path):
    folder = tmp_path / "2017-2020"
    folder.mkdir()
    (folder / "P_GHB.xpt.txt").write_bytes(b"x")
    assert find_file("2017-2020", "GHB", raw=tmp_path).name == "P_GHB.xpt.txt"

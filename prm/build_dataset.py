import pathlib

import pandas as pd


def parse_downsampled_prm_public(data_path: pathlib.Path):
    assert data_path.exists()
    df = pd.read_csv(data_path, sep="\t", header=None)
    # PRM public data is written by string.
    # For example, list: [1, 2, 3] is written to "[1, 2, 3]". so, we need to parse it to list.
    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: eval(x))
    columns = ["uid", "user_profile", "item_categorical", "item_dense", "label"]
    df.columns = columns

    return df


def build_downsampled_prm_public(data_dir: pathlib.Path, save_dir: pathlib.Path):
    assert data_dir.exists()
    assert save_dir.exists()

    for fname in ["data.train", "data.test", "data.valid"]:
        print("=" * 30)
        print(f"building {fname}")
        print("parsing...")
        df = parse_downsampled_prm_public(data_dir / fname)
        print("saving...")
        df.to_csv(save_dir / f"{fname}.csv", index=False)


if __name__ == "__main__":
    data_dir = pathlib.Path("~/Documents/data/prm-public/downsampled/").expanduser().resolve()
    save_dir = pathlib.Path("./data/prm-public/downsampled").resolve()
    for path in [*save_dir.parents[::-1], save_dir]:
        if not path.exists():
            path.mkdir()

    build_downsampled_prm_public(data_dir, save_dir)

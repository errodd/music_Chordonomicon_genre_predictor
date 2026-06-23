
import pandas as pd


def split(filename, chunksize):
    with pd.read_csv(filename, chunksize=chunksize) as reader:
        for i,chunk in enumerate(reader):
            chunk.to_csv(f'chordonomicon_v2_part_{i}.csv', index=False, header=True)


if __name__ == "__main__":
    split("chordonomicon_v2.csv", 100000)

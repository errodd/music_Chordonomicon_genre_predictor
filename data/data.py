""" Tried to upload the .csv file but github has a file size limit of 100MB, so i imported the csv from pandas instead - zajrj """
import pandas as pd

df = pd.read_csv("hf://datasets/ailsntua/Chordonomicon/chordonomicon_v2.csv")
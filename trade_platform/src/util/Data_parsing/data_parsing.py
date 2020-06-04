import pandas
'''
Takes csv in format ticker, per, date, time, open, high, low, close, volume
returns dataframe of just open, high, low, close, volume being optional
'''
def parse(csv, volume = False):
    df = pandas.read_csv(csv)
    if not volume:
        return df[["<OPEN>", "<HIGH>", "<LOW>","<CLOSE>"]]
    else:
        return df[["<OPEN>", "<HIGH>", "<LOW>","<CLOSE>", "<VOL>"]]

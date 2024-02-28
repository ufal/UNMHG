import pandas as pd
import csv
import langid

news = pd.read_csv("PENS/news.tsv", sep="\t")
news.fillna(value="", inplace=True)

news = news[
    (
        news["News body"].apply(
            lambda val: len(val.split(" ")) > 1 and langid.classify(val)[0] == "en"
        )
    )
    & (
        news["Headline"].apply(
            lambda val: len(val.split(" ")) > 1
            and len(val.split(" ")) <= 25
            and langid.classify(val)[0] == "en"
        )
    )
]
news = news.reset_index(drop=True)
news = news[["News ID", "Headline", "News body"]]
news.columns = ["ID", "Title", "Article"]

# Remove the information from footnote
news["Article"] = news["Article"].apply(lambda val: val.split("___")[0])
news["Article"] = news["Article"].apply(lambda val: " ".join(val.split()))
news["Title"] = news["Title"].apply(lambda val: " ".join(val.split()))

# De-duplicate the data
title_hash = []
article_hash = []
_filtered = []

for _iloc, row in news.iterrows():
    hash_t = hash(row["Title"])
    hash_a = hash(row["Article"])
    if hash_t in title_hash or hash_a in article_hash:
        continue
    title_hash.append(hash_t)
    article_hash.append(hash_a)
    _filtered.append(_iloc)
news = news.iloc[_filtered]

news = news.sample(frac=1, random_state=42).reset_index(drop=True)

test = news.iloc[:5000]
valid = news.iloc[5000:10000]
train = news.iloc[10000:]

test.to_csv("PENS/pens_test.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)

train.to_csv("PENS/pens_train.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)

valid.to_csv("PENS/pens_dev.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)

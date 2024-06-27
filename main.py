import pandas

dataframe = pandas.read_csv("data/features_3_sec.csv")
feature_cols = dataframe.columns[2:-1].tolist()

# This is mostly to understand output better
class_mapping = {i: label for i, label in enumerate(dataframe["label"].unique())}

# Let's start the training
from fastai.tabular.all import *
from sklearn.model_selection import train_test_split

train_dataframe, test_dataframe = train_test_split(
    dataframe, test_size=0.2, random_state=42, stratify=dataframe["label"]
)

data_loader = TabularDataLoaders.from_df(
    train_dataframe,
    path=Path("data"),
    y_names="label",
    valid_dataframe=test_dataframe,
    cont_names=feature_cols,
    procs=[Categorify, FillMissing, Normalize],
)

learn = tabular_learner(data_loader, metrics=accuracy)
learn.fit_one_cycle(200, cbs=[EarlyStoppingCallback(monitor="valid_loss", patience=10)])

from sklearn.metrics import classification_report

print(f"Classes: {class_mapping}")

predictions, targets = learn.get_preds()
print("Classification Report:")
print(classification_report(targets, predictions.argmax(dim=1)))

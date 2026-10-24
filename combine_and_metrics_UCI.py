import pandas as pd


files = ["Results/DMV_adult.csv", "Results/DMV_elnino.csv", "Results/DMV_tao.csv", 
         "Results/DMV_pima-indians-diabetes.csv"]


datasets = [pd.read_csv(file, keep_default_na=False) for file in files]


groundtruth = pd.read_csv("Data/DataWithGroundTruth/GT/GroundTruthUCI.txt",
                          sep="::",
                          header=None,
                          names=["Table Name", "Attribute Name", "DMV"],
                          keep_default_na=False,
                          engine="python")

results = pd.concat(datasets, ignore_index=True)


columns = ["Table Name", "Attribute Name", "DMV"]

for df in [groundtruth, results]:
    for col in columns:
        df[col] = df[col].astype(str).str.strip()

actual = set(map(tuple, groundtruth[columns].values))
predicted = set(map(tuple, results[columns].values))

TP = len(actual & predicted)
FP = len(predicted - actual)
FN = len(actual - predicted)

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

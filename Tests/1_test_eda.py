import pandas as pd
from NLPUtilsBERT.Utils_NLP_EDA import TextEDA

# Load dataset
dataset_path = r"data/bbc-test.csv"
df = pd.read_csv(dataset_path)

# Perform EDA
eda = TextEDA(dataframe=df,
              text_column="text",
              label_column="category",
              eda_folder="EDA",
              show_plots=False)
eda.perform_eda()
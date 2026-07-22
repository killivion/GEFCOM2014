from src.data.loader import load_all_tasks

result = load_all_tasks("data/raw/load", n_tasks=15)
print(result.df.head())
print(result.df.dtypes)
print("temp columns found:", result.temp_cols)
print("rows per task:", result.df["task"].value_counts().sort_index())
from src.data.loader import get_data

result = get_data("src/data/raw/Load", n_tasks=15)
print(result.df.head())
print(result.df.dtypes)
print("temp columns found:", result.temp_cols)
print("rows per task:", result.df["task"].value_counts().sort_index())
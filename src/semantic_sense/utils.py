import pandas as pd

def row_to_text(row) -> str:
    # Handle pandas Series (apply row-wise)
    if isinstance(row, pd.Series):
        items = row.items()
    # Handle namedtuple from itertuples
    elif hasattr(row, "_asdict"):
        items = row._asdict().items()
    else:
        raise TypeError(f"Unsupported row type: {type(row)}")

    # Format into "Column: value" string
    return ", ".join(f"{k}: {v}" for k, v in items)


def split_numeric_text(df: pd.DataFrame):
    """Separate numeric and non-numeric columns."""
    num_df = df.select_dtypes(include=["number"])
    text_df = df.drop(columns=num_df.columns)
    return num_df, text_df

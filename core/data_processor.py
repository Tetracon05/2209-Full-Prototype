"""
data_processor.py — Handles CSV loading, cleaning, feature engineering,
correlation analysis, and train/validation/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Default feature columns expected in the dataset
DEFAULT_FEATURES = [
    "Wind_Speed",
    "Weather_Temperature_Celsius",
    "Global_Horizontal_Radiation",
    "Air_Pressure",
    "Pyranometer_1",
    "Temperature_Probe_1",
    "Temperature_Probe_2",
]
TARGET_COL = "Active_Power"
TIMESTAMP_COL = "timestamp"


class DataProcessor:
    """
    Encapsulates all data management steps:
      1. load_csv       — read CSV, parse timestamps
      2. clean          — drop/fill missing values
      3. compute_correlation — rank features by correlation to target
      4. add_lag_features   — create time-lagged copies of selected columns
      5. add_imf_features   — append IMF components from decomposition
      6. split              — stratified 70/15/15 chronological split
      7. get_scaled_splits  — MinMaxScaler fit on train, transform all sets
    """

    def __init__(self):
        self.df_raw: pd.DataFrame = None          # original loaded data
        self.df: pd.DataFrame = None              # working dataframe
        self.feature_cols: list = []              # currently selected features
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Split arrays (numpy)
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

        # Inverse-scale helpers
        self._y_min = 0.0
        self._y_max = 1.0

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    def load_csv(self, path: str) -> dict:
        """
        Load a CSV file.  Parses the 'timestamp' column if present.

        Returns a summary dict for display in the UI.
        """
        df = pd.read_csv(path)

        # Parse timestamp column
        if TIMESTAMP_COL in df.columns:
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
            df.sort_values(TIMESTAMP_COL, inplace=True)
            df.reset_index(drop=True, inplace=True)

        self.df_raw = df.copy()
        self.df = df.copy()

        return {
            "rows": len(df),
            "columns": list(df.columns),
            "missing": int(df.isnull().sum().sum()),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

    # ------------------------------------------------------------------
    # 2. Clean
    # ------------------------------------------------------------------
    def clean(self, strategy: str = "drop") -> int:
        """
        Handle missing values.

        Parameters
        ----------
        strategy : 'drop'  — remove rows with any NaN
                   'mean'  — fill numeric NaN with column mean
                   'ffill' — forward fill

        Returns number of rows after cleaning.
        """
        if self.df_raw is None:
            raise ValueError("No data loaded.")
        df = self.df_raw.copy()
        if strategy == "drop":
            df.dropna(inplace=True)
        elif strategy == "mean":
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == "ffill":
            df.ffill(inplace=True)
            df.bfill(inplace=True)
        df.reset_index(drop=True, inplace=True)
        self.df = df
        return len(df)

    # ------------------------------------------------------------------
    # 3. Correlation
    # ------------------------------------------------------------------
    def compute_correlation(self, top_n: int = 7) -> pd.Series:
        """
        Compute absolute Pearson correlation of numeric features with TARGET_COL.
        Returns a Series sorted descending, limited to top_n features.
        """
        numeric = self.df.select_dtypes(include=[np.number])
        if TARGET_COL not in numeric.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found.")
        corr = numeric.corr()[TARGET_COL].drop(TARGET_COL, errors="ignore")
        corr_abs = corr.abs().sort_values(ascending=False)
        return corr_abs.head(top_n)

    # ------------------------------------------------------------------
    # 4. Lag features
    # ------------------------------------------------------------------
    def add_lag_features(self, columns: list, lags: list = [1, 2, 3]) -> list:
        """
        Append time-lagged versions of the given columns to self.df.
        Returns the list of newly added column names.
        """
        new_cols = []
        for col in columns:
            if col not in self.df.columns:
                continue
            for lag in lags:
                new_name = f"{col}_lag{lag}"
                self.df[new_name] = self.df[col].shift(lag)
                new_cols.append(new_name)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return new_cols

    # ------------------------------------------------------------------
    # 5. IMF (decomposition output) features
    # ------------------------------------------------------------------
    def add_imf_features(self, imfs: np.ndarray, prefix: str = "IMF") -> list:
        """
        Append IMF component arrays as new columns.
        imfs shape: (n_imfs, n_samples) — must match len(self.df).

        Returns list of new column names.
        """
        new_cols = []
        n_samples = len(self.df)
        for i, imf in enumerate(imfs):
            col_name = f"{prefix}_{i + 1}"
            arr = imf[:n_samples]
            if len(arr) < n_samples:
                arr = np.pad(arr, (0, n_samples - len(arr)), mode="edge")
            self.df[col_name] = arr
            new_cols.append(col_name)
        return new_cols

    # ------------------------------------------------------------------
    # 5.5 Circshift Augmentation
    # ------------------------------------------------------------------
    def add_circshift_augmentation(self, shift_steps: int):
        """
        Doubles the dataset by circularly shifting all rows by `shift_steps`
        and appending them to the bottom, effectively acting as Data Augmentation.
        A shift of 0 does nothing.
        """
        if shift_steps <= 0:
            return
        
        df_shifted = self.df.copy()
        for col in df_shifted.columns:
            df_shifted[col] = np.roll(df_shifted[col].values, shift=shift_steps)
            
        self.df = pd.concat([self.df, df_shifted], ignore_index=True)

    # ------------------------------------------------------------------
    # 6. Split
    # ------------------------------------------------------------------
    def split(self, feature_cols: list = None,
              train_ratio: float = 0.70,
              val_ratio: float = 0.15,
              horizon: int = 1) -> dict:
        """
        Chronological (time-ordered) 70/15/15 split.

        Parameters
        ----------
        feature_cols : columns to use as X; defaults to DEFAULT_FEATURES ∩ df.columns
        train_ratio  : fraction for training set
        val_ratio    : fraction for validation set (remainder = test)
        horizon      : Steps ahead to predict. 0 = current step, 1 = next step, etc.

        Returns dict with split sizes.
        """
        if feature_cols is None:
            feature_cols = [c for c in DEFAULT_FEATURES if c in self.df.columns]

        # Also include any IMF / lag columns that were added
        extra = [c for c in self.df.columns
                 if (c.startswith("IMF_") or "_lag" in c) and c in self.df.columns]
        feature_cols = list(dict.fromkeys(feature_cols + extra))  # unique, ordered

        self.feature_cols = [c for c in feature_cols if c in self.df.columns]

        df_work = self.df.copy()
        target_name = TARGET_COL
        
        # Shift target backwards by horizon steps for forecasting
        if horizon > 0:
            target_name = f"{TARGET_COL}_horizon_{horizon}"
            df_work[target_name] = df_work[TARGET_COL].shift(-horizon)
            df_work.dropna(subset=[target_name], inplace=True)

        X = df_work[self.feature_cols].values.astype(np.float32)
        y = df_work[target_name].values.astype(np.float32).reshape(-1, 1)

        n = len(X)
        i_train = int(n * train_ratio)
        i_val   = int(n * (train_ratio + val_ratio))

        self.X_train, self.y_train = X[:i_train],        y[:i_train]
        self.X_val,   self.y_val   = X[i_train:i_val],   y[i_train:i_val]
        self.X_test,  self.y_test  = X[i_val:],          y[i_val:]

        return {
            "train": len(self.X_train),
            "val":   len(self.X_val),
            "test":  len(self.X_test),
            "features": self.feature_cols,
        }

    # ------------------------------------------------------------------
    # 7. Scale
    # ------------------------------------------------------------------
    def get_scaled_splits(self):
        """
        Fit MinMaxScaler on training data only, then transform all splits.

        Returns (X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s)
        """
        X_train_s = self.scaler_X.fit_transform(self.X_train)
        X_val_s   = self.scaler_X.transform(self.X_val)
        X_test_s  = self.scaler_X.transform(self.X_test)

        y_train_s = self.scaler_y.fit_transform(self.y_train)
        y_val_s   = self.scaler_y.transform(self.y_val)
        y_test_s  = self.scaler_y.transform(self.y_test)

        return X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s

    def inverse_scale_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform scaled target predictions back to original units."""
        return self.scaler_y.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).ravel()

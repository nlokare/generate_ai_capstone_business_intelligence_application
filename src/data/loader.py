"""
Data loading and preprocessing module.
Handles CSV file loading with robust type handling for Streamlit/PyArrow compatibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime


class DataLoader:
    """Loads and preprocesses business data from CSV files with robust type handling."""

    def __init__(self):
        """Initialize data loader."""
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV with aggressive type normalization for PyArrow compatibility.

        Args:
            file_path: Path to CSV file

        Returns:
            Loaded and normalized DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Step 1: Read CSV with all columns as strings initially
        # This prevents pandas from creating mixed-type object columns
        df = pd.read_csv(file_path, dtype=str, na_values=['', 'NA', 'N/A', 'null', 'NULL'])

        if df.empty:
            raise ValueError("CSV file is empty")

        # Step 2: Normalize all data - convert types one column at a time
        df = self._normalize_dataframe_types(df)

        # Step 3: Store metadata
        self.df = df
        self._extract_metadata()

        return self.df

    def _normalize_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame columns to appropriate types, ensuring PyArrow compatibility.

        Strategy:
        1. Detect date columns and convert to datetime, then to ISO string format
        2. Detect numeric columns and convert to float64
        3. Keep everything else as clean strings

        This ensures no mixed types exist in any column.
        """
        df_normalized = df.copy()

        for col in df_normalized.columns:
            df_normalized[col] = self._normalize_column(df_normalized[col], col)

        return df_normalized

    def _normalize_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """
        Normalize a single column to a consistent type.

        Args:
            series: Column data
            col_name: Column name for context

        Returns:
            Normalized series with consistent type
        """
        # Skip if all values are NaN
        if series.isna().all():
            return series.fillna('')

        # Strategy 1: Try to parse as dates (for columns with 'date' in name or date-like values)
        if 'date' in col_name.lower() or self._looks_like_date_column(series):
            try:
                # Convert to datetime
                dt_series = pd.to_datetime(series, errors='coerce')

                # If most values converted successfully, use date format
                if dt_series.notna().sum() / len(dt_series) > 0.5:
                    # Convert to ISO string format to avoid PyArrow Timestamp issues
                    # This creates clean string dates that display nicely
                    return dt_series.dt.strftime('%Y-%m-%d').fillna('')
            except:
                pass

        # Strategy 2: Try to parse as numeric
        if self._looks_like_numeric_column(series):
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')

                # If most values converted successfully, use numeric
                if numeric_series.notna().sum() / len(numeric_series) > 0.8:
                    return numeric_series.fillna(0)
            except:
                pass

        # Strategy 3: Keep as clean string (default)
        # Force everything to string and fill NaN with empty string
        return series.astype(str).replace('nan', '').replace('None', '').fillna('')

    def _looks_like_date_column(self, series: pd.Series) -> bool:
        """Check if a column looks like it contains dates."""
        # Sample first few non-null values
        sample = series.dropna().head(10)

        if len(sample) == 0:
            return False

        # Common date patterns
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'^\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]

        # Check if values match date patterns
        matches = 0
        for val in sample:
            val_str = str(val)
            for pattern in date_patterns:
                import re
                if re.match(pattern, val_str):
                    matches += 1
                    break

        return matches / len(sample) > 0.5

    def _looks_like_numeric_column(self, series: pd.Series) -> bool:
        """Check if a column looks like it contains numbers."""
        # Sample first few non-null values
        sample = series.dropna().head(20)

        if len(sample) == 0:
            return False

        # Try to convert to numeric
        try:
            numeric = pd.to_numeric(sample, errors='coerce')
            # If more than 80% convert successfully, it's numeric
            return numeric.notna().sum() / len(sample) > 0.8
        except:
            return False

    def _extract_metadata(self):
        """Generate metadata about the loaded data."""
        if self.df is None:
            return

        self.metadata = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
        }

        # Identify column types for later use
        self.metadata['date_columns'] = [
            col for col in self.df.columns
            if 'date' in col.lower()
        ]
        self.metadata['numeric_columns'] = list(
            self.df.select_dtypes(include=[np.number]).columns
        )
        self.metadata['categorical_columns'] = [
            col for col in self.df.columns
            if col not in self.metadata['numeric_columns'] and col not in self.metadata['date_columns']
        ]

    def validate_data(self) -> List[str]:
        """
        Validate data quality.

        Returns:
            List of validation issues (empty if no issues)
        """
        if self.df is None:
            return ["No data loaded"]

        issues = []

        # Check for completely empty columns
        empty_cols = [col for col in self.df.columns if self.df[col].isna().all()]
        if empty_cols:
            issues.append(f"Empty columns: {', '.join(empty_cols)}")

        # Check for high null percentage
        null_pct = (self.df.isnull().sum() / len(self.df)) * 100
        high_null_cols = null_pct[null_pct > 50].index.tolist()
        if high_null_cols:
            issues.append(f"Columns with >50% missing values: {', '.join(high_null_cols)}")

        # Check for duplicate rows
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")

        return issues

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about loaded data.

        Returns:
            Metadata dictionary
        """
        return self.metadata

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the loaded DataFrame."""
        return self.df

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the data.

        Returns:
            Dictionary with summary statistics
        """
        if self.df is None:
            return {}

        stats = {}

        # Numeric columns statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric'] = self.df[numeric_cols].describe().to_dict()

        # Date range
        date_cols = self.metadata.get('date_columns', [])
        if date_cols:
            for col in date_cols:
                # Try to parse dates for statistics
                try:
                    dates = pd.to_datetime(self.df[col], errors='coerce')
                    valid_dates = dates.dropna()
                    if len(valid_dates) > 0:
                        stats[f'{col}_range'] = {
                            'earliest': str(valid_dates.min()),
                            'latest': str(valid_dates.max()),
                            'span_days': (valid_dates.max() - valid_dates.min()).days
                        }
                except:
                    pass

        # Categorical statistics
        text_cols = self.metadata.get('categorical_columns', [])
        for col in text_cols[:5]:  # Limit to first 5 text columns
            value_counts = self.df[col].value_counts()
            if len(value_counts) > 0 and len(value_counts) < 100:  # Only for reasonable cardinality
                stats[f'{col}_distribution'] = value_counts.head(10).to_dict()

        return stats


def load_business_data(file_path: str) -> pd.DataFrame:
    """
    Convenience function to load business data.

    Args:
        file_path: Path to the CSV file

    Returns:
        Loaded DataFrame
    """
    loader = DataLoader()
    return loader.load_data(file_path)

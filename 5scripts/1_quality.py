"""
Data Quality Analysis - Stage 1

Detects 7 issue types: sentinels, negatives, OHLC violations, missing data,
duplicates, stale prices, volume-price anomalies.

Outputs:
    4output/data_quality_flags.csv - per-row issue flags
    4output/data_quality_summary.json - comprehensive statistics
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd, numpy as np


# Private helper functions

def _check_sentinel_values(df: pd.DataFrame) -> pd.Series:
    # Detect placeholder values: 1e-08, 1e+20, -999, -9999, etc.
    assert len(df) > 0, f"empty dataframe"
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Open Interest']

    sentinel_mask = pd.Series(False, index=df.index)

    for col in numeric_cols:
        if col in df.columns:
            # Trading data never has sub-micro values - these are placeholders
            sentinel_mask |= (df[col] > 0) & (df[col] < 1e-6)

            # Values > 1 billion are unrealistic for futures prices
            sentinel_mask |= (df[col].abs() > 1e9)

            # Common placeholder patterns (-999, -9999, 999999)
            sentinel_mask |= df[col].isin([-999, -9999, 999999, 1e-08, 1e-10, 1e-20])

            # Large round numbers are clearly placeholders
            sentinel_mask |= (df[col].abs() >= 1e10) & ((df[col] % 1e10) == 0)

    return sentinel_mask


def _check_negative_values(df: pd.DataFrame) -> pd.Series:
    # Prices and volumes cannot be negative
    price_cols = ['Open', 'High', 'Low', 'Close']
    volume_cols = ['Volume', 'Open Interest']

    negative_mask = pd.Series(False, index=df.index)

    for col in price_cols + volume_cols:
        if col in df.columns:
            negative_mask |= df[col] < 0

    return negative_mask


def _check_ohlc_violations(df: pd.DataFrame) -> pd.DataFrame:
    # OHLC must satisfy: Low <= Open,Close <= High and Low <= High
    # Also check edge cases: zero prices, NaN, all equal
    violations = pd.DataFrame(index=df.index)

    # Check for valid non-zero prices before relationship checks
    has_valid_prices = (df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)

    # Standard OHLC violations (only check where prices are valid and non-null)
    violations['high_lt_low'] = has_valid_prices & (df['High'] < df['Low'])
    violations['open_outside_range'] = has_valid_prices & ((df['Open'] < df['Low']) | (df['Open'] > df['High']))
    violations['close_outside_range'] = has_valid_prices & ((df['Close'] < df['Low']) | (df['Close'] > df['High']))
    violations['any_ohlc_violation'] = violations.any(axis=1)

    return violations


def _check_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    # Check for missing/null values in each field
    missing = pd.DataFrame(index=df.index)

    for col in df.columns:
        missing[f'{col}_missing'] = df[col].isna()

    missing['any_missing'] = missing.any(axis=1)

    return missing


def _check_duplicates(df: pd.DataFrame) -> pd.Series:
    # Duplicate (Symbol, Date) combinations indicate data quality issues
    if 'Date' in df.columns:
        duplicate_mask = df.duplicated(subset=['Symbol', 'Date'], keep=False)
    elif 'Timestamp' in df.columns:
        duplicate_mask = df.duplicated(subset=['Symbol', 'Timestamp'], keep=False)
    else:
        duplicate_mask = pd.Series(False, index=df.index)

    return duplicate_mask


def _check_stale_prices(df: pd.DataFrame) -> pd.Series:
    # Consecutive identical OHLC values suggest stale/copied data
    ohlc_cols = ['Open', 'High', 'Low', 'Close']

    # Sort by symbol and date/timestamp
    if 'Date' in df.columns:
        df_sorted = df.sort_values(['Symbol', 'Date']).copy()
    else:
        df_sorted = df.sort_values(['Symbol', 'Timestamp']).copy()

    stale_mask = pd.Series(False, index=df_sorted.index)

    for symbol in df_sorted['Symbol'].unique():
        symbol_mask = df_sorted['Symbol'] == symbol
        symbol_data = df_sorted.loc[symbol_mask, ohlc_cols]

        # Compare with previous row
        is_same = (symbol_data == symbol_data.shift()).all(axis=1)
        # First row per symbol cannot be stale (no previous row to compare)
        is_same.iloc[0] = False
        stale_mask.loc[symbol_mask] = is_same

    # Reindex to original order
    stale_mask = stale_mask.reindex(df.index, fill_value=False)

    return stale_mask


def _check_volume_price_anomalies(df: pd.DataFrame) -> pd.Series:
    # Volume-price relationships should be consistent
    # Note: H==L with volume can be legitimate for futures (limit moves, thin trading)
    # - Price movement with zero volume is suspicious
    anomaly_mask = pd.Series(False, index=df.index)

    has_movement = (df['High'] != df['Low']) & (df['Volume'] == 0)
    anomaly_mask |= has_movement

    return anomaly_mask


def _check_statistical_outliers(df: pd.DataFrame) -> pd.Series:
    # Detect extreme statistical outliers using Z-score method
    # - Extreme price moves (>10% daily)
    # - Volume spikes (>10x rolling average per symbol)
    outlier_mask = pd.Series(False, index=df.index)

    # Sort by symbol and date for rolling calculations
    if 'Date' in df.columns:
        df_sorted = df.sort_values(['Symbol', 'Date']).copy()
    else:
        df_sorted = df.sort_values(['Symbol', 'Timestamp']).copy()

    for symbol in df_sorted['Symbol'].unique():
        symbol_mask = df_sorted['Symbol'] == symbol
        symbol_data = df_sorted.loc[symbol_mask].copy()

        # Price move outliers (>10% daily change)
        if len(symbol_data) > 1:
            returns = symbol_data['Close'].pct_change(fill_method=None).abs()
            extreme_moves = returns > 0.10
            outlier_mask.loc[symbol_data.index] |= extreme_moves

        # Volume spike outliers (>10x rolling 20-day average)
        if len(symbol_data) > 20:
            vol_ma = symbol_data['Volume'].rolling(window=20, min_periods=5).mean()
            vol_ratio = symbol_data['Volume'] / vol_ma
            volume_spikes = (vol_ratio > 10) & (vol_ma > 0)
            outlier_mask.loc[symbol_data.index] |= volume_spikes

    # Reindex to original order
    outlier_mask = outlier_mask.reindex(df.index, fill_value=False)

    return outlier_mask


def _check_volume_jump_outliers(df: pd.DataFrame) -> pd.Series:
    # Detect sudden volume/OI jumps (>10x day-over-day) with OI staying flat
    # Catches: Vol jumps 10x+ while OI changes <2x (data error, not real trading)
    outlier_mask = pd.Series(False, index=df.index)

    # Sort by symbol and date
    if 'Date' in df.columns:
        df_sorted = df.sort_values(['Symbol', 'Date']).copy()
    else:
        df_sorted = df.sort_values(['Symbol', 'Timestamp']).copy()

    for symbol in df_sorted['Symbol'].unique():
        symbol_mask = df_sorted['Symbol'] == symbol
        symbol_data = df_sorted.loc[symbol_mask].copy()

        if len(symbol_data) < 2:
            continue

        # Day-over-day changes
        vol_curr = symbol_data['Volume'].values
        vol_prev = symbol_data['Volume'].shift(1).values
        oi_curr = symbol_data['Open Interest'].values
        oi_prev = symbol_data['Open Interest'].shift(1).values

        for i in range(1, len(symbol_data)):
            # Skip if any values are missing or non-positive
            if any(pd.isna([vol_curr[i], vol_prev[i], oi_curr[i], oi_prev[i]])):
                continue
            if vol_prev[i] <= 0 or oi_prev[i] <= 0 or vol_curr[i] <= 0 or oi_curr[i] <= 0:
                continue

            vol_ratio = vol_curr[i] / vol_prev[i]
            oi_ratio = oi_curr[i] / oi_prev[i]

            # Flag: Volume jumps >10x while OI stays relatively stable (<2x change)
            # This pattern indicates data error (FUT2: 21k→223k vol, 97k→99k OI)
            if vol_ratio > 10.0 and oi_ratio < 2.0:
                idx = symbol_data.index[i]
                outlier_mask.loc[idx] = True

    # Reindex to original order
    outlier_mask = outlier_mask.reindex(df.index, fill_value=False)

    return outlier_mask


def _temporal_analysis(df: pd.DataFrame, flags: pd.DataFrame) -> dict:
    # Analyze error rates over time to detect temporal patterns
    if 'Date' not in df.columns:
        return {}

    df_temp = df.copy()
    df_temp['has_issue'] = flags['has_any_issue']

    # Filter out null dates before period conversion
    df_temp = df_temp[df_temp['Date'].notna()]
    if len(df_temp) == 0:
        return {}

    df_temp['year_month'] = df_temp['Date'].dt.to_period('M')
    df_temp['year_week'] = df_temp['Date'].dt.to_period('W')

    # Error rate by month - single-level column naming for memory efficiency
    monthly_errors = df_temp.groupby('year_month')['has_issue'].agg([
        ('error_count', 'sum'),
        ('total_count', 'count'),
        ('error_rate', 'mean')
    ]).round(4)

    # Error rate by week
    weekly_errors = df_temp.groupby('year_week')['has_issue'].agg([
        ('error_count', 'sum'),
        ('total_count', 'count'),
        ('error_rate', 'mean')
    ]).round(4)

    # Find worst periods
    worst_months = monthly_errors.nlargest(5, 'error_rate')

    # Convert Period index to string for JSON serialization
    monthly_errors.index = monthly_errors.index.astype(str)
    worst_months.index = worst_months.index.astype(str)

    return {
        'monthly_error_rates': monthly_errors.to_dict(),
        'worst_months': worst_months.to_dict(),
        'overall_time_range': {
            'start': df_temp['Date'].min().strftime('%Y-%m-%d'),
            'end': df_temp['Date'].max().strftime('%Y-%m-%d')
        }
    }


def _generate_summary_stats(df: pd.DataFrame, flags: pd.DataFrame) -> dict:
    # Generate comprehensive summary statistics for JSON export
    total_rows = len(df)
    issue_cols = ['sentinel_values', 'negative_values', 'ohlc_violation',
                  'has_missing_data', 'is_duplicate', 'stale_prices', 'volume_price_anomaly',
                  'statistical_outliers', 'volume_jump_outliers']

    # Overall statistics
    summary = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_rows': int(total_rows),
            'total_symbols': int(df['Symbol'].nunique()),
            'symbols': sorted(df['Symbol'].unique().tolist())
        },
        'overall_quality': {
            'clean_rows': int((~flags['has_any_issue']).sum()),
            'clean_percentage': round((~flags['has_any_issue']).sum() / total_rows * 100, 2),
            'rows_with_issues': int(flags['has_any_issue'].sum()),
            'issue_percentage': round(flags['has_any_issue'].sum() / total_rows * 100, 2)
        },
        'issue_breakdown': {}
    }

    # Issue counts
    for col in issue_cols:
        count = int(flags[col].sum())
        pct = round(count / total_rows * 100, 2)
        summary['issue_breakdown'][col] = {
            'count': count,
            'percentage': pct
        }

    # OHLC violation details
    summary['issue_breakdown']['ohlc_violation']['details'] = {
        'high_lt_low': int(flags['high_lt_low'].sum()),
        'open_outside_range': int(flags['open_outside_range'].sum()),
        'close_outside_range': int(flags['close_outside_range'].sum())
    }

    # Per-symbol statistics
    symbol_stats = {}
    for symbol in df['Symbol'].unique():
        symbol_mask = flags['Symbol'] == symbol
        symbol_total = symbol_mask.sum()

        symbol_stats[symbol] = {
            'total_rows': int(symbol_total),
            'rows_with_issues': int(flags.loc[symbol_mask, 'has_any_issue'].sum()),
            'issue_percentage': round(flags.loc[symbol_mask, 'has_any_issue'].sum() / symbol_total * 100, 2),
            'issues': {}
        }

        for col in issue_cols:
            symbol_stats[symbol]['issues'][col] = int(flags.loc[symbol_mask, col].sum())

    summary['per_symbol'] = symbol_stats

    # Temporal analysis
    print("Performing temporal analysis...")
    temporal_stats = _temporal_analysis(df, flags)
    if temporal_stats:
        summary['temporal'] = temporal_stats

    return summary


def _print_summary(df: pd.DataFrame, flags: pd.DataFrame) -> None:
    # Print summary statistics of data quality issues
    total_rows = len(df)

    print("\n" + "="*60)
    print("DATA QUALITY SUMMARY")
    print("="*60)
    print(f"\nTotal rows: {total_rows:,}")
    print(f"Rows with issues: {flags['has_any_issue'].sum():,} ({flags['has_any_issue'].sum()/total_rows*100:.1f}%)")
    print(f"Clean rows: {(~flags['has_any_issue']).sum():,} ({(~flags['has_any_issue']).sum()/total_rows*100:.1f}%)")

    print("\n" + "-"*60)
    print("ISSUE BREAKDOWN")
    print("-"*60)
    print(f"Sentinel/corrupted:    {flags['sentinel_values'].sum():,} ({flags['sentinel_values'].sum()/total_rows*100:.1f}%)")
    print(f"  (1e-08, 1e+20, -10000000000, etc.)")
    print(f"Negative values:       {flags['negative_values'].sum():,} ({flags['negative_values'].sum()/total_rows*100:.1f}%)")
    print(f"OHLC violations:       {flags['ohlc_violation'].sum():,} ({flags['ohlc_violation'].sum()/total_rows*100:.1f}%)")
    print(f"  - High < Low:        {flags['high_lt_low'].sum():,}")
    print(f"  - Open out of range: {flags['open_outside_range'].sum():,}")
    print(f"  - Close out of range: {flags['close_outside_range'].sum():,}")
    print(f"Missing data:          {flags['has_missing_data'].sum():,} ({flags['has_missing_data'].sum()/total_rows*100:.1f}%)")
    print(f"Duplicates:            {flags['is_duplicate'].sum():,} ({flags['is_duplicate'].sum()/total_rows*100:.1f}%)")
    print(f"Stale prices:          {flags['stale_prices'].sum():,} ({flags['stale_prices'].sum()/total_rows*100:.1f}%)")
    print(f"Volume-price anomaly:  {flags['volume_price_anomaly'].sum():,} ({flags['volume_price_anomaly'].sum()/total_rows*100:.1f}%)")
    print(f"Statistical outliers:  {flags['statistical_outliers'].sum():,} ({flags['statistical_outliers'].sum()/total_rows*100:.1f}%)")
    print(f"Volume jump outliers:  {flags['volume_jump_outliers'].sum():,} ({flags['volume_jump_outliers'].sum()/total_rows*100:.1f}%)")
    print(f"  (Volume >10x day-over-day with OI <2x)")

    print("\n" + "-"*60)
    print("ISSUES BY SYMBOL")
    print("-"*60)
    symbol_summary = flags.groupby('Symbol').agg({
        'has_any_issue': 'sum',
        'sentinel_values': 'sum',
        'negative_values': 'sum',
        'ohlc_violation': 'sum',
        'has_missing_data': 'sum',
        'is_duplicate': 'sum',
        'stale_prices': 'sum',
        'volume_price_anomaly': 'sum',
        'statistical_outliers': 'sum',
        'volume_jump_outliers': 'sum'
    })
    symbol_summary.columns = ['Total', 'Sentinel', 'Negative', 'OHLC', 'Missing', 'Dup', 'Stale', 'Vol-Price', 'Outliers', 'Vol-Jump']
    print(symbol_summary.to_string())

    print("\n" + "="*60)


# Public API functions

def load_data(filepath: str) -> pd.DataFrame:
    """Load the futures dataset and perform basic preprocessing."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError as e:
        raise ValueError(f"data file not found: {filepath}") from e

    assert len(df) > 0, f"loaded empty dataframe from {filepath}"
    assert 'Symbol' in df.columns, f"missing required column: Symbol"
    assert 'Timestamp' in df.columns, f"missing required column: Timestamp"

    # Convert timestamp to datetime (Excel serial date format: days since 1899-12-30)
    # Row-by-row conversion for non-null timestamps only
    mask = df['Timestamp'].notna()
    df['Date'] = pd.NaT
    df.loc[mask, 'Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df.loc[mask, 'Timestamp'], unit='D')

    return df


def generate_issue_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a DataFrame with boolean flags for each type of data quality issue.
    """
    assert len(df) > 0, f"empty dataframe"
    flags = pd.DataFrame(index=df.index)

    # Add identifier columns
    flags['Symbol'] = df['Symbol']
    if 'Date' in df.columns:
        flags['Date'] = df['Date']
    flags['Timestamp'] = df['Timestamp']

    # Check each issue type
    print("Checking for sentinel/corrupted values...")
    flags['sentinel_values'] = _check_sentinel_values(df)

    print("Checking for negative values...")
    flags['negative_values'] = _check_negative_values(df)

    print("Checking for OHLC violations...")
    ohlc_violations = _check_ohlc_violations(df)
    flags['ohlc_violation'] = ohlc_violations['any_ohlc_violation']
    flags['high_lt_low'] = ohlc_violations['high_lt_low']
    flags['open_outside_range'] = ohlc_violations['open_outside_range']
    flags['close_outside_range'] = ohlc_violations['close_outside_range']

    print("Checking for missing data...")
    missing = _check_missing_data(df)
    flags['has_missing_data'] = missing['any_missing']

    print("Checking for duplicates...")
    flags['is_duplicate'] = _check_duplicates(df)

    print("Checking for stale prices...")
    flags['stale_prices'] = _check_stale_prices(df)

    print("Checking volume-price anomalies...")
    flags['volume_price_anomaly'] = _check_volume_price_anomalies(df)

    print("Checking statistical outliers...")
    flags['statistical_outliers'] = _check_statistical_outliers(df)

    print("Checking volume jump outliers (>10x with flat OI)...")
    flags['volume_jump_outliers'] = _check_volume_jump_outliers(df)

    # Overall flag: any issue detected
    issue_cols = ['sentinel_values', 'negative_values', 'ohlc_violation',
                  'has_missing_data', 'is_duplicate', 'stale_prices', 'volume_price_anomaly',
                  'statistical_outliers', 'volume_jump_outliers']
    flags['has_any_issue'] = flags[issue_cols].any(axis=1)

    return flags


# Main entry point

def main() -> None:
    """Main execution function."""
    # Configuration
    input_file = "data_eng_assessment_dataset.csv"
    output_flag_file = "4output/data_quality_flags.csv"
    output_summary_file = "4output/data_quality_summary.json"

    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Ensure output directory exists
    Path("4output").mkdir(exist_ok=True)

    print(f"Loading data from {input_file}...")
    df = load_data(input_file)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    print("\nRunning data quality checks...")
    flags = generate_issue_flags(df)

    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summary_stats = _generate_summary_stats(df, flags)

    # Print summary to console
    _print_summary(df, flags)

    # Save outputs
    print(f"\nSaving issue flags to {output_flag_file}...")
    flags.to_csv(output_flag_file, index=False)

    print(f"Saving summary statistics to {output_summary_file}...")
    with open(output_summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)

    print("Done!")
    print(f"\nOutputs:")
    print(f"  - {output_flag_file}")
    print(f"  - {output_summary_file}")


if __name__ == "__main__":
    main()

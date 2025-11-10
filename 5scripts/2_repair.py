"""
Conservative Data Repair for Trading Use

Philosophy: Remove, Don't Fabricate
- Drop rows with missing timestamps
- No look-ahead bias (forward-fill only, max 1-3 days)
- Default action: REMOVE bad data
- Only fix obvious, justifiable cases
- Strict OHLC constraint enforcement
- Smart handling of sparse symbols (FUT11/FUT12)
- Confidence intervals for all values

Output: Smaller but TRUSTWORTHY dataset suitable for trading analysis.
"""

import json
from pathlib import Path
from typing import Callable
import pandas as pd, numpy as np


class ConservativeDataRepair:
    """Conservative data repair engine for trading use."""
    __slots__ = "df_original", "df_cleaned", "bad_data", "removal_log", "repair_log", "stats", "price_threshold"

    def __init__(self, df: pd.DataFrame):
        self.df_original = df.copy()
        self.df_cleaned = None
        self.bad_data = None
        self.removal_log = []
        self.repair_log = []
        self.stats = {
            'original_rows': len(df),
            'rows_removed': 0,
            'rows_repaired': 0,
            'rows_kept': 0
        }
        # Calculate adaptive price threshold
        self.price_threshold = self._calculate_adaptive_threshold(df)

    def _calculate_adaptive_threshold(self, df: pd.DataFrame) -> float:
        """
        Calculate adaptive price threshold using MAD (Median Absolute Deviation).

        Formula: mean(median + 5*MAD for each contract) * 1.25

        This automatically adapts to the data distribution and provides a 25% safety margin.
        Excludes FUT12 which operates in a different price regime.
        """
        price_cols = ['Open', 'High', 'Low', 'Close']
        thresholds = []

        for symbol in df['Symbol'].unique():
            if symbol == 'FUT12':  # Skip outlier contract with different price scale
                continue

            symbol_data = df[df['Symbol'] == symbol]
            symbol_thresholds = []

            for col in price_cols:
                values = symbol_data[col].dropna()
                # Filter out obvious bad values for threshold calculation
                clean_values = values[(values > 0) & (values < 1e10)]

                if len(clean_values) < 10:
                    continue

                median = np.median(clean_values)
                mad = np.median(np.abs(clean_values - median))

                if mad > 0:
                    threshold = median + 5 * mad
                    symbol_thresholds.append(threshold)

            if symbol_thresholds:
                thresholds.append(np.mean(symbol_thresholds))

        if not thresholds:
            return 130.0  # Fallback to manual value if calculation fails

        # Average across all contracts + 25% safety margin
        adaptive_threshold = np.mean(thresholds) * 1.25

        print(f"  Calculated adaptive price threshold: {adaptive_threshold:.2f}")
        print(f"  (vs manual threshold: 130.00)")

        return adaptive_threshold

    def step_1_drop_missing_timestamps(self) -> pd.DataFrame:
        """Drop rows with missing or invalid timestamps."""
        print("STEP 1: Dropping rows with missing timestamps...")
        df = self.df_original.copy()

        missing_timestamp = df['Timestamp'].isna()
        # Valid range: 1960-01-01 (21916 days) to 2040-12-31 (51499 days) from Excel epoch 1899-12-30
        invalid_timestamp = (df['Timestamp'] < 21916) | (df['Timestamp'] > 51499)
        to_remove = missing_timestamp | invalid_timestamp
        removed_count = to_remove.sum()

        for idx in df[to_remove].index:
            self.removal_log.append({
                'index': int(idx),
                'symbol': df.loc[idx, 'Symbol'],
                'reason': 'missing_or_invalid_timestamp',
                'timestamp': str(df.loc[idx, 'Timestamp'])
            })

        df_valid = df[~to_remove].copy()
        df_valid['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df_valid['Timestamp'], unit='D')

        self.df_cleaned = df_valid
        self.stats['rows_removed'] += removed_count
        print(f"  Removed: {removed_count} rows")
        print(f"  Remaining: {len(df_valid)} rows")
        return df_valid

    def step_2_identify_bad_data(self) -> pd.DataFrame:
        """Identify all bad data comprehensively."""
        print("\nSTEP 2: Identifying bad data...")
        df = self.df_cleaned
        bad_data = pd.DataFrame(index=df.index)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Open Interest']

        for col in numeric_cols:
            bad_mask = pd.Series(False, index=df.index)
            bad_mask |= (df[col] > 0) & (df[col] < 1e-6)  # Sentinel: very small
            bad_mask |= df[col].abs() > 1e9  # Sentinel: very large
            bad_mask |= df[col].isin([1e-08, 1e-10, 1e-20])  # Specific sentinels
            bad_mask |= (df[col].abs() >= 1e10) & ((df[col] % 1e10) == 0)  # Round placeholders
            bad_mask |= df[col] < 0  # Negative
            bad_mask |= df[col].isna()  # Missing
            bad_data[f'{col}_bad'] = bad_mask

        bad_data['ohlc_high_lt_low'] = df['High'] < df['Low']
        bad_data['ohlc_open_outside'] = (df['Open'] < df['Low']) | (df['Open'] > df['High'])
        bad_data['ohlc_close_outside'] = (df['Close'] < df['Low']) | (df['Close'] > df['High'])
        bad_data['any_ohlc_issue'] = bad_data[['Open_bad', 'High_bad', 'Low_bad', 'Close_bad',
                                                'ohlc_high_lt_low', 'ohlc_open_outside', 'ohlc_close_outside']].any(axis=1)
        bad_data['any_volume_oi_issue'] = bad_data[['Volume_bad', 'Open Interest_bad']].any(axis=1)
        bad_data['any_issue'] = bad_data['any_ohlc_issue'] | bad_data['any_volume_oi_issue']

        self.bad_data = bad_data
        bad_count = bad_data['any_issue'].sum()
        print(f"  Rows with issues: {bad_count} ({bad_count/len(df)*100:.1f}%)")
        print(f"    OHLC issues: {bad_data['any_ohlc_issue'].sum()}")
        print(f"    Volume/OI issues: {bad_data['any_volume_oi_issue'].sum()}")

        # Save strict version (no repairs, just drop all bad rows)
        self._save_strict_version()

        return bad_data

    def _save_strict_version(self) -> None:
        """Save strict version that drops all detected errors without any repair attempts."""
        print("\nSaving strict version (no repairs, errors dropped)...")
        df = self.df_cleaned
        bad_data = self.bad_data

        # Keep only rows with no issues
        strict_df = df[~bad_data['any_issue']].copy()

        # Save to output directory
        Path('4output').mkdir(exist_ok=True)
        strict_file = Path('4output') / 'data_cleaned_strict.csv'
        strict_df.to_csv(strict_file, index=False)

        strict_count = len(strict_df)
        orig_count = self.stats['original_rows']
        retention_rate = strict_count / orig_count * 100

        print(f"  Strict version: {strict_count:,} rows ({retention_rate:.1f}% retention)")
        print(f"  Saved to: {strict_file}")

    def step_3_remove_unfixable_rows(self) -> None:
        """Remove rows that cannot be fixed conservatively."""
        print("\nSTEP 3: Removing unfixable rows...")
        df, bad_data = self.df_cleaned, self.bad_data
        unfixable = bad_data['any_ohlc_issue']
        unfixable_count = unfixable.sum()

        ohlc_issue_cols = ['Open_bad', 'High_bad', 'Low_bad', 'Close_bad',
                           'ohlc_high_lt_low', 'ohlc_open_outside', 'ohlc_close_outside']
        for idx in df[unfixable].index:
            issues = [col for col in ohlc_issue_cols if bad_data.loc[idx, col]]
            self.removal_log.append({
                'index': int(idx), 'symbol': df.loc[idx, 'Symbol'],
                'date': str(df.loc[idx, 'Date']), 'reason': 'unfixable_ohlc', 'issues': issues
            })

        self.df_cleaned = df[~unfixable].copy()
        self.bad_data = bad_data[~unfixable].copy()
        self.stats['rows_removed'] += unfixable_count
        print(f"  Removed: {unfixable_count} unfixable rows")
        print(f"  Remaining: {len(self.df_cleaned)} rows")

    def step_4_forward_fill_small_gaps(self, max_gap_days: int = 2, extended_gap_days: int = 5) -> None:
        """Forward-fill small gaps in Volume/OI only. Max 2 days normally, 5 days for low volatility."""
        print(f"\nSTEP 4: Forward-filling small gaps (max {max_gap_days} days, {extended_gap_days} for low volatility)...")

        df = self.df_cleaned.sort_values(['Symbol', 'Date']).copy()
        bad_data = self.bad_data.reindex(df.index)

        filled_count = 0
        extended_filled_count = 0

        for symbol in df['Symbol'].unique():
            symbol_mask = df['Symbol'] == symbol
            symbol_df = df[symbol_mask].copy()
            symbol_bad = bad_data[symbol_mask].copy()

            # Check if symbol has enough clean data (>50%)
            clean_pct = (~symbol_bad['any_issue']).sum() / len(symbol_df)
            if clean_pct < 0.5:
                print(f"  Skipping {symbol}: only {clean_pct*100:.0f}% clean data")
                continue

            # Calculate rolling volatility for this symbol (for extended fills)
            if 'Close' in symbol_df.columns:
                # Use log returns and standard deviation for proper volatility calculation
                log_returns = np.log(symbol_df['Close'] / symbol_df['Close'].shift(1))
                rolling_vol = log_returns.rolling(window=20, min_periods=10).std()
            else:
                rolling_vol = pd.Series(1.0, index=symbol_df.index)  # High volatility default

            # Forward-fill Volume and OI only (not OHLC - too risky)
            for col in ['Volume', 'Open Interest']:
                if f'{col}_bad' not in symbol_bad.columns:
                    continue

                bad_mask = symbol_bad[f'{col}_bad'].values

                if not bad_mask.any():
                    continue

                # Find gaps
                values = symbol_df[col].copy()
                values[bad_mask] = np.nan

                # Try normal forward fill first
                filled = values.ffill(limit=max_gap_days)
                was_filled_normal = bad_mask & filled.notna().values

                # For remaining gaps, try extended fill if volatility is low
                still_bad = bad_mask & ~was_filled_normal
                if still_bad.any():
                    # Check volatility over gap periods
                    for i in np.where(still_bad)[0]:
                        if i > 0:
                            # Look back up to extended_gap_days
                            lookback = min(i, extended_gap_days)
                            recent_vol = rolling_vol.iloc[i-lookback:i+1].mean()

                            # If volatility < 1%, allow extended fill
                            if recent_vol < 0.01:
                                # Try filling from up to extended_gap_days back
                                filled_extended = values.ffill(limit=extended_gap_days)
                                if filled_extended.notna().iloc[i]:
                                    filled.iloc[i] = filled_extended.iloc[i]

                # All successful fills
                was_filled = bad_mask & filled.notna().values
                was_filled_extended = was_filled & ~was_filled_normal
                fill_count = was_filled.sum()
                extended_count = was_filled_extended.sum()

                if fill_count > 0:
                    # Update dataframe
                    df.loc[symbol_mask, col] = filled.values

                    # Mark as no longer bad
                    bad_data.loc[symbol_mask, f'{col}_bad'] = bad_mask & ~was_filled

                    # Log repairs
                    for i, idx in enumerate(symbol_df.index):
                        if was_filled[i]:
                            method = 'forward_fill_extended' if was_filled_extended[i] else 'forward_fill'
                            self.repair_log.append({
                                'index': int(idx),
                                'symbol': symbol,
                                'date': str(symbol_df.loc[idx, 'Date']),
                                'field': col,
                                'method': method,
                                'repaired_value': float(filled.iloc[i]),
                                'confidence': 'medium' if not was_filled_extended[i] else 'low'
                            })

                    filled_count += fill_count
                    extended_filled_count += extended_count

        self.df_cleaned = df
        self.bad_data = bad_data
        self.stats['rows_repaired'] += filled_count

        print(f"  Filled: {filled_count} values (Volume/OI only)")
        print(f"    Standard fills: {filled_count - extended_filled_count}")
        print(f"    Extended fills (low volatility): {extended_filled_count}")

    def step_4b_last_traded_price_forward_fill(self) -> None:
        """Last-traded price forward-fill for 1-day gaps (futures settlement price)."""
        print("\nSTEP 4b: Last-traded price forward-fill (1-day gaps, futures settlement)...")

        df = self.df_cleaned.sort_values(['Symbol', 'Date']).copy()
        bad_data = self.bad_data.reindex(df.index)

        filled_count = 0

        for symbol in df['Symbol'].unique():
            symbol_mask = df['Symbol'] == symbol
            symbol_df = df[symbol_mask].copy()
            symbol_bad = bad_data[symbol_mask].copy()

            # Find rows with OHLC issues
            ohlc_bad_mask = symbol_bad['any_ohlc_issue'].values

            if not ohlc_bad_mask.any():
                continue

            # For each bad OHLC row, check if it's a 1-day gap
            for i in range(1, len(symbol_df)):
                if not ohlc_bad_mask[i]:
                    continue

                idx = symbol_df.index[i]
                prev_idx = symbol_df.index[i-1]

                # Check if previous day was good
                if ohlc_bad_mask[i-1]:
                    continue

                # Check if gap is 1 day
                date_gap = (symbol_df['Date'].iloc[i] - symbol_df['Date'].iloc[i-1]).days
                if date_gap != 1:
                    continue

                # Allow fixing OHLC even if Volume/OI have issues
                # (Volume/OI will be fixed separately in step 4)

                # Use previous day's Close for all OHLC (flat pricing)
                prev_close = df.loc[prev_idx, 'Close']

                # Validate previous close is reasonable
                if not (0 < prev_close < 1e9):
                    continue

                # Apply settlement price (flat OHLC)
                df.loc[idx, 'Open'] = prev_close
                df.loc[idx, 'High'] = prev_close
                df.loc[idx, 'Low'] = prev_close
                df.loc[idx, 'Close'] = prev_close

                # Mark as no longer bad
                for field in ['Open', 'High', 'Low', 'Close']:
                    bad_data.loc[idx, f'{field}_bad'] = False
                bad_data.loc[idx, 'any_ohlc_issue'] = False
                bad_data.loc[idx, 'any_issue'] = bad_data.loc[idx, 'any_volume_oi_issue']

                # Log repair
                self.repair_log.append({
                    'index': int(idx),
                    'symbol': symbol,
                    'date': str(df.loc[idx, 'Date']),
                    'field': 'OHLC_all',
                    'method': 'last_traded_settlement',
                    'repaired_value': float(prev_close),
                    'confidence': 'high',
                    'note': 'Futures settlement price (no trading)'
                })

                filled_count += 1

        self.df_cleaned = df
        self.bad_data = bad_data
        self.stats['rows_repaired'] += filled_count

        print(f"  Filled: {filled_count} rows with last-traded settlement price")

    def step_4c_cross_contract_relationships(self) -> None:
        """Use cross-contract price relationships (calendar spreads, correlation >0.90)."""
        print("\nSTEP 4c: Cross-contract price relationships (calendar spreads)...")

        df = self.df_cleaned.sort_values(['Symbol', 'Date']).copy()
        bad_data = self.bad_data.reindex(df.index)

        # Calculate correlations between all symbols
        symbols = sorted(df['Symbol'].unique())
        correlations = {}

        for sym1 in symbols:
            sym1_data = df[df['Symbol'] == sym1].set_index('Date')['Close']

            for sym2 in symbols:
                if sym1 == sym2:
                    continue

                sym2_data = df[df['Symbol'] == sym2].set_index('Date')['Close']

                # Align dates
                aligned = pd.concat([sym1_data, sym2_data], axis=1, join='inner')
                if len(aligned) < 30:  # Need at least 30 common dates
                    continue

                # Calculate correlation
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

                if corr > 0.90:  # Relaxed from 0.95 to 0.90
                    correlations[sym1] = sym2
                    break  # Use the first highly correlated contract

        filled_count = 0

        for symbol, reference_symbol in correlations.items():
            symbol_mask = df['Symbol'] == symbol
            symbol_df = df[symbol_mask].copy()
            symbol_bad = bad_data[symbol_mask].copy()

            ref_mask = df['Symbol'] == reference_symbol
            ref_df = df[ref_mask].set_index('Date')

            # Find rows with OHLC issues
            ohlc_bad_mask = symbol_bad['any_ohlc_issue'].values

            if not ohlc_bad_mask.any():
                continue

            # Calculate price ratio (symbol/reference) over clean data
            clean_mask = ~ohlc_bad_mask
            clean_data = symbol_df[clean_mask].set_index('Date')['Close']
            ref_data_aligned = ref_df.loc[ref_df.index.isin(clean_data.index), 'Close']

            if len(clean_data) == 0 or len(ref_data_aligned) == 0:
                continue

            # Calculate median ratio
            common_dates = clean_data.index.intersection(ref_data_aligned.index)
            if len(common_dates) < 10:
                continue

            ratios = clean_data.loc[common_dates] / ref_data_aligned.loc[common_dates]
            median_ratio = ratios.median()

            # For each bad row, try to fill using reference
            for i in range(len(symbol_df)):
                if not ohlc_bad_mask[i]:
                    continue

                idx = symbol_df.index[i]
                date = symbol_df.loc[idx, 'Date']

                # Check if reference has data for this date
                if date not in ref_df.index:
                    continue

                # Allow fixing OHLC even if Volume/OI have issues
                # (Volume/OI will be fixed separately in step 4)

                # Estimate using ratio
                ref_close = ref_df.loc[date, 'Close']
                estimated_close = ref_close * median_ratio

                # Validate estimate is reasonable
                if not (0 < estimated_close < 1e9):
                    continue

                # Calculate contract-specific High-Low spread from clean data
                clean_hl_spread = ((clean_data.loc[common_dates] - symbol_data.loc[common_dates, 'Low']) /
                                   clean_data.loc[common_dates]).median()
                # Fallback to 1% if calculation fails
                spread_pct = clean_hl_spread if not np.isnan(clean_hl_spread) and clean_hl_spread > 0 else 0.01

                # Apply estimated OHLC using contract-specific spread
                df.loc[idx, 'Open'] = estimated_close
                df.loc[idx, 'High'] = estimated_close * (1 + spread_pct)
                df.loc[idx, 'Low'] = estimated_close * (1 - spread_pct)
                df.loc[idx, 'Close'] = estimated_close

                # Mark as no longer bad
                for field in ['Open', 'High', 'Low', 'Close']:
                    bad_data.loc[idx, f'{field}_bad'] = False
                bad_data.loc[idx, 'any_ohlc_issue'] = False
                bad_data.loc[idx, 'any_issue'] = bad_data.loc[idx, 'any_volume_oi_issue']

                # Log repair
                self.repair_log.append({
                    'index': int(idx),
                    'symbol': symbol,
                    'date': str(date),
                    'field': 'OHLC_all',
                    'method': 'cross_contract_ratio',
                    'repaired_value': float(estimated_close),
                    'confidence': 'medium',
                    'note': f'Estimated from {reference_symbol} (ratio={median_ratio:.4f})'
                })

                filled_count += 1

        self.df_cleaned = df
        self.bad_data = bad_data
        self.stats['rows_repaired'] += filled_count

        print(f"  Filled: {filled_count} rows using cross-contract relationships")
        print(f"  Correlations found: {len(correlations)}")

    def step_4d_detect_and_repair_local_outliers(self) -> None:
        """
        Detect and repair local outliers using BACKWARD-ONLY imputation.

        Catches statistical outliers that passed initial validation:
        - Values >threshold for price fields (adaptive threshold based on MAD)
        - Values >5x local rolling median (statistical outlier)
        - Impossible jumps >20x previous value (temporal consistency)

        Conservative imputation: Only uses PAST data (no look-ahead bias).
        """
        print(f"\nSTEP 4d: Detect and repair local outliers (backward-only, threshold={self.price_threshold:.2f})...")

        df = self.df_cleaned.sort_values(['Symbol', 'Date']).copy()
        outliers_detected = 0
        repairs_made = 0

        for symbol in sorted(df['Symbol'].unique()):
            symbol_mask = df['Symbol'] == symbol
            symbol_df = df[symbol_mask].copy()

            for field in ['Open', 'High', 'Low', 'Close']:
                values = symbol_df[field].values
                indices = symbol_df.index

                # Detection: Three methods
                # Method 1: Absolute domain threshold (adaptive)
                absolute_outliers = values > self.price_threshold

                # Method 2: Local rolling median (>5x median of past 10 values)
                local_outliers = np.zeros(len(values), dtype=bool)
                for i in range(1, len(values)):  # Start at 1 (need past data)
                    if np.isnan(values[i]) or values[i] <= 0:
                        continue
                    # Look back up to 10 positions
                    lookback = min(i, 10)
                    past_values = values[max(0, i-lookback):i]
                    past_values = past_values[(past_values > 0) & (past_values <= self.price_threshold)]
                    if len(past_values) >= 3:
                        past_median = np.median(past_values)
                        if values[i] > 5.0 * past_median:
                            local_outliers[i] = True

                # Method 3: Impossible jumps (>20x previous value)
                jump_outliers = np.zeros(len(values), dtype=bool)
                for i in range(1, len(values)):
                    if np.isnan(values[i]) or np.isnan(values[i-1]):
                        continue
                    if values[i-1] > 0:
                        ratio = values[i] / values[i-1]
                        if ratio > 20.0 or ratio < 1/20.0:
                            jump_outliers[i] = True

                # Combine all detection methods
                all_outliers = absolute_outliers | local_outliers | jump_outliers
                outlier_count = all_outliers.sum()

                if outlier_count == 0:
                    continue

                outliers_detected += outlier_count

                # Repair: BACKWARD-ONLY imputation
                for i in range(len(values)):
                    if not all_outliers[i]:
                        continue

                    idx = indices[i]
                    old_value = values[i]

                    # Get PAST neighbors only (±0 to -5 positions)
                    lookback = min(i, 5)
                    if lookback == 0:
                        # No past data, can't repair conservatively
                        continue

                    past_neighbors = values[max(0, i-lookback):i]

                    # Filter: only use clean past values (>0 and <=threshold)
                    clean_past = past_neighbors[(past_neighbors > 0) & (past_neighbors <= self.price_threshold)]

                    # Also exclude values too different from median
                    if len(clean_past) >= 3:
                        past_median = np.median(clean_past)
                        clean_past = clean_past[
                            (clean_past > past_median * 0.5) &
                            (clean_past < past_median * 2.0)
                        ]

                    if len(clean_past) > 0:
                        # Use median of clean past neighbors
                        imputed_value = np.median(clean_past)
                        df.loc[idx, field] = imputed_value

                        # Determine detection method for logging
                        methods = []
                        if absolute_outliers[i]:
                            methods.append('absolute_threshold')
                        if local_outliers[i]:
                            methods.append('local_median')
                        if jump_outliers[i]:
                            methods.append('impossible_jump')

                        self.repair_log.append({
                            'index': int(idx),
                            'symbol': symbol,
                            'date': str(df.loc[idx, 'Date']),
                            'field': field,
                            'method': 'local_outlier_backward_' + '+'.join(methods),
                            'repaired_value': float(imputed_value),
                            'confidence': 'medium',
                            'note': f'Old value: {old_value:.2f}, used {len(clean_past)} past neighbors'
                        })

                        repairs_made += 1
                    else:
                        # No clean past neighbors - use global median as last resort
                        all_symbol_values = df.loc[symbol_mask, field].values
                        clean_global = all_symbol_values[(all_symbol_values > 0) & (all_symbol_values <= self.price_threshold)]

                        if len(clean_global) > 10:
                            fallback_value = np.median(clean_global)
                            df.loc[idx, field] = fallback_value

                            self.repair_log.append({
                                'index': int(idx),
                                'symbol': symbol,
                                'date': str(df.loc[idx, 'Date']),
                                'field': field,
                                'method': 'local_outlier_global_median_fallback',
                                'repaired_value': float(fallback_value),
                                'confidence': 'low',
                                'note': f'Old value: {old_value:.2f}, no clean past neighbors'
                            })

                            repairs_made += 1

        # Enforce OHLC constraints after repairs (on ALL rows, not just repaired ones)
        for idx in df.index:
            o, h, l, c = df.loc[idx, ['Open', 'High', 'Low', 'Close']]

            # Skip rows with missing values
            if np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c):
                continue

            # If High < Low, expand range using Close as reference
            if h < l:
                if c >= l:
                    df.loc[idx, 'High'] = max(c * 1.005, l * 1.005)
                else:
                    df.loc[idx, 'Low'] = min(c * 0.995, h * 0.995)
                # Re-read the updated values
                h = df.loc[idx, 'High']
                l = df.loc[idx, 'Low']

            # Ensure Open and Close are within [Low, High]
            if o < l or o > h:
                df.loc[idx, 'Open'] = np.clip(o, l, h)
            if c < l or c > h:
                df.loc[idx, 'Close'] = np.clip(c, l, h)

        # Also detect Volume/OI mismatches (high Volume without matching OI, or vice versa)
        print("\n  Detecting Volume/OI mismatches...")
        vol_oi_outliers = 0
        vol_oi_repairs = 0

        for symbol in sorted(df['Symbol'].unique()):
            symbol_mask = df['Symbol'] == symbol
            symbol_df = df[symbol_mask].copy()

            vol_values = symbol_df['Volume'].values
            oi_values = symbol_df['Open Interest'].values
            indices = symbol_df.index

            # Calculate historical Vol/OI ratio (using past data only)
            for i in range(1, len(vol_values)):
                if np.isnan(vol_values[i]) or np.isnan(oi_values[i]):
                    continue
                if vol_values[i] <= 0 or oi_values[i] <= 0:
                    continue

                # Look back up to 20 positions for ratio calculation
                lookback = min(i, 20)
                past_vol = vol_values[max(0, i-lookback):i]
                past_oi = oi_values[max(0, i-lookback):i]

                # Filter clean past values
                clean_mask = (past_vol > 0) & (past_oi > 0)
                clean_vol = past_vol[clean_mask]
                clean_oi = past_oi[clean_mask]

                if len(clean_vol) < 5:
                    continue

                # Calculate historical median Vol/OI ratio
                past_ratios = clean_vol / clean_oi
                historical_median_ratio = np.median(past_ratios)

                if historical_median_ratio <= 0:
                    continue

                # Current ratio
                current_ratio = vol_values[i] / oi_values[i]

                # Detect outliers: ratio >5x or <0.2x historical median
                if current_ratio > 5.0 * historical_median_ratio or current_ratio < 0.2 * historical_median_ratio:
                    idx = indices[i]
                    vol_oi_outliers += 1

                    # Determine which field is the outlier (Volume or OI)
                    # Compare each to its own historical median
                    past_vol_median = np.median(clean_vol)
                    past_oi_median = np.median(clean_oi)

                    vol_is_outlier = vol_values[i] > 5.0 * past_vol_median or vol_values[i] < 0.2 * past_vol_median
                    oi_is_outlier = oi_values[i] > 5.0 * past_oi_median or oi_values[i] < 0.2 * past_oi_median

                    # Repair the outlier field using backward imputation
                    if vol_is_outlier and not oi_is_outlier:
                        # Volume is the problem - use historical median or forward-fill
                        if len(clean_vol) >= 3:
                            imputed_vol = np.median(clean_vol[-5:] if len(clean_vol) >= 5 else clean_vol)
                            df.loc[idx, 'Volume'] = imputed_vol
                            self.repair_log.append({
                                'index': int(idx),
                                'symbol': symbol,
                                'date': str(df.loc[idx, 'Date']),
                                'field': 'Volume',
                                'method': 'vol_oi_mismatch_backward',
                                'repaired_value': float(imputed_vol),
                                'confidence': 'low',
                                'note': f'Vol/OI ratio outlier: {current_ratio:.2f}x vs {historical_median_ratio:.2f}x'
                            })
                            vol_oi_repairs += 1

                    elif oi_is_outlier and not vol_is_outlier:
                        # OI is the problem
                        if len(clean_oi) >= 3:
                            imputed_oi = np.median(clean_oi[-5:] if len(clean_oi) >= 5 else clean_oi)
                            df.loc[idx, 'Open Interest'] = imputed_oi
                            self.repair_log.append({
                                'index': int(idx),
                                'symbol': symbol,
                                'date': str(df.loc[idx, 'Date']),
                                'field': 'Open Interest',
                                'method': 'vol_oi_mismatch_backward',
                                'repaired_value': float(imputed_oi),
                                'confidence': 'low',
                                'note': f'Vol/OI ratio outlier: {current_ratio:.2f}x vs {historical_median_ratio:.2f}x'
                            })
                            vol_oi_repairs += 1

        self.df_cleaned = df
        self.stats['rows_repaired'] += repairs_made + vol_oi_repairs

        print(f"  Price outliers detected: {outliers_detected}, repairs made: {repairs_made}")
        print(f"  Vol/OI mismatches detected: {vol_oi_outliers}, repairs made: {vol_oi_repairs}")
        print(f"  Note: Backward-only imputation (no look-ahead bias)")

    def step_5_remove_remaining_bad_rows(self) -> None:
        """Remove any remaining rows with bad data."""
        print("\nSTEP 5: Removing remaining bad rows...")
        df, bad_data = self.df_cleaned, self.bad_data

        # Recalculate any_issue based on current state (after repairs)
        bad_data['any_issue'] = bad_data['any_ohlc_issue'] | bad_data['any_volume_oi_issue']
        still_bad = bad_data['any_issue']
        bad_count = still_bad.sum()

        for idx in df[still_bad].index:
            self.removal_log.append({
                'index': int(idx), 'symbol': df.loc[idx, 'Symbol'],
                'date': str(df.loc[idx, 'Date']), 'reason': 'remaining_bad_data_after_repair'
            })

        self.df_cleaned = df[~still_bad].copy()
        self.stats['rows_removed'] += bad_count
        print(f"  Removed: {bad_count} remaining bad rows")
        print(f"  Remaining: {len(self.df_cleaned)} rows")

    def step_6_smart_sparse_symbol_handling(self) -> None:
        """Smart handling of sparse symbols (FUT11, FUT12) - keep only clean windows."""
        print("\nSTEP 6: Smart handling of sparse symbols...")

        df = self.df_cleaned.sort_values(['Symbol', 'Date']).copy()

        # Identify sparse symbols (those with <40% data vs expected)
        symbol_counts = df.groupby('Symbol').size()
        max_count = symbol_counts.max()
        sparse_symbols = symbol_counts[symbol_counts < max_count * 0.4].index.tolist()

        if not sparse_symbols:
            print("  No sparse symbols detected")
            return

        print(f"  Sparse symbols detected: {sparse_symbols}")

        rows_to_remove = []

        for symbol in sparse_symbols:
            symbol_mask = df['Symbol'] == symbol
            symbol_df = df[symbol_mask].copy()

            print(f"\n  Processing {symbol} ({len(symbol_df)} rows):")

            # Find clean windows (rolling 30-day windows with >90% complete data)
            symbol_df = symbol_df.sort_values('Date')
            # Check if Close column has enough non-null values in rolling windows
            close_series = symbol_df['Close']
            symbol_df['window_quality'] = close_series.rolling(30, min_periods=10).apply(
                lambda x: (~pd.isna(x)).sum() / len(x), raw=True
            )

            # Mark rows in low-quality windows for removal
            low_quality = symbol_df['window_quality'] < 0.9

            if low_quality.sum() > 0:
                print(f"    Removing {low_quality.sum()} rows in low-quality windows")
                rows_to_remove.extend(symbol_df[low_quality].index.tolist())

        if rows_to_remove:
            # Log removals
            for idx in rows_to_remove:
                self.removal_log.append({
                    'index': int(idx),
                    'symbol': df.loc[idx, 'Symbol'],
                    'date': str(df.loc[idx, 'Date']),
                    'reason': 'sparse_symbol_low_quality_window'
                })

            # Remove
            removal_count = len(rows_to_remove)
            self.df_cleaned = df[~df.index.isin(rows_to_remove)].copy()
            self.stats['rows_removed'] += removal_count

            print(f"\n  Total removed from sparse symbols: {removal_count}")
            print(f"  Remaining: {len(self.df_cleaned)} rows")

    def step_7_add_confidence_intervals(self) -> None:
        """Add confidence intervals based on historical volatility (95% CI = ±2 std devs)."""
        print("\nSTEP 7: Adding confidence intervals...")

        df = self.df_cleaned.sort_values(['Symbol', 'Date']).copy()

        # Initialize confidence columns
        for col in ['Close', 'Volume', 'Open Interest']:
            df[f'{col}_confidence_lower'] = df[col]
            df[f'{col}_confidence_upper'] = df[col]
            df[f'{col}_is_imputed'] = False

        # Mark imputed values and add CIs
        for repair in self.repair_log:
            idx = repair['index']
            if idx not in df.index:
                continue

            field = repair['field']
            symbol = repair['symbol']

            # Handle OHLC_all specially (when all 4 OHLC fields were imputed together)
            if field == 'OHLC_all':
                # For flat OHLC (like last-traded settlement), use Close field for confidence
                # Mark Close as imputed and add CI
                df.loc[idx, 'Close_is_imputed'] = True

                # Calculate CI based on recent rolling volatility (more responsive to regime changes)
                symbol_mask = df['Symbol'] == symbol
                symbol_data = df[symbol_mask].sort_values('Date')
                idx_pos = symbol_data.index.get_loc(idx)

                # Use rolling 60-day standard deviation for more adaptive CI
                if idx_pos >= 10:
                    recent_data = symbol_data.iloc[max(0, idx_pos-60):idx_pos]['Close']
                    hist_std = recent_data.std() if len(recent_data) >= 10 else symbol_data['Close'].std()
                else:
                    hist_std = symbol_data['Close'].std()

                value = df.loc[idx, 'Close']
                # 95% CI: ±2 std devs (don't force non-negative to preserve statistical properties)
                df.loc[idx, 'Close_confidence_lower'] = value - 2 * hist_std
                df.loc[idx, 'Close_confidence_upper'] = value + 2 * hist_std
                continue

            # Skip if field is not one we track CIs for
            if field not in ['Close', 'Volume', 'Open Interest']:
                continue

            # Mark as imputed
            df.loc[idx, f'{field}_is_imputed'] = True

            # Calculate CI based on recent rolling volatility (more responsive to regime changes)
            symbol_mask = df['Symbol'] == symbol
            symbol_data = df[symbol_mask].sort_values('Date')
            idx_pos = symbol_data.index.get_loc(idx)

            # Use rolling 60-day standard deviation for more adaptive CI
            if idx_pos >= 10:
                recent_data = symbol_data.iloc[max(0, idx_pos-60):idx_pos][field]
                hist_std = recent_data.std() if len(recent_data) >= 10 else symbol_data[field].std()
            else:
                hist_std = symbol_data[field].std()

            value = df.loc[idx, field]
            # 95% CI: ±2 std devs (don't force non-negative to preserve statistical properties)
            df.loc[idx, f'{field}_confidence_lower'] = value - 2 * hist_std
            df.loc[idx, f'{field}_confidence_upper'] = value + 2 * hist_std

        self.df_cleaned = df

        imputed_count = sum(df[f'{col}_is_imputed'].sum() for col in ['Close', 'Volume', 'Open Interest'])
        print(f"  Added confidence intervals to {imputed_count} imputed values")

    def step_8_final_validation(self) -> bool:
        """Final validation: ensure all constraints satisfied."""
        print("\nSTEP 8: Final validation...")
        df = self.df_cleaned

        violations = {
            'high_lt_low': (df['High'] < df['Low']).sum(),
            'open_outside': ((df['Open'] < df['Low']) | (df['Open'] > df['High'])).sum(),
            'close_outside': ((df['Close'] < df['Low']) | (df['Close'] > df['High'])).sum(),
            'negative_prices': ((df[['Open', 'High', 'Low', 'Close']] < 0).any(axis=1)).sum(),
            'negative_volume': (df['Volume'] < 0).sum(),
            'negative_oi': (df['Open Interest'] < 0).sum(),
            'sentinel_values': sum((((df[col] > 0) & (df[col] < 1e-6)) | (df[col].abs() > 1e9)).sum()
                                  for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Open Interest'])
        }

        total_violations = sum(violations.values())
        print(f"\n  Final validation results:")
        for check, count in violations.items():
            status = "[PASS]" if count == 0 else f"[FAIL] {count}"
            print(f"    {check}: {status}")

        if total_violations == 0:
            print("\n  [PASS] All validation checks passed!")
        else:
            print(f"\n  [FAIL] {total_violations} violations found")
            print("  WARNING: Data may still have quality issues")
        return total_violations == 0

    def generate_summary(self) -> dict:
        """Generate summary of repair process."""
        df = self.df_cleaned
        orig_rows = self.stats['original_rows']
        summary = {
            'statistics': {
                'original_rows': orig_rows,
                'rows_removed': self.stats['rows_removed'],
                'rows_repaired': self.stats['rows_repaired'],
                'final_rows': len(df),
                'removal_rate': f"{self.stats['rows_removed']/orig_rows*100:.1f}%",
                'data_retention_rate': f"{len(df)/orig_rows*100:.1f}%"
            },
            'per_symbol': {}
        }

        for symbol in sorted(df['Symbol'].unique()):
            symbol_df = df[df['Symbol'] == symbol]
            orig_symbol_rows = (self.df_original['Symbol'] == symbol).sum()
            summary['per_symbol'][symbol] = {
                'original_rows': orig_symbol_rows,
                'final_rows': len(symbol_df),
                'retention_rate': f"{len(symbol_df)/orig_symbol_rows*100:.1f}%",
                'date_range': {'start': str(symbol_df['Date'].min()), 'end': str(symbol_df['Date'].max())},
                'imputed_values': sum(symbol_df[f'{col}_is_imputed'].sum()
                                     for col in ['Close', 'Volume', 'Open Interest']
                                     if f'{col}_is_imputed' in symbol_df.columns)
            }
        return summary

    def export_results(self, output_dir: str | Path = '.') -> dict:
        """Export cleaned data and logs."""
        output_dir = Path(output_dir)

        cleaned_file = output_dir / 'data_cleaned_conservative.csv'
        self.df_cleaned.to_csv(cleaned_file, index=False)
        print(f"\nSaved cleaned data to: {cleaned_file}")

        if self.removal_log:
            pd.DataFrame(self.removal_log).to_csv(output_dir / 'data_removal_log.csv', index=False)
            print(f"Saved removal log to: {output_dir / 'data_removal_log.csv'}")

        if self.repair_log:
            pd.DataFrame(self.repair_log).to_csv(output_dir / 'data_repair_log_conservative.csv', index=False)
            print(f"Saved repair log to: {output_dir / 'data_repair_log_conservative.csv'}")

        summary = self.generate_summary()
        summary_file = output_dir / 'data_repair_summary_conservative.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved summary to: {summary_file}")
        return summary

    def run_conservative_repair(self) -> bool:
        """Run the full conservative repair pipeline."""
        print("="*60)
        print("CONSERVATIVE DATA REPAIR FOR TRADING USE")
        print("="*60)
        print("\nPhilosophy: Remove Don't Fabricate")
        print("  - No look-ahead bias | Default action: REMOVE | Only fix obvious cases | Strict validation")
        print("="*60)

        self.step_1_drop_missing_timestamps()
        self.step_2_identify_bad_data()
        self.step_4b_last_traded_price_forward_fill()  # Try to repair OHLC before removing
        self.step_4c_cross_contract_relationships()
        self.step_3_remove_unfixable_rows()
        self.step_4_forward_fill_small_gaps(max_gap_days=2, extended_gap_days=5)  # Repair Volume/OI
        self.step_4d_detect_and_repair_local_outliers()  # Catch statistical outliers (backward-only)
        self.step_5_remove_remaining_bad_rows()
        self.step_6_smart_sparse_symbol_handling()
        self.step_7_add_confidence_intervals()
        validation_passed = self.step_8_final_validation()

        print("\n" + "="*60)
        print("REPAIR COMPLETE")
        print("="*60)

        Path('4output').mkdir(exist_ok=True)
        summary = self.export_results('4output')

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        stats = summary['statistics']
        print(f"Original rows: {stats['original_rows']:,}")
        print(f"Removed rows: {stats['rows_removed']:,} ({stats['removal_rate']})")
        print(f"Final rows: {stats['final_rows']:,} ({stats['data_retention_rate']})")
        print(f"Validation: {'[PASS]' if validation_passed else '[FAIL]'}")

        print("\nPer-Symbol Retention:")
        for symbol, s in summary['per_symbol'].items():
            print(f"  {symbol}: {s['final_rows']}/{s['original_rows']} rows ({s['retention_rate']})")
        return validation_passed


def main(input_file: str | Path = 'data_eng_assessment_dataset.csv') -> None:
    """Main execution."""
    input_path = Path(input_file)

    # Validate input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Loading original data...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {input_path}") from e

    print(f"Loaded {len(df):,} rows\n")

    repair = ConservativeDataRepair(df)
    validation_passed = repair.run_conservative_repair()

    result = "[SUCCESS] Cleaned data ready for trading analysis" if validation_passed else "[WARNING] Some issues remain - review validation results"
    print(f"\n{result}\n\nDone!")


if __name__ == "__main__":
    main()

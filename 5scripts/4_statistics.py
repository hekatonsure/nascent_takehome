"""
Exploratory Analysis Suite for Futures Trading Data

Consolidated analysis utilities:
1. Sentinel Value Analysis - categorize corrupted/placeholder values
2. Outlier Detection - IQR/Z-score methods for statistical outliers
3. Statistical Analysis - comprehensive exploratory analysis

Usage:
    python exploratory_analysis.py --analysis [sentinel|outliers|statistical|all]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
from pathlib import Path
from collections import defaultdict
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# SENTINEL VALUE ANALYSIS
# ============================================================================

def analyze_sentinel_patterns(df: pd.DataFrame, flags_path: str | None = None):
    """
    Analyze and categorize all types of bad/sentinel values.

    Uses data_quality_flags.csv if available to avoid re-analyzing.
    """
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Open Interest']
    examples = defaultdict(list)
    counts = defaultdict(int)

    # Try to load flags to avoid re-analyzing
    flags_df = None
    if flags_path:
        try:
            flags_df = pd.read_csv(flags_path)
            print("  Using existing data_quality_flags.csv (avoiding re-analysis)")
        except FileNotFoundError:
            print("  Flags file not found, analyzing original data...")

    # If flags available, only examine flagged rows
    if flags_df is not None and 'sentinel_values' in flags_df.columns:
        sentinel_mask = flags_df['sentinel_values']
        df_to_analyze = df[sentinel_mask].copy()
        print(f"  Analyzing {len(df_to_analyze)} flagged rows (instead of all {len(df)} rows)")
    else:
        df_to_analyze = df.copy()
        print(f"  Analyzing all {len(df)} rows (no flags available)")

    for col in numeric_cols:
        if col not in df_to_analyze.columns:
            continue

        col_data = df_to_analyze[col].dropna()

        if len(col_data) == 0:
            continue

        # Scientific notation small (< 1e-6)
        small_sci = col_data[(col_data > 0) & (col_data < 1e-6)]
        if len(small_sci) > 0:
            unique_vals = small_sci.unique()
            for val in unique_vals[:5]:
                examples[f'{col}_scientific_small'].append(val)
            counts[f'{col}_scientific_small'] = len(small_sci)

        # Scientific notation large (> 1e15)
        large_sci = col_data[col_data.abs() > 1e15]
        if len(large_sci) > 0:
            unique_vals = large_sci.unique()
            for val in unique_vals[:5]:
                examples[f'{col}_scientific_large'].append(val)
            counts[f'{col}_scientific_large'] = len(large_sci)

        # Large round numbers (>= 1e10 and divisible by 1e10)
        large_round = col_data[(col_data.abs() >= 1e10) & ((col_data % 1e10) == 0)]
        if len(large_round) > 0:
            unique_vals = large_round.unique()
            for val in unique_vals[:10]:
                key = f'{col}_large_round'
                examples[key].append(val)
            counts[f'{col}_large_round'] = len(large_round)

        # Negative values
        negatives = col_data[col_data < 0]
        if len(negatives) > 0:
            unique_vals = negatives.unique()
            for val in sorted(unique_vals)[:10]:
                examples[f'{col}_negative'].append(val)
            counts[f'{col}_negative'] = len(negatives)

        # Zero values in price fields
        if col in ['Open', 'High', 'Low', 'Close']:
            zeros = col_data[col_data == 0]
            if len(zeros) > 0:
                counts[f'{col}_zero'] = len(zeros)

        # Extremely large non-round values (1e9 to 1e15)
        extreme = col_data[(col_data.abs() >= 1e9) & (col_data.abs() < 1e15) & ((col_data % 1e10) != 0)]
        if len(extreme) > 0:
            unique_vals = extreme.unique()
            for val in unique_vals[:10]:
                examples[f'{col}_extreme'].append(val)
            counts[f'{col}_extreme'] = len(extreme)

    return examples, counts


def print_sentinel_analysis(df: pd.DataFrame, flags_path: str | None = None):
    """Print detailed analysis of all sentinel/corrupted values."""

    print("="*70)
    print("SENTINEL/CORRUPTED VALUE ANALYSIS")
    print("="*70)

    examples, counts = analyze_sentinel_patterns(df, flags_path)

    categories = {
        'VERY SMALL VALUES (Scientific Notation < 1e-6)': 'scientific_small',
        'VERY LARGE VALUES (Scientific Notation > 1e15)': 'scientific_large',
        'LARGE ROUND PLACEHOLDER VALUES (± billions/trillions)': 'large_round',
        'NEGATIVE VALUES': 'negative',
        'ZERO PRICE VALUES': 'zero',
        'EXTREME NON-ROUND VALUES (1e9 to 1e15)': 'extreme'
    }

    for category_name, pattern in categories.items():
        print(f"\n{category_name}")
        print("-"*70)

        relevant_keys = [k for k in counts.keys() if pattern in k]

        if not relevant_keys:
            print("  None found")
            continue

        total = sum(counts[k] for k in relevant_keys)
        print(f"  Total occurrences: {total:,}")
        print()

        for key in sorted(relevant_keys):
            field = key.split('_')[0]
            count = counts[key]
            print(f"  {field:15s}: {count:5,} occurrences")

            if key in examples and examples[key]:
                unique_examples = sorted(set(examples[key]))
                print(f"                   Examples: {unique_examples[:5]}")

    print("\n" + "="*70)


def print_value_range_summary(df: pd.DataFrame):
    """Print summary of value ranges for each field."""

    print("\n" + "="*70)
    print("VALUE RANGE SUMMARY (for context)")
    print("="*70)

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Open Interest']

    for col in numeric_cols:
        if col not in df.columns:
            continue

        col_data = df[col].dropna()

        # Filter out obvious sentinels for "realistic" range
        realistic = col_data[
            (col_data > 0) &
            (col_data < 1e6) &
            (col_data >= 1e-2)
        ]

        print(f"\n{col}:")
        print(f"  All values:        min={col_data.min():.2e}, max={col_data.max():.2e}")
        if len(realistic) > 0:
            print(f"  'Realistic' range: min={realistic.min():.2f}, max={realistic.max():.2f}")
            print(f"  'Realistic' count: {len(realistic):,} / {len(col_data):,} ({len(realistic)/len(col_data)*100:.1f}%)")
        else:
            print(f"  'Realistic' range: NO REALISTIC VALUES FOUND")


# ============================================================================
# OUTLIER DETECTION
# ============================================================================

def detect_outliers_iqr(series: pd.Series, multiplier: float = 3.0) -> pd.Series:
    """
    Detect outliers using IQR method.

    Args:
        series: Data series
        multiplier: IQR multiplier (1.5 = mild, 3.0 = extreme)

    Returns:
        Boolean series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    return (series < lower_bound) | (series > upper_bound)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using Z-score method.

    Args:
        series: Data series
        threshold: Z-score threshold (3.0 = 99.7% of normal distribution)

    Returns:
        Boolean series indicating outliers
    """
    z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
    return z_scores > threshold


def detect_price_jumps(df: pd.DataFrame, symbol: str, threshold: float = 0.5) -> pd.Series:
    """
    Detect unrealistic price jumps (>50% day-over-day).

    Args:
        df: DataFrame with Date and Close columns
        symbol: Symbol to analyze
        threshold: Max allowed daily change (0.5 = 50%)

    Returns:
        Boolean series indicating suspicious jumps
    """
    symbol_df = df[df['Symbol'] == symbol].sort_values('Date')

    if len(symbol_df) < 2:
        return pd.Series(False, index=symbol_df.index)

    pct_change = symbol_df['Close'].pct_change().abs()
    large_jumps = pct_change > threshold

    return large_jumps


def analyze_outliers_by_symbol(df: pd.DataFrame):
    """Comprehensive outlier analysis by symbol."""

    print("="*70)
    print("OUTLIER ANALYSIS BY SYMBOL")
    print("="*70)

    results = {}

    for symbol in sorted(df['Symbol'].unique()):
        symbol_df = df[df['Symbol'] == symbol].copy()

        print(f"\n{symbol}:")
        print("-" * 70)

        symbol_results = {
            'total_rows': len(symbol_df),
            'outliers': {}
        }

        # Analyze each numeric field
        for field in ['Close', 'Volume', 'Open Interest']:
            if field not in symbol_df.columns:
                continue

            # Get clean data
            clean_data = symbol_df[field].copy()
            clean_data = clean_data[(clean_data > 0) & (clean_data < 1e9)]

            if len(clean_data) < 10:
                print(f"  {field}: Insufficient clean data ({len(clean_data)} rows)")
                continue

            # Detect outliers
            outliers_iqr = detect_outliers_iqr(clean_data, multiplier=3.0)
            outliers_zscore = detect_outliers_zscore(clean_data, threshold=3.0)
            strong_outliers = outliers_iqr & outliers_zscore

            # Stats
            Q1, Q3 = clean_data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 3.0 * IQR
            upper_bound = Q3 + 3.0 * IQR

            outlier_count = strong_outliers.sum()

            symbol_results['outliers'][field] = {
                'count': int(outlier_count),
                'percentage': float(outlier_count / len(clean_data) * 100),
                'bounds': {
                    'lower': float(lower_bound),
                    'upper': float(upper_bound)
                },
                'actual_range': {
                    'min': float(clean_data.min()),
                    'max': float(clean_data.max())
                }
            }

            if outlier_count > 0:
                outlier_vals = clean_data[strong_outliers]
                print(f"  {field}: {outlier_count} outliers ({outlier_count/len(clean_data)*100:.1f}%)")
                print(f"    Expected range: [{lower_bound:.2f}, {upper_bound:.2f}]")
                print(f"    Actual range: [{clean_data.min():.2f}, {clean_data.max():.2f}]")
                print(f"    Outlier examples: {outlier_vals.head(3).tolist()}")
            else:
                print(f"  {field}: No outliers detected")

        # Price jump analysis
        if 'Date' in symbol_df.columns:
            price_jumps = detect_price_jumps(df, symbol, threshold=0.5)
            jump_count = price_jumps.sum()

            symbol_results['price_jumps'] = {
                'count': int(jump_count),
                'percentage': float(jump_count / len(symbol_df) * 100)
            }

            if jump_count > 0:
                print(f"  Price jumps (>50%): {jump_count} ({jump_count/len(symbol_df)*100:.1f}%)")

        results[symbol] = symbol_results

    return results


def visualize_outliers(df: pd.DataFrame, symbol: str, field: str = 'Close', output_dir: Path = Path('.')):
    """Visualize outliers for a specific symbol."""

    symbol_df = df[df['Symbol'] == symbol].sort_values('Date')

    if len(symbol_df) < 10:
        print(f"Insufficient data for {symbol}")
        return

    # Get data
    data = symbol_df[field].copy()
    dates = symbol_df['Date']

    # Clean data
    clean_mask = (data > 0) & (data < 1e9)
    data_clean = data[clean_mask]

    # Detect outliers
    outliers = detect_outliers_iqr(data_clean, multiplier=3.0)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Time series
    ax1.plot(dates[clean_mask], data_clean, 'b-', alpha=0.6, label='Normal')
    if outliers.any():
        ax1.scatter(dates[clean_mask][outliers], data_clean[outliers],
                   color='red', s=50, label='Outliers', zorder=5)
    ax1.set_title(f'{symbol} - {field} Time Series with Outliers')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(field)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(data_clean, vert=False)
    ax2.set_title(f'{symbol} - {field} Distribution')
    ax2.set_xlabel(field)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f'outlier_analysis_{symbol}_{field}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")


# ============================================================================
# STATISTICAL ANALYSIS (Full exploratory framework)
# ============================================================================

class StatisticalAnalysis:
    """Statistical analysis engine for futures data."""

    def __init__(self, df_cleaned: pd.DataFrame, df_original: pd.DataFrame, flags: pd.DataFrame):
        """
        Initialize analysis engine.

        Args:
            df_cleaned: Cleaned data from data_repair.py
            df_original: Original data for comparison
            flags: Quality flags
        """
        self.df_cleaned = df_cleaned.copy()
        self.df_original = df_original.copy()
        self.flags = flags.copy()

        # Ensure Date column
        for df in [self.df_cleaned, self.df_original]:
            if 'Date' not in df.columns and 'Timestamp' in df.columns:
                df['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['Timestamp'], unit='D')
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])

        # Create output directory
        # Check if we're in root or 5scripts/ directory
        if Path('3figures').exists() or Path('4output').exists():
            self.output_dir = Path('3figures')
        else:
            self.output_dir = Path('../3figures')
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Results storage
        self.results = {}

    def analyze_distributions(self):
        """Analyze distributions of prices, returns, volume, and OI."""
        print("Analyzing distributions...")

        results = {}

        # Price distributions by symbol
        price_stats = self.df_cleaned.groupby('Symbol')['Close'].agg([
            'count', 'mean', 'std', 'min', 'max', 'skew',
            lambda x: stats.kurtosis(x.dropna())
        ])
        price_stats.columns = ['count', 'mean', 'std', 'min', 'max', 'skew', 'kurtosis']
        results['price_statistics'] = price_stats.to_dict()

        # Calculate returns
        self.df_cleaned = self.df_cleaned.sort_values(['Symbol', 'Date'])
        self.df_cleaned['returns'] = self.df_cleaned.groupby('Symbol')['Close'].pct_change()

        # Return distributions
        return_stats = self.df_cleaned.groupby('Symbol')['returns'].agg([
            'mean', 'std', 'skew',
            lambda x: stats.kurtosis(x.dropna())
        ])
        return_stats.columns = ['mean', 'std', 'skew', 'kurtosis']
        results['return_statistics'] = return_stats.to_dict()

        # Test for normality
        normality_tests = {}
        for symbol in self.df_cleaned['Symbol'].unique():
            returns = self.df_cleaned[self.df_cleaned['Symbol'] == symbol]['returns'].dropna()
            if len(returns) > 10:
                jb_stat, jb_pval = stats.jarque_bera(returns)
                normality_tests[symbol] = {
                    'statistic': float(jb_stat),
                    'p_value': float(jb_pval),
                    'is_normal': jb_pval > 0.05
                }

        results['normality_tests'] = normality_tests

        # Volume and OI
        vol_stats = self.df_cleaned.groupby('Symbol')['Volume'].agg(['mean', 'std', 'median'])
        oi_stats = self.df_cleaned.groupby('Symbol')['Open Interest'].agg(['mean', 'std', 'median'])

        results['volume_statistics'] = vol_stats.to_dict()
        results['oi_statistics'] = oi_stats.to_dict()

        self.results['distributions'] = results

        # Visualizations
        self._plot_price_distributions()
        self._plot_return_distributions()

    def _plot_price_distributions(self):
        """Plot price distributions by symbol."""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        symbols = sorted(self.df_cleaned['Symbol'].unique())

        for i, symbol in enumerate(symbols):
            if i >= len(axes):
                break

            data = self.df_cleaned[self.df_cleaned['Symbol'] == symbol]['Close']
            valid_data = data[(data > 0) & (data < 1e6)]

            axes[i].hist(valid_data, bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{symbol} Price Distribution')
            axes[i].set_xlabel('Price')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(valid_data.mean(), color='red', linestyle='--',
                           label=f'Mean: {valid_data.mean():.2f}')
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'price_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'price_distributions.png'}")

    def _plot_return_distributions(self):
        """Plot return distributions."""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        symbols = sorted(self.df_cleaned['Symbol'].unique())

        for i, symbol in enumerate(symbols):
            if i >= len(axes):
                break

            returns = self.df_cleaned[self.df_cleaned['Symbol'] == symbol]['returns'].dropna()

            axes[i].hist(returns, bins=50, alpha=0.7, edgecolor='black', density=True)
            axes[i].set_title(f'{symbol} Return Distribution')
            axes[i].set_xlabel('Returns')
            axes[i].set_ylabel('Density')

            # Overlay normal distribution
            mu, sigma = returns.mean(), returns.std()
            x = np.linspace(returns.min(), returns.max(), 100)
            axes[i].plot(x, stats.norm.pdf(x, mu, sigma), 'r-',
                        label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'return_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'return_distributions.png'}")

    def analyze_time_series_properties(self):
        """Analyze time series properties."""
        print("Analyzing time series properties...")

        results = {}

        # Autocorrelation
        acf_results = {}
        for symbol in self.df_cleaned['Symbol'].unique():
            returns = self.df_cleaned[self.df_cleaned['Symbol'] == symbol]['returns'].dropna()

            if len(returns) > 20:
                acf_vals = acf(returns, nlags=20, fft=True)
                significant_lags = np.where(np.abs(acf_vals[1:]) > 1.96/np.sqrt(len(returns)))[0] + 1

                acf_results[symbol] = {
                    'first_lag_acf': float(acf_vals[1]),
                    'significant_lags': significant_lags.tolist()[:5],
                    'has_autocorrelation': len(significant_lags) > 0
                }

        results['autocorrelation'] = acf_results

        # Stationarity tests
        stationarity_results = {}
        for symbol in self.df_cleaned['Symbol'].unique():
            prices = self.df_cleaned[self.df_cleaned['Symbol'] == symbol]['Close']
            valid_prices = prices[(prices > 0) & (prices < 1e6)].dropna()

            if len(valid_prices) > 30:
                try:
                    adf_result = adfuller(valid_prices, maxlag=10)
                    stationarity_results[symbol] = {
                        'adf_statistic': float(adf_result[0]),
                        'p_value': float(adf_result[1]),
                        'is_stationary': adf_result[1] < 0.05,
                        'critical_values': {k: float(v) for k, v in adf_result[4].items()}
                    }
                except:
                    stationarity_results[symbol] = {'error': 'Could not compute ADF test'}

        results['stationarity'] = stationarity_results

        # Volatility
        volatility_results = {}
        for symbol in self.df_cleaned['Symbol'].unique():
            returns = self.df_cleaned[self.df_cleaned['Symbol'] == symbol]['returns'].dropna()

            if len(returns) > 30:
                rolling_vol = returns.rolling(window=20).std()

                volatility_results[symbol] = {
                    'mean_volatility': float(returns.std()),
                    'volatility_of_volatility': float(rolling_vol.std()),
                    'min_volatility': float(rolling_vol.min()),
                    'max_volatility': float(rolling_vol.max())
                }

        results['volatility'] = volatility_results

        self.results['time_series'] = results

        # Visualizations (methods defined in original statistical_analysis.py)
        self._plot_autocorrelation()
        self._plot_volatility()

    def _plot_autocorrelation(self):
        """Plot autocorrelation functions."""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        symbols = sorted(self.df_cleaned['Symbol'].unique())

        for i, symbol in enumerate(symbols):
            if i >= len(axes):
                break

            returns = self.df_cleaned[self.df_cleaned['Symbol'] == symbol]['returns'].dropna()

            if len(returns) > 20:
                acf_vals = acf(returns, nlags=20, fft=True)
                lags = range(len(acf_vals))

                axes[i].stem(lags, acf_vals, basefmt=' ')
                axes[i].axhline(0, color='black', linewidth=0.8)
                axes[i].axhline(1.96/np.sqrt(len(returns)), color='red', linestyle='--',
                               label='95% CI')
                axes[i].axhline(-1.96/np.sqrt(len(returns)), color='red', linestyle='--')
                axes[i].set_title(f'{symbol} ACF')
                axes[i].set_xlabel('Lag')
                axes[i].set_ylabel('Autocorrelation')
                axes[i].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'autocorrelation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'autocorrelation.png'}")

    def _plot_volatility(self):
        """Plot rolling volatility."""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        symbols = sorted(self.df_cleaned['Symbol'].unique())

        for i, symbol in enumerate(symbols):
            if i >= len(axes):
                break

            symbol_data = self.df_cleaned[self.df_cleaned['Symbol'] == symbol].sort_values('Date')
            returns = symbol_data['returns'].dropna()

            if len(returns) > 30:
                rolling_vol = returns.rolling(window=20).std()

                vol_data = pd.DataFrame({
                    'date': symbol_data.loc[rolling_vol.index, 'Date'].values,
                    'vol': rolling_vol.values
                }).dropna()

                axes[i].plot(vol_data['date'], vol_data['vol'], linewidth=1)
                axes[i].set_title(f'{symbol} Rolling Volatility (20-day)')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Volatility')
                axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'volatility.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'volatility.png'}")

    def analyze_cross_sectional(self):
        """Analyze cross-sectional relationships."""
        print("Analyzing cross-sectional relationships...")

        # Create wide-format returns
        returns_wide = self.df_cleaned.pivot_table(
            index='Date',
            columns='Symbol',
            values='returns'
        )

        # Correlation matrix
        corr_matrix = returns_wide.corr()

        self.results['cross_sectional'] = {
            'correlation_matrix': corr_matrix.to_dict(),
            'mean_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean())
        }

        # PCA - exclude sparse symbols
        from sklearn.decomposition import PCA

        date_coverage = returns_wide.notna().sum() / len(returns_wide)
        sparse_symbols = [col for col in returns_wide.columns if date_coverage[col] < 0.5]
        good_symbols = [col for col in returns_wide.columns if col not in sparse_symbols]

        returns_wide_filtered = returns_wide[good_symbols]
        returns_clean = returns_wide_filtered.dropna()

        self.results['cross_sectional']['pca_info'] = {
            'total_dates': len(returns_wide),
            'symbols_analyzed': good_symbols,
            'symbols_excluded': sparse_symbols,
            'dates_with_complete_data': len(returns_clean),
            'date_coverage_per_symbol': {sym: float(date_coverage[sym]) for sym in returns_wide.columns}
        }

        if len(returns_clean) > 10 and len(returns_clean.columns) > 2:
            pca = PCA()
            pca.fit(returns_clean)

            self.results['cross_sectional']['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'n_components_90pct': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9) + 1),
                'n_samples': len(returns_clean),
                'n_components': len(good_symbols)
            }
            print(f"  PCA: Using {len(good_symbols)} symbols, {len(returns_clean)} complete dates")
        else:
            self.results['cross_sectional']['pca'] = {
                'note': 'Insufficient overlapping dates for PCA after conservative cleaning',
                'reason': 'Conservative data cleaning removed different dates per symbol',
                'recommendation': 'Use pairwise correlations instead'
            }
            print(f"  PCA: Insufficient data ({len(returns_clean)} complete dates)")

        # Visualizations (abbreviated - full versions in original file)
        self._plot_correlation_matrix(corr_matrix)
        self._plot_pca_variance()

    def _plot_correlation_matrix(self, corr_matrix):
        """Plot correlation heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, square=True)
        plt.title('Return Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'correlation_matrix.png'}")

    def _plot_pca_variance(self):
        """Plot PCA variance (abbreviated version)."""
        if 'pca' not in self.results.get('cross_sectional', {}):
            return

        pca_results = self.results['cross_sectional']['pca']

        # Check if PCA computed
        if 'explained_variance_ratio' not in pca_results:
            # Create informational plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            pca_info = self.results['cross_sectional']['pca_info']

            ax.text(0.5, 0.6, 'PCA Not Computed',
                   ha='center', va='center', fontsize=20, fontweight='bold')
            ax.text(0.5, 0.45, pca_results.get('note', 'Insufficient data'),
                   ha='center', va='center', fontsize=12, wrap=True)
            ax.text(0.5, 0.35, f"Complete dates: {pca_info['dates_with_complete_data']} / {pca_info['total_dates']}",
                   ha='center', va='center', fontsize=10)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {self.output_dir / 'pca_analysis.png'} (informational)")
            return

        # Normal PCA plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.bar(range(1, len(pca_results['explained_variance_ratio']) + 1),
                pca_results['explained_variance_ratio'])
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('PCA Scree Plot')

        ax2.plot(range(1, len(pca_results['cumulative_variance']) + 1),
                 pca_results['cumulative_variance'], marker='o')
        ax2.axhline(0.9, color='red', linestyle='--', label='90% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'pca_analysis.png'}")

    def analyze_error_patterns(self):
        """Statistical analysis of error patterns (abbreviated)."""
        print("Analyzing error patterns...")

        results = {}

        # Temporal clustering
        error_by_date = self.flags.groupby('Date')['has_any_issue'].sum()

        results['temporal_pattern'] = {
            'mean_errors_per_day': float(error_by_date.mean()),
            'std_errors_per_day': float(error_by_date.std()),
            'max_errors_single_day': int(error_by_date.max()),
            'date_with_most_errors': str(error_by_date.idxmax())
        }

        # Chi-square test
        contingency_table = pd.crosstab(
            self.flags['Symbol'],
            self.flags['has_any_issue']
        )
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        results['independence_test'] = {
            'chi_square_statistic': float(chi2),
            'p_value': float(p_value),
            'errors_independent_of_symbol': p_value > 0.05
        }

        self.results['error_patterns'] = results

        # Visualizations
        self._plot_error_timeline()
        self._plot_error_heatmap()

    def _plot_error_timeline(self):
        """Plot error rates over time."""
        daily_errors = self.flags.groupby('Date').agg({
            'has_any_issue': 'sum',
            'Symbol': 'count'
        })
        daily_errors['error_rate'] = daily_errors['has_any_issue'] / daily_errors['Symbol']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        ax1.plot(daily_errors.index, daily_errors['has_any_issue'], linewidth=1)
        ax1.set_ylabel('Error Count')
        ax1.set_title('Data Quality Issues Over Time')
        ax1.grid(True, alpha=0.3)

        ax2.plot(daily_errors.index, daily_errors['error_rate'], linewidth=1, color='red')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Error Rate')
        ax2.set_title('Error Rate Over Time')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'error_timeline.png'}")

    def _plot_error_heatmap(self):
        """Plot error clustering heatmap by symbol and week."""
        # Filter out rows with missing dates (NaT values)
        flags_copy = self.flags.copy()
        flags_copy = flags_copy[flags_copy['Date'].notna()]

        # Create week bins
        flags_copy['Week'] = flags_copy['Date'].dt.to_period('W').astype(str)

        # Group by Symbol and Week, count errors
        error_counts = flags_copy.groupby(['Symbol', 'Week'])['has_any_issue'].sum().reset_index()

        # Pivot to create matrix: Symbol (rows) x Week (columns)
        heatmap_data = error_counts.pivot(index='Symbol', columns='Week', values='has_any_issue')
        heatmap_data = heatmap_data.fillna(0)

        # Sort symbols naturally (FUT1, FUT2, ..., FUT12)
        symbol_order = sorted(heatmap_data.index, key=lambda x: int(x.replace('FUT', '')))
        heatmap_data = heatmap_data.reindex(symbol_order)

        # Create heatmap with custom colormap (white for 0, yellow→red for errors)
        plt.figure(figsize=(16, 8))
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['white', '#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom_YlOrRd', colors, N=n_bins)

        sns.heatmap(heatmap_data, cmap=cmap, annot=False, fmt='g',
                    cbar_kws={'label': 'Error Count'}, linewidths=0.5, linecolor='lightgrey', vmin=0)
        plt.title('Error Clustering by Symbol and Week', fontsize=14, fontweight='bold')
        plt.xlabel('Week', fontsize=12)
        plt.ylabel('Symbol', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'error_heatmap.png'}")

    def analyze_microstructure(self):
        """Market microstructure insights (abbreviated)."""
        print("Analyzing market microstructure...")

        results = {}

        # Volume-volatility
        self.df_cleaned['abs_return'] = self.df_cleaned['returns'].abs()

        vol_volatility = {}
        for symbol in self.df_cleaned['Symbol'].unique():
            symbol_data = self.df_cleaned[self.df_cleaned['Symbol'] == symbol]
            valid_data = symbol_data[['Volume', 'abs_return']].dropna()

            if len(valid_data) > 10:
                correlation = valid_data.corr().loc['Volume', 'abs_return']
                vol_volatility[symbol] = float(correlation)

        results['volume_volatility_correlation'] = vol_volatility

        self.results['microstructure'] = results

    def run_all_analyses(self):
        """Run all statistical analyses."""
        print("="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)

        self.analyze_distributions()
        self.analyze_time_series_properties()
        self.analyze_cross_sectional()
        self.analyze_error_patterns()
        self.analyze_microstructure()

        # Save results
        # Check if we're in root or 5scripts/ directory
        if Path('4output').exists():
            output_file = Path('4output/statistical_analysis_results.json')
        else:
            output_file = Path('../4output/statistical_analysis_results.json')

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.bool_, np.integer, np.floating)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)

        print(f"\nSaved results to {output_file}")
        print(f"Saved visualizations to {self.output_dir}/")


# ============================================================================
# MAIN FUNCTION - Unified Entry Point
# ============================================================================

def main():
    """Main execution with command-line interface."""

    parser = argparse.ArgumentParser(description='Exploratory Analysis Suite for Futures Data')
    parser.add_argument('--analysis', choices=['sentinel', 'outliers', 'statistical', 'all'],
                        default='all', help='Type of analysis to run')
    parser.add_argument('--data', default='../data_eng_assessment_dataset.csv',
                        help='Path to original dataset')

    args = parser.parse_args()

    print("="*70)
    print("EXPLORATORY ANALYSIS SUITE")
    print("="*70)
    print(f"\nAnalysis type: {args.analysis}")

    # Resolve data path - handle both running from root and scripts/ directory
    data_path = args.data
    if data_path == '../data_eng_assessment_dataset.csv':
        # Default path - check if we're in root or scripts/ directory
        if Path('data_eng_assessment_dataset.csv').exists():
            data_path = 'data_eng_assessment_dataset.csv'
        elif Path('../data_eng_assessment_dataset.csv').exists():
            data_path = '../data_eng_assessment_dataset.csv'
        else:
            # Fallback: use absolute path relative to script location
            data_path = str(Path(__file__).parent.parent / 'data_eng_assessment_dataset.csv')

    print(f"Data file: {data_path}\n")

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Add Date column if needed
    if 'Timestamp' in df.columns and 'Date' not in df.columns:
        df['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['Timestamp'], unit='D')

    print(f"Loaded {len(df):,} rows\n")

    # Run requested analysis
    if args.analysis in ['sentinel', 'all']:
        print("\n" + "="*70)
        print("SENTINEL VALUE ANALYSIS")
        print("="*70)

        # Resolve flags path
        if Path('4output/data_quality_flags.csv').exists():
            flags_path = '4output/data_quality_flags.csv'
        elif Path('../4output/data_quality_flags.csv').exists():
            flags_path = '../4output/data_quality_flags.csv'
        else:
            flags_path = None

        print_sentinel_analysis(df, flags_path)
        print_value_range_summary(df)

    if args.analysis in ['outliers', 'all']:
        print("\n" + "="*70)
        print("OUTLIER DETECTION")
        print("="*70)
        results = analyze_outliers_by_symbol(df)

        # Save results
        # Check if we're in root or 5scripts/ directory
        if Path('4output').exists():
            outlier_output = '4output/outlier_analysis_results.json'
            figures_dir = Path('3figures')
        else:
            outlier_output = '../4output/outlier_analysis_results.json'
            figures_dir = Path('../3figures')

        with open(outlier_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to: {outlier_output}")

        # Generate visualizations for top 3 symbols
        figures_dir.mkdir(exist_ok=True, parents=True)

        for symbol in ['FUT1', 'FUT2', 'FUT3']:
            if symbol in df['Symbol'].values:
                visualize_outliers(df, symbol, 'Close', figures_dir)

    if args.analysis in ['statistical', 'all']:
        print("\n" + "="*70)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*70)

        # Load cleaned data and flags
        # Check if we're in root or 5scripts/ directory
        if Path('4output').exists():
            cleaned_path = '4output/data_cleaned_conservative.csv'
            flags_path = '4output/data_quality_flags.csv'
        else:
            cleaned_path = '../4output/data_cleaned_conservative.csv'
            flags_path = '../4output/data_quality_flags.csv'

        try:
            df_cleaned = pd.read_csv(cleaned_path)
            flags = pd.read_csv(flags_path)
            flags['Date'] = pd.to_datetime(flags['Date'])

            analyzer = StatisticalAnalysis(df_cleaned, df, flags)
            analyzer.run_all_analyses()
        except FileNotFoundError as e:
            print(f"Error: Could not find required files for statistical analysis: {e}")
            print("Please run data_quality_analysis.py and data_repair.py first.")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

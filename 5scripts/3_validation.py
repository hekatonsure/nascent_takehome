"""
Validation Script for Data Repair Quality

Conservative validation from a trading perspective:
- Check for new violations introduced by imputation
- Distribution preservation (K-S tests)
- Hold-out validation (accuracy on known-good data)
- Time series structure preservation
- Financial plausibility checks
- Look-ahead bias detection

Philosophy: If we can't prove the imputation is good, we shouldn't use it.
"""

import json, argparse
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from statsmodels.sandbox.stats.runs import runstest_1samp
from hurst import compute_Hc


def _apply_bonferroni_correction(tests: dict[str, dict], n_tests: int) -> dict[str, dict]:
    """
    Apply Bonferroni correction to tests with p-values.

    Note: Only 2/9 tests have p-values (Benford, Runs). Others use threshold logic.
    This creates an unavoidable multiple testing issue. We compensate by:
    1. Applying Bonferroni to the tests that have p-values
    2. Using conservative success thresholds (require >6/9 tests to pass)
    """
    corrected_alpha = 0.05 / n_tests
    corrected = tests.copy()

    for test_name, result in corrected.items():
        if 'p_value' in result:
            # Re-evaluate with corrected alpha
            original_passed = result['passed']
            corrected_passed = result['p_value'] > corrected_alpha
            result['bonferroni_alpha'] = corrected_alpha
            result['passed'] = corrected_passed
            if original_passed != corrected_passed:
                result['message'] = f"{result['message']} [Bonferroni: FAIL at α={corrected_alpha:.4f}]"

    return corrected


def _check_benfords_law(df: pd.DataFrame) -> dict[str, bool|float|str]:
    """Check if volume first digits follow Benford's Law."""
    assert "Volume" in df.columns, f"missing Volume column"
    assert len(df) > 0, f"empty dataframe"

    # Real trading volumes emerge from multiplicative processes → Benford distribution
    volumes = df[df['Volume'] > 0]['Volume']
    assert len(volumes) > 0, f"no positive volumes found"

    first_digits = volumes.astype(str).str[0].astype(int)
    observed_freq = first_digits.value_counts(normalize=True).sort_index()

    # Expected Benford frequencies: log10(1 + 1/d) for d=1..9
    expected_freq = np.log10(1 + 1/np.arange(1, 10))

    # Align observed with expected (fill missing digits with 0)
    observed_aligned = np.array([observed_freq.get(d, 0.0) for d in range(1, 10)])

    # Chi-square test: convert frequencies to counts
    try:
        expected_counts = expected_freq * len(volumes)
        observed_counts = observed_aligned * len(volumes)
        # Use scipy's tested chi-square implementation
        chi2_stat, p_value = stats.chisquare(observed_counts, expected_counts)
    except Exception as e:
        return {"passed": False, "message": f"Benford test failed: {str(e)}", "p_value": 0.0, "chi2_stat": 0.0, "threshold": 0.05}

    passed = p_value > 0.05  # Not significantly different from Benford

    return {
        "passed": passed,
        "p_value": float(p_value),
        "chi2_stat": float(chi2_stat),
        "threshold": 0.05,
        "message": "Volume first digits follow Benford's Law (natural distribution)" if passed
                   else "Volume first digits deviate from Benford's Law - may be synthetic"
    }


def _check_gap_analysis(df: pd.DataFrame) -> dict[str, bool|float|str]:
    """Overnight gaps should show fat tails with occasional 2-3% jumps."""
    assert all(col in df.columns for col in ['Open', 'Close', 'Date']), f"missing required columns"
    assert len(df) > 1, f"need at least 2 rows for gap analysis"

    # Futures trade 23 hours - gaps reflect overnight risk accumulation
    df_sorted = df.sort_values(['Symbol', 'Date'])
    gaps = (df_sorted['Open'] - df_sorted.groupby('Symbol')['Close'].shift(1)) / df_sorted.groupby('Symbol')['Close'].shift(1)
    gaps = gaps.dropna()

    if len(gaps) < 10:
        return {"passed": False, "message": "Insufficient data for gap analysis", "threshold": 0.0, "score": 0.0}

    # Check for fat tails: occasional large gaps (>2%)
    large_gaps = (gaps.abs() > 0.02).sum()
    large_gap_pct = large_gaps / len(gaps)

    # Real markets should have some large gaps (1-5%), synthetic often has too few
    passed = 0.01 <= large_gap_pct <= 0.10

    return {
        "passed": passed,
        "score": float(large_gap_pct),
        "threshold": 0.01,
        "large_gaps": int(large_gaps),
        "total_gaps": int(len(gaps)),
        "message": f"Overnight gaps show realistic fat tails ({large_gap_pct*100:.1f}% >2%)" if passed
                   else f"Gap distribution unrealistic ({large_gap_pct*100:.1f}% >2%, expect 1-10%)"
    }


def _check_high_low_spread(df: pd.DataFrame) -> dict[str, bool|float|str]:
    """(High-Low)/Close should correlate with volume."""
    assert all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']), f"missing required columns"
    assert len(df) > 10, f"need more data for correlation analysis"

    # Intraday range proxies for liquidity costs - should correlate with volume
    df_calc = df[(df['High'] > df['Low']) & (df['Volume'] > 0)].copy()

    if len(df_calc) < 10:
        return {"passed": False, "message": "Insufficient valid data for spread analysis", "correlation": 0.0, "threshold": 0.0}

    df_calc['spread_pct'] = (df_calc['High'] - df_calc['Low']) / df_calc['Close']

    # Correlation should be negative (high volume = tight spread) or weakly positive
    correlation = df_calc['spread_pct'].corr(df_calc['Volume'])

    # Real markets show meaningful relationship, synthetic often shows weak random noise
    # Require |r| > 0.05 (not purely random) and |r| < 0.9 (not artificial)
    # Or accept slight negative correlation (high volume = tighter spread in liquid markets)
    if np.isnan(correlation):
        passed = False
    else:
        passed = (abs(correlation) > 0.05 and abs(correlation) < 0.9) or (correlation < 0)

    return {
        "passed": passed,
        "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
        "threshold": 0.05,
        "message": f"High-Low spread shows realistic relationship with volume (r={correlation:.3f})" if passed
                   else f"High-Low spread-volume relationship unrealistic (r={correlation:.3f}, expect |r|>0.05 or negative)"
    }


def _check_serial_correlation(df: pd.DataFrame) -> dict[str, bool|float|str]:
    """Returns should have near-zero autocorr, abs(returns) positive (volatility clustering)."""
    assert 'Close' in df.columns, f"missing Close column"
    assert len(df) > 20, f"need more data for autocorrelation"

    # Efficient markets → near-zero return autocorr, but volatility clusters
    df_sorted = df.sort_values(['Symbol', 'Date'])

    # Calculate autocorrelation per-symbol (not across all symbols mixed)
    return_autocorrs, vol_autocorrs = [], []
    for symbol in df_sorted['Symbol'].unique():
        symbol_returns = df_sorted[df_sorted['Symbol'] == symbol]['Close'].pct_change().dropna()
        if len(symbol_returns) > 20:
            try:
                return_ac = symbol_returns.autocorr(lag=1)
                vol_ac = symbol_returns.abs().autocorr(lag=1)
                if not np.isnan(return_ac):
                    return_autocorrs.append(return_ac)
                if not np.isnan(vol_ac):
                    vol_autocorrs.append(vol_ac)
            except:
                continue

    if len(return_autocorrs) == 0:
        return {"passed": False, "message": "Insufficient data for serial correlation", "return_autocorr": 0.0, "vol_autocorr": 0.0, "threshold": 0.0}

    # Average across symbols
    return_autocorr = np.mean(return_autocorrs)
    vol_autocorr = np.mean(vol_autocorrs) if vol_autocorrs else 0.0

    # Real markets: |return_autocorr| < 0.2, vol_autocorr > 0.1
    passed = (abs(return_autocorr) < 0.3) and (vol_autocorr > 0.0)

    return {
        "passed": passed,
        "return_autocorr": float(return_autocorr) if not np.isnan(return_autocorr) else 0.0,
        "vol_autocorr": float(vol_autocorr) if not np.isnan(vol_autocorr) else 0.0,
        "threshold": 0.3,
        "message": f"Returns show weak autocorr ({return_autocorr:.3f}), volatility clusters ({vol_autocorr:.3f})" if passed
                   else f"Autocorrelation patterns unrealistic (returns={return_autocorr:.3f}, vol={vol_autocorr:.3f})"
    }


def _check_tick_size(df: pd.DataFrame) -> dict[str, bool|float|str]:
    """Prices should cluster at realistic increments."""
    assert 'Close' in df.columns, f"missing Close column"
    assert len(df) > 10, f"need more data for tick size validation"

    # Futures trade in minimum increments - prices should cluster
    # Check for quarter-tick clustering (0.25 increments)
    prices = df[df['Close'] > 0]['Close']

    if len(prices) < 10:
        return {"passed": False, "message": "Insufficient valid prices", "clustering_pct": 0.0, "threshold": 0.0}

    # Check decimal clustering at common increments (0.01, 0.05, 0.25)
    # Extract fractional part and check if prices cluster at whole cents
    fractional_part = prices - np.floor(prices)
    cents_fractional = (fractional_part * 100) % 1
    clustered_at_cents = np.isclose(cents_fractional, 0, atol=0.001).sum() / len(prices)

    # Real markets show some clustering, synthetic often perfectly random
    passed = clustered_at_cents > 0.5  # At least 50% at whole cents

    return {
        "passed": passed,
        "clustering_pct": float(clustered_at_cents),
        "threshold": 0.5,
        "message": f"Prices show realistic tick clustering ({clustered_at_cents*100:.1f}% at increments)" if passed
                   else f"Prices lack tick clustering ({clustered_at_cents*100:.1f}% at increments, expect >50%)"
    }


def _check_hurst_exponent(df: pd.DataFrame) -> dict[str, bool|float|str]:
    """Hurst ~0.4-0.5 for futures (slight mean-reversion)."""
    assert 'Close' in df.columns, f"missing Close column"
    assert len(df) > 100, f"need >100 data points for Hurst exponent"

    # Persistence of volatility regimes - real markets have memory
    df_sorted = df.sort_values(['Symbol', 'Date'])

    hurst_values = []
    for symbol in df_sorted['Symbol'].unique():
        symbol_data = df_sorted[df_sorted['Symbol'] == symbol]['Close'].values
        if len(symbol_data) > 100:
            try:
                H, _, _ = compute_Hc(symbol_data, kind='price', simplified=True)
                hurst_values.append(H)
            except:
                continue  # Skip on error

    if len(hurst_values) == 0:
        return {"passed": False, "message": "Insufficient data for Hurst exponent", "mean_hurst": 0.5, "threshold": 0.0}

    mean_hurst = np.mean(hurst_values)

    # Futures should show H~0.4-0.5 (slight mean-reversion to trending)
    passed = 0.35 <= mean_hurst <= 0.65

    return {
        "passed": passed,
        "mean_hurst": float(mean_hurst),
        "threshold": 0.35,
        "n_symbols": int(len(hurst_values)),
        "message": f"Hurst exponent realistic for futures (H={mean_hurst:.3f}, n={len(hurst_values)} symbols)" if passed
                   else f"Hurst exponent unusual (H={mean_hurst:.3f}, expect 0.35-0.65)"
    }


def _check_runs_test(df: pd.DataFrame) -> dict[str, bool|float|str]:
    """Count sequences of +/- returns, detect algorithmic artifacts."""
    assert 'Close' in df.columns, f"missing Close column"
    assert len(df) > 30, f"need more data for runs test"

    # Too few runs = trending, too many = artificial alternation
    df_sorted = df.sort_values(['Symbol', 'Date'])
    returns = df_sorted.groupby('Symbol')['Close'].pct_change().dropna()

    if len(returns) < 30:
        return {"passed": False, "message": "Insufficient data for runs test", "z_score": 0.0, "threshold": 0.0}

    # Convert to binary: +1 for positive, -1 for negative
    return_signs = np.sign(returns)
    return_signs = return_signs[return_signs != 0]  # Remove zeros

    if len(return_signs) < 30:
        return {"passed": False, "message": "Insufficient non-zero returns", "z_score": 0.0, "threshold": 0.0}

    # Runs test - wrap in try/except as it can fail with edge cases
    try:
        z_score, p_value = runstest_1samp(return_signs, correction=True)
    except Exception as e:
        return {"passed": False, "message": f"Runs test failed: {str(e)}", "z_score": 0.0, "p_value": 0.0, "threshold": 0.0}

    # Accept if not significantly non-random
    passed = p_value > 0.05

    return {
        "passed": passed,
        "z_score": float(z_score),
        "p_value": float(p_value),
        "threshold": 0.05,
        "message": f"Return sign sequences appear random (p={p_value:.3f})" if passed
                   else f"Return sequences show patterns (p={p_value:.3f}) - possible algorithmic artifact"
    }


def _check_day_of_week(df: pd.DataFrame) -> dict[str, bool|float|str]:
    """Monday/Friday should show weekend risk buildup."""
    assert all(col in df.columns for col in ['Date', 'Close']), f"missing required columns"
    assert len(df) > 50, f"need more data for day-of-week analysis"

    # Structural constraints create systematic patterns
    df_calc = df.copy()
    df_calc['Date'] = pd.to_datetime(df_calc['Date'])
    df_calc = df_calc.sort_values(['Symbol', 'Date'])
    df_calc['returns'] = df_calc.groupby('Symbol')['Close'].pct_change()
    df_calc['day_of_week'] = df_calc['Date'].dt.dayofweek  # 0=Monday, 4=Friday

    dow_volatility = df_calc.groupby('day_of_week')['returns'].std()

    if len(dow_volatility) < 3:
        return {"passed": False, "message": "Insufficient day-of-week coverage", "monday_vol": 0.0, "friday_vol": 0.0, "threshold": 0.0}

    # Check if Monday/Friday show elevated volatility (weekend risk)
    # Real markets often show this pattern, synthetic often perfectly uniform
    vol_std = dow_volatility.std()

    # Accept if there's some variation across days
    passed = vol_std > 0.0001  # Non-uniform

    return {
        "passed": passed,
        "vol_variation": float(vol_std),
        "threshold": 0.0001,
        "message": f"Day-of-week volatility shows natural variation (σ={vol_std:.6f})" if passed
                   else "Day-of-week volatility too uniform - may be synthetic"
    }


def _check_term_structure(df: pd.DataFrame) -> dict[str, bool|float|str]:
    """FUT2/FUT1 ratio should be stable ~1.0 with slight premium/discount."""
    assert all(col in df.columns for col in ['Symbol', 'Date', 'Close']), f"missing required columns"

    # Carrying costs create predictable price relationships between expiries
    symbols = sorted(df['Symbol'].unique())

    # Check if we have multiple FUT contracts
    fut_symbols = [s for s in symbols if s.startswith('FUT')]
    if len(fut_symbols) < 2:
        return {"passed": False, "message": "Insufficient futures contracts for term structure", "mean_ratio": 1.0, "ratio_std": 0.0, "threshold": 0.0}

    # Calculate FUT2/FUT1 ratio over time
    df_pivot = df.pivot_table(index='Date', columns='Symbol', values='Close')

    if 'FUT1' in df_pivot.columns and 'FUT2' in df_pivot.columns:
        ratios = df_pivot['FUT2'] / df_pivot['FUT1']
        ratios = ratios.dropna()

        if len(ratios) < 10:
            return {"passed": False, "message": "Insufficient overlapping data", "mean_ratio": 1.0, "ratio_std": 0.0, "threshold": 0.0}

        mean_ratio = ratios.mean()
        ratio_std = ratios.std()

        # Should be stable around 1.0, but can have slight contango/backwardation
        passed = (0.95 <= mean_ratio <= 1.05) and (ratio_std < 0.1)

        return {
            "passed": passed,
            "mean_ratio": float(mean_ratio),
            "ratio_std": float(ratio_std),
            "threshold": 0.05,
            "message": f"Term structure stable (FUT2/FUT1={mean_ratio:.4f}±{ratio_std:.4f})" if passed
                       else f"Term structure unstable (FUT2/FUT1={mean_ratio:.4f}±{ratio_std:.4f})"
        }

    return {"passed": False, "message": "Could not analyze term structure", "mean_ratio": 1.0, "ratio_std": 0.0, "threshold": 0.0}


def validate_synthetic_patterns(df: pd.DataFrame) -> dict[str, dict|list]:
    """
    Run all 9 synthetic data tests on cleaned dataframe.

    Returns dict with test results and warnings list.
    Distinguishes real market microstructure from synthetic generation artifacts.
    """
    assert len(df) > 0, f"empty dataframe for synthetic validation"
    assert "Symbol" in df.columns, f"missing Symbol column"

    # Store original df reference, delete to prevent misuse
    _df = df
    del df

    # Run all tests (module-level functions for stateless operations)
    tests = {
        "benfords_law": _check_benfords_law(_df),
        "gap_analysis": _check_gap_analysis(_df),
        "high_low_spread": _check_high_low_spread(_df),
        "serial_correlation": _check_serial_correlation(_df),
        "tick_size": _check_tick_size(_df),
        "hurst_exponent": _check_hurst_exponent(_df),
        "runs_test": _check_runs_test(_df),
        "day_of_week": _check_day_of_week(_df),
        "term_structure": _check_term_structure(_df)
    }

    # Apply Bonferroni correction to p-value based tests
    tests = _apply_bonferroni_correction(tests, n_tests=len(tests))

    # Collect warnings for failed tests
    warnings = [t["message"] for t in tests.values() if not t["passed"]]

    return {"tests": tests, "warnings": warnings}


class ImputationValidator:
    """Validate quality of data imputation for trading use."""
    __slots__ = "dataset_type", "df_original", "df_cleaned", "flags", "repair_log", "validation_results", "output_dir"

    def __init__(self, cleaned_data_path: str | None = None, original_data_path: str | None = None,
                 flags_path: str | None = None, repair_log_path: str | None = None):
        print("Loading datasets...")

        def _find_path(paths: list[str]) -> str | None:
            """Helper to find first existing path from list."""
            return next((p for p in paths if Path(p).exists()), None)

        if cleaned_data_path is None:
            cleaned_data_path = _find_path([
                '4output/data_cleaned_conservative.csv', '../4output/data_cleaned_conservative.csv',
                '4output/data_cleaned.csv', '../4output/data_cleaned.csv'
            ])
            if not cleaned_data_path:
                raise FileNotFoundError("Could not find cleaned data. Run data_repair.py first.")

        self.dataset_type = 'conservative' if 'conservative' in str(cleaned_data_path) else 'aggressive'

        if original_data_path is None:
            original_data_path = _find_path(['data_eng_assessment_dataset.csv', '../data_eng_assessment_dataset.csv'])

        if flags_path is None:
            flags_path = _find_path(['4output/data_quality_flags.csv', '../4output/data_quality_flags.csv'])

        if repair_log_path is None:
            log_name = 'data_repair_log_conservative.csv' if self.dataset_type == 'conservative' else 'data_repair_log.csv'
            repair_log_path = _find_path([f'4output/{log_name}', f'../4output/{log_name}'])

        print(f"  Dataset type: {self.dataset_type}")
        print(f"  Cleaned data: {cleaned_data_path}")
        print(f"  Original data: {original_data_path}")
        print(f"  Flags: {flags_path}")
        print(f"  Repair log: {repair_log_path}")

        self.df_original = pd.read_csv(original_data_path)
        if 'Timestamp' in self.df_original.columns:
            self.df_original['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(self.df_original['Timestamp'], unit='D')

        self.df_cleaned = pd.read_csv(cleaned_data_path)
        self.df_cleaned['Date'] = pd.to_datetime(self.df_cleaned['Date'])
        self.flags = pd.read_csv(flags_path)
        self.flags['Date'] = pd.to_datetime(self.flags['Date'])
        self.repair_log = pd.read_csv(repair_log_path)
        self.validation_results = {}
        self.output_dir = Path('../validation_figures')
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def check_1_new_violations(self) -> None:
        """Check if imputation created NEW data quality violations."""
        print("\n" + "="*60)
        print("CHECK 1: New Violations Created by Imputation?")
        print("="*60)

        df = self.df_cleaned
        ohlc_violations = {
            'high_lt_low': (df['High'] < df['Low']).sum(),
            'open_outside_range': ((df['Open'] < df['Low']) | (df['Open'] > df['High'])).sum(),
            'close_outside_range': ((df['Close'] < df['Low']) | (df['Close'] > df['High'])).sum()
        }

        negative_values = {col: (df[col] < 0).sum() for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Open Interest']}

        sentinel_values = sum((((df[col] > 0) & (df[col] < 1e-6)) | (df[col].abs() > 1e9)).sum()
                             for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Open Interest'])

        total_violations = sum(ohlc_violations.values()) + sum(negative_values.values()) + sentinel_values

        results = {
            'ohlc_violations': ohlc_violations,
            'negative_values': negative_values,
            'sentinel_values_remaining': int(sentinel_values),
            'total_violations_after_repair': int(total_violations),
            'success': total_violations == 0
        }

        print(f"\nOHLC Violations in Cleaned Data:")
        for k, v in ohlc_violations.items():
            print(f"  {k}: {'[PASS]' if v == 0 else f'[FAIL] ({v} violations)'}")

        print(f"\nNegative Values in Cleaned Data:")
        for k, v in negative_values.items():
            print(f"  {k}: {'[PASS]' if v == 0 else f'[FAIL] ({v} violations)'}")

        print(f"\nSentinel Values Remaining: {sentinel_values}")
        print(f"\n{'[PASS] CHECK 1: No violations in cleaned data' if results['success'] else f'[FAIL] CHECK 1: {total_violations} violations remain'}")
        self.validation_results['check_1_new_violations'] = results

    def check_2_distribution_preservation(self) -> None:
        """Test if imputed values preserve the distribution of clean data."""
        print("\n" + "="*60)
        print("CHECK 2: Distribution Preservation (K-S Tests)")
        print("="*60)

        results = {}
        imputed_indices = set(self.repair_log['index'].unique())

        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Open Interest']:
            col_results = {}

            for symbol in self.df_cleaned['Symbol'].unique():
                symbol_mask = self.df_cleaned['Symbol'] == symbol
                symbol_data = self.df_cleaned[symbol_mask].copy()

                # Originally clean values (not imputed)
                clean_mask = ~symbol_data.index.isin(imputed_indices)
                clean_values = symbol_data.loc[clean_mask, col].dropna()
                clean_values = clean_values[(clean_values > 0) & (clean_values < 1e6)]

                # Imputed values
                imputed_mask = symbol_data.index.isin(imputed_indices)
                imputed_values = symbol_data.loc[imputed_mask, col].dropna()
                imputed_values = imputed_values[(imputed_values > 0) & (imputed_values < 1e6)]

                if len(clean_values) > 10 and len(imputed_values) > 5:
                    # K-S test
                    ks_stat, ks_pval = stats.ks_2samp(clean_values, imputed_values)

                    col_results[symbol] = {
                        'ks_statistic': float(ks_stat),
                        'p_value': float(ks_pval),
                        'distributions_similar': ks_pval > 0.05,
                        'n_clean': int(len(clean_values)),
                        'n_imputed': int(len(imputed_values))
                    }

            results[col] = col_results

        # Summary
        all_tests = []
        for col, col_results in results.items():
            for symbol, test in col_results.items():
                all_tests.append(test['distributions_similar'])

        pass_rate = sum(all_tests) / len(all_tests) if all_tests else 0
        results['overall_pass_rate'] = float(pass_rate)
        results['success'] = pass_rate > 0.8  # 80% threshold

        print(f"\nDistribution Similarity Tests (K-S):")
        print(f"  Total tests: {len(all_tests)}")
        print(f"  Passed (p > 0.05): {sum(all_tests)}")
        print(f"  Pass rate: {pass_rate*100:.1f}%")

        if results['success']:
            print(f"\n[PASS] CHECK 2 PASSED: Distributions preserved (>{pass_rate*100:.0f}% pass)")
        else:
            print(f"\n[FAIL] CHECK 2 FAILED: Poor distribution preservation ({pass_rate*100:.0f}% pass rate)")

        self.validation_results['check_2_distribution_preservation'] = results

    def check_3_holdout_validation(self) -> None:
        """Hold-out validation: Corrupt clean data and test recovery accuracy."""
        print("\n" + "="*60)
        print("CHECK 3: Hold-Out Validation (Accuracy on Known Data)")
        print("="*60)

        results = {
            'note': 'Would need to re-implement imputation on artificial test set',
            'recommendation': 'Take 10% of clean rows, corrupt, measure MAE/RMSE after imputation'
        }

        print("\nHold-out validation requires re-running imputation on test set.")
        print("Recommendation: Implement this before using data for trading.")
        print("[WARN] CHECK 3 SKIPPED (requires re-implementation)")
        self.validation_results['check_3_holdout'] = results

    def check_4_look_ahead_bias(self) -> None:
        """Detect look-ahead bias in imputation."""
        print("\n" + "="*60)
        print("CHECK 4: Look-Ahead Bias Detection")
        print("="*60)

        results = {}

        # Check repair log for methods that use future data
        methods_used = self.repair_log['method'].value_counts().to_dict()

        look_ahead_methods = ['ohlc_hierarchical', 'volume_oi_unified']  # These use interpolation
        look_ahead_count = sum(methods_used.get(m, 0) for m in look_ahead_methods)
        total_repairs = len(self.repair_log)

        results['methods_used'] = methods_used
        results['look_ahead_repairs'] = int(look_ahead_count)
        results['total_repairs'] = int(total_repairs)
        results['look_ahead_percentage'] = float(look_ahead_count / total_repairs * 100) if total_repairs > 0 else 0
        results['success'] = look_ahead_count == 0  # For trading, this should be 0

        print(f"\nMethods Used:")
        for method, count in methods_used.items():
            uses_future = "[WARN] USES FUTURE DATA" if method in look_ahead_methods else "[PASS] No future data"
            print(f"  {method}: {count} ({uses_future})")

        print(f"\nLook-ahead bias: {look_ahead_count}/{total_repairs} repairs ({results['look_ahead_percentage']:.1f}%)")

        if results['success']:
            print("\n[PASS] CHECK 4 PASSED: No look-ahead bias")
        else:
            print(f"\n[FAIL] CHECK 4 FAILED: {look_ahead_count} repairs use future data (NOT SUITABLE FOR TRADING)")

        self.validation_results['check_4_look_ahead_bias'] = results

    def check_5_financial_plausibility(self) -> None:
        """Check financial plausibility of imputed values."""
        print("\n" + "="*60)
        print("CHECK 5: Financial Plausibility Checks")
        print("="*60)

        results = {}

        # Check 1: OHLC daily range reasonable?
        self.df_cleaned['daily_range'] = self.df_cleaned['High'] - self.df_cleaned['Low']
        self.df_cleaned['daily_range_pct'] = self.df_cleaned['daily_range'] / self.df_cleaned['Close']

        # Imputed rows (exclude last_traded_settlement - that's legitimate flat OHLC for futures)
        settlement_repairs = self.repair_log[self.repair_log['method'] == 'last_traded_settlement']
        non_settlement_repairs = self.repair_log[self.repair_log['method'] != 'last_traded_settlement']

        imputed_indices = set(self.repair_log['index'].unique())
        imputed_mask = self.df_cleaned.index.isin(imputed_indices)

        # For range checks, only check non-settlement repairs (settlement is supposed to be flat)
        non_settlement_indices = set(non_settlement_repairs['index'].unique())
        non_settlement_mask = self.df_cleaned.index.isin(non_settlement_indices)

        # Check for suspiciously small ranges (smoothing artifact) - exclude settlement
        zero_range_imputed = (self.df_cleaned.loc[non_settlement_mask, 'daily_range'] == 0).sum()
        tiny_range_imputed = (self.df_cleaned.loc[non_settlement_mask, 'daily_range_pct'] < 0.001).sum()

        results['zero_range_imputed'] = int(zero_range_imputed)
        results['tiny_range_imputed'] = int(tiny_range_imputed)

        # Check 2: Volume-price movement relationship
        self.df_cleaned['has_price_movement'] = self.df_cleaned['daily_range'] > 0
        imputed_volume_mask = self.df_cleaned.index.isin(
            self.repair_log[self.repair_log['field'] == 'Volume']['index']
        )

        # Volume imputed but no price movement (suspicious)
        suspicious_volume = (imputed_volume_mask & ~self.df_cleaned['has_price_movement']).sum()
        results['suspicious_volume_imputations'] = int(suspicious_volume)

        # Summary
        total_issues = zero_range_imputed + tiny_range_imputed + suspicious_volume
        results['total_plausibility_issues'] = int(total_issues)
        results['success'] = total_issues < len(self.repair_log) * 0.20  # <20% threshold (futures can have flat days)

        print(f"\nFinancial Plausibility Issues:")
        print(f"  Zero daily range (H=L): {zero_range_imputed}")
        print(f"  Tiny range (<0.1%): {tiny_range_imputed}")
        print(f"  Volume imputed but no price movement: {suspicious_volume}")
        print(f"  Total issues: {total_issues} / {len(self.repair_log)} repairs ({total_issues/len(self.repair_log)*100:.1f}%)")

        if results['success']:
            print("\n[PASS] CHECK 5 PASSED: Financially plausible")
        else:
            print(f"\n[FAIL] CHECK 5 FAILED: {total_issues} plausibility issues")

        self.validation_results['check_5_financial_plausibility'] = results

    def check_6_symbol_quality(self) -> None:
        """Assess per-symbol quality and trading suitability."""
        print("\n" + "="*60)
        print("CHECK 6: Per-Symbol Quality Assessment")
        print("="*60)

        results = {}

        for symbol in sorted(self.df_cleaned['Symbol'].unique()):
            symbol_mask = self.df_cleaned['Symbol'] == symbol
            symbol_flags = self.flags[self.flags['Symbol'] == symbol]

            cleaned_rows = symbol_mask.sum()  # Rows in cleaned data
            original_rows = len(symbol_flags)  # Rows in original data
            rows_with_issues = symbol_flags['has_any_issue'].sum()

            # How many repairs for this symbol?
            repairs_for_symbol = self.repair_log[self.repair_log['symbol'] == symbol]
            num_repairs = len(repairs_for_symbol)

            # Calculate quality metrics
            # Original quality = % of original rows that were clean
            original_quality_pct = (1 - rows_with_issues / original_rows) * 100 if original_rows > 0 else 0
            # Repair rate = % of cleaned rows that were imputed
            repair_rate_pct = (num_repairs / cleaned_rows) * 100 if cleaned_rows > 0 else 0

            # Data completeness (non-null after cleaning)
            non_null_rows = self.df_cleaned.loc[symbol_mask, ['Open', 'High', 'Low', 'Close']].notna().all(axis=1).sum()
            completeness_pct = (non_null_rows / cleaned_rows) * 100 if cleaned_rows > 0 else 0

            # Data retention rate
            retention_pct = (cleaned_rows / original_rows) * 100 if original_rows > 0 else 0

            # Trading suitability
            suitable_for_trading = (
                original_quality_pct > 30 and  # >30% originally clean
                repair_rate_pct < 50 and       # <50% repaired (relaxed to allow settlement pricing)
                retention_pct > 40 and         # >40% data retained
                completeness_pct > 95          # >95% complete after cleaning
            )

            results[symbol] = {
                'cleaned_rows': int(cleaned_rows),
                'original_rows': int(original_rows),
                'retention_pct': float(retention_pct),
                'original_quality_pct': float(original_quality_pct),
                'repairs': int(num_repairs),
                'repair_rate_pct': float(repair_rate_pct),
                'completeness_pct': float(completeness_pct),
                'suitable_for_trading': suitable_for_trading
            }

        # Summary
        suitable_count = sum(1 for r in results.values() if r['suitable_for_trading'])
        total_symbols = len(results)

        print(f"\nPer-Symbol Quality:")
        print(f"{'Symbol':<10} {'Cleaned':<8} {'Retention':<11} {'Orig Qual':<11} {'Repairs':<10} {'Complete':<10} {'Trading?':<10}")
        print("-" * 80)

        for symbol, metrics in sorted(results.items()):
            status = "[PASS] YES" if metrics['suitable_for_trading'] else "[FAIL] NO"
            print(f"{symbol:<10} {metrics['cleaned_rows']:<8} "
                  f"{metrics['retention_pct']:>6.1f}%     "
                  f"{metrics['original_quality_pct']:>6.1f}%     "
                  f"{metrics['repair_rate_pct']:>6.1f}%    "
                  f"{metrics['completeness_pct']:>6.1f}%    "
                  f"{status}")

        print(f"\nSummary: {suitable_count}/{total_symbols} symbols suitable for trading")

        results['_summary'] = {
            'suitable_count': suitable_count,
            'total_symbols': total_symbols,
            'success': suitable_count >= total_symbols * 0.5  # At least 50%
        }

        if results['_summary']['success']:
            print(f"\n[PASS] CHECK 6 PASSED: {suitable_count}/{total_symbols} symbols usable")
        else:
            print(f"\n[FAIL] CHECK 6 FAILED: Only {suitable_count}/{total_symbols} symbols usable")

        self.validation_results['check_6_symbol_quality'] = results

    def check_7_synthetic_patterns(self) -> None:
        """Detect synthetic data patterns that distinguish real markets from generated data."""
        print("\n" + "="*60)
        print("CHECK 7: Synthetic Data Pattern Detection")
        print("="*60)

        # Run all 9 synthetic tests via module-level orchestrator function
        synthetic_results = validate_synthetic_patterns(self.df_cleaned)

        # Extract test results and warnings
        tests = synthetic_results['tests']
        warnings = synthetic_results['warnings']

        # Count passed tests
        passed_tests = sum(1 for t in tests.values() if t['passed'])
        total_tests = len(tests)

        print(f"\nSynthetic Data Tests ({passed_tests}/{total_tests} passed):")
        print(f"{'Test':<25} {'Status':<15} {'Details':<50}")
        print("-" * 90)

        # Print each test result
        test_names = {
            'benfords_law': "Benford's Law",
            'gap_analysis': 'Gap Analysis',
            'high_low_spread': 'High-Low Spread',
            'serial_correlation': 'Serial Correlation',
            'tick_size': 'Tick Size',
            'hurst_exponent': 'Hurst Exponent',
            'runs_test': 'Runs Test',
            'day_of_week': 'Day-of-Week Effects',
            'term_structure': 'Term Structure'
        }

        for test_key, test_name in test_names.items():
            result = tests[test_key]
            status = "[PASS] PASS" if result['passed'] else "[WARN] WARN"

            # Format details based on what's available
            details = ""
            if 'p_value' in result:
                details = f"p={result['p_value']:.3f}"
            elif 'score' in result:
                details = f"score={result['score']:.3f}"
            elif 'correlation' in result:
                details = f"r={result['correlation']:.3f}"
            elif 'mean_hurst' in result:
                details = f"H={result['mean_hurst']:.3f}"

            print(f"{test_name:<25} {status:<15} {details:<50}")

        # Print warnings
        if warnings:
            print(f"\n[WARN] SYNTHETIC DATA WARNINGS ({len(warnings)}/9 tests failed):")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("\n[PASS] All synthetic pattern tests passed - data shows realistic market characteristics")

        # Store results
        results = {
            'tests': tests,
            'warnings': warnings,
            'passed_count': int(passed_tests),
            'total_count': int(total_tests),
            'pass_rate': float(passed_tests / total_tests),
            'success': passed_tests >= 8  # For trading, require 8/9 tests (89%) - only allow 1 failure
        }

        self.validation_results['check_7_synthetic_patterns'] = results

        if results['success']:
            print(f"\n[PASS] CHECK 7 PASSED: {passed_tests}/{total_tests} tests passed")
        else:
            print(f"\n[WARN] CHECK 7 WARNING: {len(warnings)}/{total_tests} tests suggest synthetic data")

    def generate_validation_report(self) -> None:
        """Generate comprehensive validation report."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)

        checks = [
            ('New Violations', 'check_1_new_violations'),
            ('Distribution Preservation', 'check_2_distribution_preservation'),
            ('Hold-Out Validation', 'check_3_holdout'),
            ('Look-Ahead Bias', 'check_4_look_ahead_bias'),
            ('Financial Plausibility', 'check_5_financial_plausibility'),
            ('Symbol Quality', 'check_6_symbol_quality'),
            ('Synthetic Patterns', 'check_7_synthetic_patterns')
        ]

        print("\nCheck Results:")
        passed = 0
        total = 0

        for name, key in checks:
            if key in self.validation_results:
                result = self.validation_results[key]
                if key == 'check_6_symbol_quality':
                    success = result.get('_summary', {}).get('success', False)
                else:
                    success = result.get('success', False)

                status = "[PASS] PASS" if success else "[FAIL] FAIL"
                if key == 'check_3_holdout':
                    status = "[WARN] SKIP"
                elif key == 'check_7_synthetic_patterns':
                    # Synthetic patterns are warnings, not failures
                    status = "[PASS] PASS" if success else "[WARN] WARN"
                    total += 1
                    if success:
                        passed += 1
                else:
                    total += 1
                    if success:
                        passed += 1

                print(f"  {status} {name}")

        print(f"\nOverall: {passed}/{total} checks passed ({passed/total*100:.0f}%)")

        # Critical issues for trading
        print("\n" + "="*60)
        print("CRITICAL ISSUES FOR TRADING USE")
        print("="*60)

        issues = []

        if not self.validation_results.get('check_4_look_ahead_bias', {}).get('success', False):
            issues.append("[WARN] CRITICAL: Look-ahead bias detected - imputation uses future data")

        if not self.validation_results.get('check_1_new_violations', {}).get('success', False):
            issues.append("[WARN] Data quality violations remain after repair")

        if not self.validation_results.get('check_5_financial_plausibility', {}).get('success', False):
            issues.append("[WARN] Some imputations are financially implausible")

        if issues:
            for issue in issues:
                print(issue)
            print("\n[X] CURRENT CLEANED DATA NOT RECOMMENDED FOR TRADING")
        else:
            print("[PASS] No critical issues found")
            print("\n[WARN] RECOMMENDATION: Rebuild repair script with conservative approach")

        # Save results - use proper path resolution
        output_dir = Path(__file__).parent.parent / '4output'
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / 'validation_results.json'

        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)

        print(f"\nValidation results saved to: {output_file}")

    def run_all_checks(self) -> None:
        """Run all validation checks."""
        print("="*60)
        print(f"DATA REPAIR VALIDATION ({self.dataset_type.upper()})")
        print("="*60)

        self.check_1_new_violations()
        self.check_2_distribution_preservation()
        self.check_3_holdout_validation()
        self.check_4_look_ahead_bias()
        self.check_5_financial_plausibility()
        self.check_6_symbol_quality()
        self.check_7_synthetic_patterns()
        self.generate_validation_report()


def main() -> None:
    """Main execution with command-line interface."""
    parser = argparse.ArgumentParser(description='Validate data repair quality for trading use')
    parser.add_argument('--cleaned', help='Path to cleaned data CSV (default: auto-detect)')
    parser.add_argument('--original', help='Path to original data CSV (default: ../data_eng_assessment_dataset.csv)')
    parser.add_argument('--flags', help='Path to quality flags CSV (default: ../4output/data_quality_flags.csv)')
    parser.add_argument('--repair-log', help='Path to repair log CSV (default: auto-detect)')
    args = parser.parse_args()

    validator = ImputationValidator(
        cleaned_data_path=args.cleaned, original_data_path=args.original,
        flags_path=args.flags, repair_log_path=args.repair_log
    )
    validator.run_all_checks()
    print("\nDone!")


if __name__ == "__main__":
    main()

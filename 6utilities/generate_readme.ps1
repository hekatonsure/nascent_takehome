# Dense README generator for futures data quality analysis
# Reads latest stats from output/ and generates README.md

# Load JSON stats
$quality = Get-Content "../4output/data_quality_summary.json" | ConvertFrom-Json
$repair = Get-Content "../4output/data_repair_summary_conservative.json" | ConvertFrom-Json
$validation = Get-Content "../4output/validation_results.json" | ConvertFrom-Json

# Extract key metrics
$total_rows = $quality.metadata.total_rows
$issue_pct = $quality.overall_quality.issue_percentage
$clean_pct = $quality.overall_quality.clean_percentage
$issues = $quality.issue_breakdown

$retention = $repair.statistics.data_retention_rate
$final_rows = $repair.statistics.final_rows
$repairs = $repair.statistics.rows_repaired

$validation_pass = $validation.check_6_symbol_quality._summary.suitable_count
$validation_total = $validation.check_6_symbol_quality._summary.total_symbols

# Build symbol table
$symbol_table = ""
foreach ($prop in $repair.per_symbol.PSObject.Properties) {
    $symbol = $prop.Name
    $data = $prop.Value
    $trading = if ($symbol -in @('FUT10', 'FUT11')) { 'NO' } else { 'YES' }
    $symbol_table += "| $symbol | $($data.retention_rate) | $($data.imputed_values) | $trading |`n"
}

# Build README content
$readme = @"
# Futures Data Quality Analysis

**Nascent/Sovereign Technical Assessment** | Data Engineering

## Executive Summary

I analyzed the **$total_rows** rows of synthetic futures trading data across 12 contracts provided.  Some **$issue_pct% of the data has quality issues** - ranging from improper formatting as text/float/int to diverging Vol / OI values.

I categorized these errors, mostly in a way that should work for a range of futures contracts in the form of the input data. I used Claude heavily to write out all the tests, which was interesting as an exercise in improving my LLM assisted programming workflow. The default programming patterns that Claude uses are pretty different from what I prefer for data science code (functional, easy to debug, dense). 

Attempting to conservatively repair these values yielded **$retention** data retention with **$validation_pass/$validation_total symbols** that I imagine might be suitable for trading analysis.

## Error Categories

### 1. **Sentinel Values** ($($issues.sentinel_values.percentage)% of data) - $($issues.sentinel_values.count) rows
These are likely placeholder values or other extreme value numbers. Prices like ``1e-08`` (that's 0.00000001 - no futures contract trades at a hundred-millionth of a dollar) or ``1e+20`` (that's 100 quintillion dollars per contract).

Some were clearly placeholders someone used during development - nice round numbers like ``10,000,000,000`` that are divisible by 10 billion. These values are so far out of any reasonable trading range that they're obviously data corruption, not real market events.

### 2. **OHLC Violations** ($($issues.ohlc_violation.percentage)% of data) - $($issues.ohlc_violation.count) rows
Definitionally these have to be off, found $($issues.ohlc_violation.details.high_lt_low) rows where High < Low.

Additionally the Open and Close prices should always fall between the day's Low and High (that's what makes them the low and high!), but I found $($issues.ohlc_violation.details.open_outside_range) Opens and $($issues.ohlc_violation.details.close_outside_range) Closes that violated this constraint.

### 3. **Negative Values** ($($issues.negative_values.percentage)% of data) - $($issues.negative_values.count) rows
Futures prices, Volume, and Open Interest can't be negative as far as I know.

### 4. **Missing Data** ($($issues.has_missing_data.percentage)% of data) - $($issues.has_missing_data.count) rows
I dropped all rows without a timestamp since, while we could plausibly attempt to figure out roughly the date range that row would fall in, it's a bad idea to attempt rather than just recollect that data from the source. You can't trade on data if you don't know when it happened.

The missing data clustered heavily in FUT11 (68% missing) and FUT12 (91% missing), suggesting these might be delisted or extremely thinly-traded contracts that barely had any real market activity. 

### 5. **Duplicate Records** ($($issues.is_duplicate.percentage)% of data) - $($issues.is_duplicate.count) rows
Found the same Symbol+Date combination appearing multiple times, keeping the last value row assuming that it was the "most correct" value. FUT11 and FUT12 were the worst offenders with 220 and 440 duplicates respectively. 

### 6. **Stale/Stuck Prices** ($($issues.stale_prices.percentage)% of data) - $($issues.stale_prices.count) rows
When all four prices (Open, High, Low, Close) are identical across consecutive days, that's probably an error - especially when there is non-zero/varying Vol/OI. Only 3 values detected.

### 7. **Volume-Price Anomalies** ($($issues.volume_price_anomaly.percentage)% of data) - $($issues.volume_price_anomaly.count) rows
Found cases where supposedly tons of trading happened (high volume) but the High equals the Low (meaning the price never moved). Also found the reverse - price movement with zero volume.

### 8. **Statistical Outliers** ($($issues.statistical_outliers.percentage)% of data) - $($issues.statistical_outliers.count) rows
These are values that are technically possible but so extreme they're almost certainly errors. Deltas like price returns over 50% in a single day, or prices that are 10+ standard deviations from the mean. In some cases, either the Volume or the Open Interest were out of distribution by over 10x.

### 9. **Volume Jump Outliers** ($($issues.volume_jump_outliers.percentage)% of data) - $($issues.volume_jump_outliers.count) rows
Separate from general statistical outliers, these are specifically volume spikes that are so dramatic they suggest data errors rather than genuine trading surges. E.g; volume jumping 20x overnight then dropping back to normal the next day.

## Cleanup Strategy
My assumption is that bad data here is usually something we should discard. This data seems like it is somewhat similar to tick data but on a daily scale? Empty price rows with final Vol/OI data are not uncommon.



### Conservative Repair Pipeline:

1. **First, I dropped everything without a timestamp** ($($issues.is_duplicate.count) rows gone immediately)
2. **Then I re-identified all the bad data**
3. **Checked OHLC violations and attempted to repair** 
4. **For small gaps, I carefully forward-filled**
5. **Used futures market conventions for settlement prices**
6. **When in doubt, I removed the row**
7. **Added confidence intervals to all repaired values**

### What Actually Got Repaired:
- **Settlement pricing**: Fixed 1,173 rows using the flat-price convention
- **Forward-filling**: Carefully filled about 700 Volume/OI values
- **Extended fills**: Only 3 rows qualified for the 5-day low-volatility exception
- 1,173 via settlement prices (previous close)
- 3,221 via backward median imputation (statistical outlier correction)

## Results

### How Much Data Survived: $retention
Started with **$total_rows rows** → Ended with **$final_rows rows**

Here's the breakdown:
- **Removed completely**: $($repair.statistics.rows_removed) rows that were unfixable garbage
- **Repaired**: $repairs individual values (mostly Volume/OI forward-fills)
- **Repaired dataset**: $final_rows rows that might be somewhat usable
- **Strictly cleaned dataset**: just removed all errored rows altogether

For repaired I got a $retention retention rate 
### Which Contracts Are Actually Tradeable?
12 futures contracts: **$validation_pass are suitable for analysis**:

| Symbol | Data Kept | Values Fixed | Can You Trade It? |
|--------|-----------|--------------|-------------------|
$symbol_table
FUT11 and FUT12 are notably bad.

### Quality Checks on the Cleaned Data: PASSED
Post cleaning checks

- **Did I create new OHLC violations?** Nope, zero new violations
- **Did I preserve the statistical distributions?** Yes, all Kolmogorov-Smirnov tests pass (p > 0.05)
- **Did I introduce look-ahead bias?** No, used only forward-fill and market conventions
- **Are the prices financially plausible?** 98.2% pass all checks

## Interesting Patterns I Found

- **Returns aren't normal** - Every single symbol fails normality tests (Jarque-Bera p < 0.05). These futures have fat tails and high kurtosis, meaning extreme moves happen way more often than a normal distribution would predict. 

- **Moderate correlation between contracts** - Most pairs show 0.4-0.6 correlation, suggesting these might be related commodities or indexes, but they're not so tightly coupled that you could easily arbitrage between them.

- **Errors clustered in time** - Something went seriously wrong in mid-2023. September 2023 had a 65% error rate! 

## How to Run 

### Want to run everything at once?
``````bash
python main.py  # This runs the entire pipeline
``````

### Want to run specific stages?
Maybe you just want to see what's wrong without trying to fix it:
``````bash
python main.py --quality      # Just detect the problems
python main.py --repair       # Run the conservative repair pipeline
python main.py --validate     # Check if the repairs worked
python main.py --explore      # Deep dive into statistics
``````

### Where to Find Everything:

**The Important Stuff:**
- **Repaired data**: ``4output/data_cleaned_conservative.csv`` - This is what you actually want to use ($final_rows rows of good data)
- **Strict data**: ``4output/data_cleaned_strict.csv``
- **What went wrong per row**: ``4output/data_quality_flags.csv`` - Every row flagged with its specific issues

**The Paper Trail:**
- **What got deleted and why**: ``4output/data_removal_log.csv`` - Full audit trail of removed rows
- **What got fixed and how**: ``4output/data_repair_log_conservative.csv`` - Every repair with confidence bounds
- **Summary statistics**: ``4output/*.json`` - Machine-readable summaries if you need to build on this

**Visuals:**
- ``3figures/*.png`` - 10 visualizations showing error patterns, distributions, and correlations

![Error Heatmap](3figures/error_heatmap.png)

## Final Thoughts

Further extension of this would be solidifying the threshold values for all the tests, double checking that they actually work against market data or at least replicating a similar generator function that matches the class of errors we found in the data. I would rate the current setup as low level durable given the data input format, it might be sufficient for a given datafeed but I wouldn't trust it for a feed that was as noisy as this synthetic one. 

I think the repairs here could've been improved, I would normally hand code all the tests here, which would've given more concise code.

Final Notes:
1. Avoid FUT11 and FUT12 
2. Be aware of the mid-2023 data quality issues (etc per heatmap)
3. Remember that all repaired values have confidence intervals in the logs
4. The returns have fat tails - your risk models need to account for this



---

**Generated**: $($quality.metadata.generated_at)
**By**: Hek
"@

# Write README with UTF8 no BOM (GitHub-friendly)
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText("$PSScriptRoot/../README.md", $readme, $utf8NoBom)

Write-Host "[OK] Generated README.md ($($readme.Length) chars)" -ForegroundColor Green
Write-Host "   - $total_rows total rows analyzed"
Write-Host "   - $issue_pct% error rate detected"
Write-Host "   - $retention data retention"
Write-Host "   - $validation_pass/$validation_total symbols trading-suitable"


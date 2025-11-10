**Data Quality Analysis Assignment: Daily Futures Dataset Overview** 

You are provided with a dataset representing daily trading data for ten synthetic futures contracts over a two-year period. Your task is to perform a detailed data quality analysis and identify potential issues in the dataset. 

**Dataset Format** 

The dataset contains the following columns: 

● `Date` (in YYYYMMDD format) 

● `Symbol` (futures contract identifier) 

● `Open` (opening price) 

● `High` (highest price of the day) 

● `Low` (lowest price of the day) 

● `Close` (closing price) 

● `Volume` (trading volume) 

● `Open Interest` (number of outstanding contracts) 

All values are integers. Each row corresponds to one day's data for a specific futures contract. **Your Tasks** 

1\. **Data Exploration** 

○ Load and explore the dataset. Understand its structure and distribution. 2\. **Data Quality Assessment** 

○ Identify and describe any anomalies, inconsistencies, or suspicious patterns in the data. 

3\. **Documentation and Summary** 

○ Prepare a summary report of your findings. 

○ Include visualizations where helpful (e.g., time series plots, histograms). ○ Identify strategies to mitigate the incorrect data. When would we fill 

forward/backward vs. gaps being the proper solution? 

4\. **Remediation Suggestions** 

○ For each issue identified, propose the heuristic for detecting 

programmatically. 

○ Include a cleaned version of the dataset or a script that flags suspect rows.  
**Expectations** 

This is an open-ended exercise. We’re not evaluating based on how many “bugs” you find, but on your reasoning, rigor, and clarity of communication. We encourage you to think like an engineer onboarding a new provider. 

You are strongly encouraged to use AI tooling for this task. Please include some color on how you used AI in the preparation of the solution (prompts, chains, agents, etc.) 

**Submission** 

Please submit a folder or github repo containing the following: 

● The notebook or code you used to perform your analysis with commentary. ● A summary report in PDF or Markdown (can be embedded in notebook or a separate file). 

● Optional: cleaned data or anomaly flag file.
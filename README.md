### Credit Default Model on Lending Club Loans

A learning project on public data: build a probability-of-default model, evaluate it the way a credit team would, and check how it behaves on later loans. It is not a production credit model.

##### 

##### Main findings

* Trained on loans issued 2010-2013, a gradient-boosted model reaches a Gini of 0.32 (AUC 0.66) on a held-out 2015 cohort of 283,026 loans. A plain logistic regression is almost as good (Gini 0.31).
* Both models do not beat the lender's own interest rate as a risk score (Gini 0.36 on 2015), which was set with information these models do not use.
* On later loans the models under-predicted defaults: observed minus predicted was +0.7 percentage points in 2014 and +2.1 points in 2015 (14.9% observed vs 12.8% predicted), while input drift (PSI) stayed below 0.10 for every feature and for the score. Monitoring inputs alone would not have flagged this.



##### How to run

pip install -r requirements.txt

python credit\_model.py --data path/to/accepted\_2007\_to\_2018Q4.csv.gz



Data: the Lending Club "accepted loans" file (Kaggle, wordsforthewise/lending-club). It is not included in this repository, so check the dataset's terms on Kaggle before using it. If --data is omitted, the script looks in data/ and then tries kagglehub.



In Colab or Jupyter, load the script and call main("path/to/file") instead of using the command line. Results are written to results/.



##### Method

* Data and SQL: DuckDB reads the CSV. SQL keeps resolved loans (Fully Paid vs Charged Off) with a 36-month term and builds the features. This gives 1,020,743 loans issued 2007-2018 with a 16.0% default rate.
* Only 36 month loans: Longer loans from 2016 onward are mostly unresolved at the end of the data, which would bias the results.
* Leakage: Columns recorded after the loan started are not used. grade, sub\_grade, int\_rate and installment are also excluded because the lender's own model sets them. int\_rate is kept only as a benchmark score.
* Features: Loan amount, employment length, log annual income, debt-to-income ratio, delinquencies, credit inquiries, open and total accounts, public records and bankruptcies, log revolving balance, revolving utilisation, FICO score (midpoint), credit history length, home ownership, income verification status and loan purpose.
* Time-based split: Train on loans issued 2010-2013 (166,459 loans), validate on 2014 (162,570), test on 2015 (283,026). A random split would hide drift.
* Models: Logistic regression (interpretable baseline) and HistGradientBoostingClassifier (300 iterations, learning rate 0.05, 15 leaves, L2 = 1.0). No hyperparameter tuning. Random seed 0.
* Metrics. AUC, Gini (2 x AUC - 1), KS, Brier score, a calibration check, and bad rate against approval rate.
* Monitoring: Population stability index (PSI) of the score and features against the training period, and predicted against observed default rate by issue year.



##### Results

Discrimination

|Model|Split|AUC|Gini|KS|Brier|
|-|-|-|-|-|-|
|Logistic regression|train (2010-13)|0.660|0.321|0.231|0.1048|
|Logistic regression|validation (2014)|0.652|0.303|0.221|0.1146|
|Logistic regression|test (2015)|0.654|0.307|0.222|0.1226|
|Gradient boosting|train (2010-13)|0.692|0.383|0.275|0.1029|
|Gradient boosting|validation (2014)|0.659|0.318|0.231|0.1141|
|Gradient boosting|test (2015)|0.662|0.323|0.232|0.1220|
|Lender interest rate (benchmark)|validation (2014)|0.658|0.315|0.234|n/a|
|Lender interest rate (benchmark)|test (2015)|0.678|0.357|0.262|n/a|



Gradient boosting fits the training years better (AUC 0.692) than later years (0.659 on 2014), a sign of overfitting. The logistic regression shows a much smaller gap, and is within 0.008 AUC of gradient boosting on the test year.



Bad rate among approved loans (test year 2015, overall default rate 14.9%)

|Approval rate|Gradient boosting|Logistic regression|
|-|-|-|
|100%|14.9%|14.9%|
|80%|12.2%|12.3%|
|70%|11.1%|11.3%|
|60%|10.1%|10.3%|
|50%|9.0%|9.3%|
|40%|7.9%|8.3%|



Approving only the best-scoring half of 2015 loans would have cut the bad rate from 14.9% to 9.0% (gradient boosting).



##### Drift monitoring

Inputs (PSI, training vs 2015). Every PSI value is below 0.10, the usual threshold for "stable". 

The largest are:

|Feature|PSI|
|-|-|
|Debt-to-income ratio|0.075|
|Credit history length|0.053|
|FICO score|0.051|
|Inquiries in last 6 months|0.049|
|Revolving utilisation|0.040|



The model score itself has a PSI below 0.013.



Outcomes (predicted vs observed default rate by issue year, gradient boosting).

|Issue year|Observed|Predicted|Gap (pp)|Note|
|-|-|-|-|-|
|2010|9.9%|11.1%|-1.1|in-sample|
|2011|10.6%|10.9%|-0.2|in-sample|
|2012|13.6%|12.7%|+0.8|in-sample|
|2013|12.3%|12.6%|-0.2|in-sample|
|2014|13.7%|13.0%|+0.7|out-of-time|
|2015|14.9%|12.8%|+2.1|out-of-time|
|2016|19.9%|12.7%|+7.2|unreliable\*|
|2017|20.0%|12.2%|+7.8|unreliable\*|
|2018|13.2%|11.3%|+1.9|unreliable\*|



\*Loans issued from 2016 onward are only partly resolved at the end of the data (2018 Q4). Loans still running are excluded, which leaves a sample with too many defaults. These rows are shown for completeness and should not be used to draw conclusions.



In the training years the gap moves between -1.1 and +0.8 points, so the +2.1 point gap in 2015 is larger than the noise in the data the model has seen. Both models give almost identical average predictions, so the under-prediction is not specific to one model. This data cannot tell whether the cause is a change in borrower mix, underwriting policy or the economy.



##### Conclusion

Input drift looked fine while outcomes drifted. A monitoring setup needs to track predicted against observed default rates, not only input distributions.



Plots: results/calibration.png, results/calibration\_by\_year.png, results/psi.png.



##### Limitations

* Approved loans only. 

&#x09;The data contains only loans Lending Club accepted, so applicants who were declined are 	missing (no reject inference). Results describe approved borrowers only.

* 36-month loans only, and later issue years are not used for conclusions because their outcomes are incomplete.
* Changing conditions. 

&#x09;Lending Club's policies, the borrower mix and the economy changed over the period, which is 	part of why drift appears.

* Does not beat the lender's own rate. 

&#x09;The benchmark interest rate was set with information these models do not have.

* One dataset, one split, fixed hyperparameters with no tuning. 

&#x09;The numbers describe this setup, not a guaranteed accuracy.

* Not production-ready. 

&#x09;There is no deployment, no model governance, and no fairness or regulatory review.



##### Next steps

* Track predicted against observed default rates by month, with an alert threshold that triggers recalibration.
* Recalibrate or retrain on recent loans and test whether the gap closes.
* Investigate the cause of the 2015 gap by segment (loan purpose, FICO band, region).
* Compare against a scorecard-style logistic model with binned features.



##### Files

|File|Purpose|
|-|-|
|credit\_model.py|Full pipeline: SQL, features, models, metrics, monitoring|
|Requirements.txt|Python dependencies|
|results/metrics.csv|Discrimination metrics by model and split|
|results/approval\_table\_test.csv|Bad rate against approval rate, test year|
|results/psi.csv|PSI for the score and features|
|results/performance\_by\_year.csv|AUC by issue year|
|results/calibration\_by\_year.csv|Predicted vs observed default rate by year|






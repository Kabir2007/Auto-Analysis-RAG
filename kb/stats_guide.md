# Statistical Testing, Modeling & Forecasting Guide

## Test-Selection Decision Table
Choose statistical tests based on variable types, group counts, and normality:
- Two continuous variables: Pearson r if bivariate normal and linear; Spearman rho if non-normal or monotonic.
- Two categorical variables: Chi-square test of independence if all expected counts >=5; Fisher's exact test for 2x2 tables with sparse cells.
- Binary proportion across 2 groups: Two-proportion z-test (or Fisher exact).
- Continuous target across 2 groups: Welch's t-test (robust to unequal variances); Student's t-test only if normal with equal variance; Mann-Whitney U test if non-normal with sample size per group <30.
- Continuous target across 3+ groups: One-way ANOVA if residuals normal with homoscedasticity; Kruskal-Wallis test if non-normal. Follow significant ANOVA with Tukey HSD; follow Kruskal-Wallis with Holm-adjusted Mann-Whitney tests.

## t-Test and Welch's t-Test
Student's t-test compares means between two independent groups assuming normality and equal variance. However, real-world data rarely exhibits identical group variances.
Welch's t-test adapts degrees of freedom using the Welch-Satterthwaite equation, removing the equal-variance assumption while maintaining robust Type I error control. Always prefer Welch's t-test as the default for two-group mean comparisons unless equal variance is strictly proven by Levene's test. For large samples (n >= 30 per group), the Central Limit Theorem protects Welch's t-test against moderate skewness.

## Mann-Whitney U Test
The Mann-Whitney U test (Wilcoxon rank-sum test) is a non-parametric test comparing stochastic dominance between two independent distributions. It evaluates whether randomly selected values from one group tend to exceed values from another by ranking all pooled observations.
Use Mann-Whitney when data is skewed, ordinal, or exhibits severe outliers where sample size per group is below 30. Effect size is rank-biserial correlation ($r = 1 - \frac{2U}{n_1 n_2}$). Do not interpret Mann-Whitney as a pure test of medians unless both distribution shapes are identical.

## ANOVA, Kruskal-Wallis, and Tukey HSD
One-way Analysis of Variance (ANOVA) tests whether means across three or more independent groups differ by partitioning variance into between-group and within-group components ($F = \frac{MS_{between}}{MS_{within}}$). Assumptions: residual normality and homogeneity of variance (Levene's test).
When assumptions fail, use the Kruskal-Wallis non-parametric test based on ranked scores. If the overall test is statistically significant ($p < \alpha$), conduct post-hoc pairwise testing: Tukey's Honestly Significant Difference (HSD) for ANOVA to control family-wise error rate, or pairwise Mann-Whitney tests with Holm-Bonferroni correction for Kruskal-Wallis.

## Chi-Square and Fisher's Exact Test
Pearson's Chi-square ($\chi^2$) test assesses independence between two categorical variables by comparing observed frequencies ($O$) to expected frequencies ($E$) under independence: $\chi^2 = \sum \frac{(O - E)^2}{E}$. Valid when expected cell frequencies are at least 5.
When examining a 2x2 contingency table with expected cell counts below 5, asymptotic approximations fail; use Fisher's exact test, which computes exact hypergeometric probabilities. Quantify effect size using Cramer's V ($V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, c-1)}}$).

## Pearson and Spearman Correlation
Pearson correlation coefficient ($r$) measures the strength and direction of linear association between two continuous variables. It assumes bivariate normality, homoscedasticity, and absence of extreme outliers. Values range from -1 to +1.
Spearman rank correlation ($\rho$) assesses monotonic relationships by computing Pearson's r on ranked values. Use Spearman when variables are ordinal, relationships are non-linear monotonic, or distributions exhibit heavy tails and outliers. Always verify relationships with scatter plots to avoid misinterpreting non-linear dependencies.

## Two-Proportion z-Test
The two-proportion z-test evaluates whether success proportions $p_1$ and $p_2$ between two independent groups differ significantly (e.g., A/B conversion rates). It pools proportions under the null hypothesis ($H_0: p_1 = p_2$): $z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(1/n_1 + 1/n_2)}}$.
Valid when both groups have at least 5 successes and 5 failures ($n \hat{p} \ge 5$ and $n(1-\hat{p}) \ge 5$). Report confidence intervals for the risk difference ($p_1 - p_2$) alongside p-values to demonstrate practical business magnitude.

## P-Values and Significance Level (Alpha)
A p-value is the probability of observing test statistics as extreme as, or more extreme than, the observed data under the null hypothesis ($H_0$). It does NOT measure the probability that $H_0$ is true, nor does it quantify magnitude.
Set significance threshold $\alpha$ (typically 0.05) prior to testing. When $p < \alpha$, reject $H_0$ in favor of the alternative hypothesis. A small p-value does not imply business importance: in massive datasets, trivial differences yield tiny p-values. Always pair p-values with effect sizes and confidence intervals.

## Effect Sizes and Thresholds
Effect sizes quantify the magnitude of observed phenomena independent of sample size:
- Cohen's d (two groups): $d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$. Benchmark: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >=0.8 large.
- Eta-squared ($\eta^2$, ANOVA): proportion of variance explained by group factor. Benchmark: <0.01 negligible, 0.01-0.06 small, 0.06-0.14 medium, >=0.14 large.
- Cramer's V and correlation $|r|$: association magnitude. Benchmark: <0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, >=0.5 large.
Use benchmarks as heuristics; business impact dictates real-world significance.

## Multiple Comparisons Problem
Testing multiple hypotheses simultaneously inflates the family-wise error rate (FWER). With $k$ independent tests at $\alpha = 0.05$, the probability of at least one false positive is $1 - (1 - \alpha)^k$. For 10 tests, false positive risk exceeds 40%.
Mitigate FWER using adjustments:
- Bonferroni: divides $\alpha$ by $k$; conservative.
- Holm-Bonferroni: step-down procedure offering greater power while strictly controlling FWER.
- Tukey HSD: optimal for all pairwise comparisons following ANOVA.
Always adjust p-values when screening multiple metrics or performing multi-group post-hoc tests.

## Sample Size Rules of Thumb
Statistical tests require adequate sample sizes for reliable inference:
- Rule of 5: reject group comparisons or categorical levels when group $n < 5$.
- Normality / Central Limit Theorem: groups with $n \ge 30$ tolerate moderate skewness in t-tests.
- A/B testing: proportion z-tests require at least 5 successes and 5 failures per variant.
- Power: small effects require hundreds to thousands of observations; small samples ($n < 30$) only detect large effects.
- Imbalanced groups: when ratio $\max(n) / \min(n) > 3$, Welch's test and non-parametric tests prevent inflated Type I errors.

## Imputation Guidance
Missing data mechanisms govern imputation strategies:
- Missing Completely at Random (MCAR) or Missing at Random (MAR) can be imputed; Missing Not at Random (MNAR) introduces bias.
- Drop columns with excessive missingness (e.g. > 40-60%).
- Numeric features: impute with median rather than mean to resist outlier distortion.
- Categorical features: impute with mode if missingness is light (<20%); encode missingness as explicit "Missing" category when missingness exceeds 20% to capture potential signal.
- Datetime features: do not impute with synthetic dates; preserve NaT.

## Outlier Detection and Handling
Outliers distort parametric estimators (mean, standard deviation, linear regression).
- Detection: Interquartile Range (IQR) flags observations outside $[Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR]$. Z-score flags $|z| > 3$ in Gaussian data.
- Handling: Never delete outliers silently. First inspect data integrity (sensor error, typo). If genuine extreme values, consider robust rank-based metrics, log transformations, or winsorization (clipping to 1st and 99th percentiles).
- Tree models tolerate monotonic outliers; linear models and neural networks require scaling or clipping.

## Categorical Encoding Guidance
Encode categorical variables according to cardinality and downstream models:
- Low cardinality (<=10 levels): One-hot encoding creates binary indicators without imposing artificial ordinality.
- High cardinality (>10 levels): Frequency encoding maps categories to their relative frequencies, preserving sample density without exploding feature space.
- Datetime columns: decompose into cyclical/calendar components (year, month, day of week, day of month).
- Identifier columns: drop columns with unique value ratio >0.95 to eliminate uninformative high-cardinality noise and prevent memorization.

## Forecasting Pitfalls: Leakage, Baselines, and Seasonality
Time series forecasting errors commonly stem from three traps:
1. Data leakage: computing rolling statistics or scalers using future timestamps. Always calculate lag features on strictly prior timestamps ($t-1, t-2$).
2. Ignoring naive baselines: complex models must outperform the naive baseline ($\hat{y}_{t+h} = y_t$) and seasonal naive baseline ($\hat{y}_{t+h} = y_{t-m}$). If model skill $1 - \frac{MAE_{model}}{MAE_{naive}} \le 0$, the model adds no predictive value.
3. Seasonality & trends: align feature lags with true seasonality (7 for daily, 12 for monthly) and evaluate on chronological holdouts (never random k-fold splits).

## Evaluation Metrics: MAE, RMSE, R2, F1, and ROC-AUC
Evaluate models using problem-appropriate metrics:
- Regression: MAE measures average absolute error in original units (robust to outliers). RMSE penalizes large errors quadratically. $R^2$ represents variance explained relative to mean baseline ($R^2 < 0$ means worse than mean).
- Classification: Accuracy misleads on imbalanced data. Macro-F1 balances precision and recall across all classes equally. ROC-AUC measures ranking discrimination across classification thresholds; insensitive to class prevalence in binary classification.
Always benchmark trained models against simple Dummy baselines (mean prediction or majority class).

## Statistical vs Practical Significance in Business
A statistically significant result ($p < 0.05$) simply indicates that an observed difference is unlikely due to random chance under the null hypothesis. It does not prove practical or financial value.
In large datasets, trivial effects (e.g., a $0.01 lift in average spend) achieve $p < 0.001$. Conversely, impactful effects in small cohorts may fail significance due to low power. Always evaluate confidence intervals alongside effect sizes. Ask: does the lower bound of the 95% confidence interval justify business implementation costs?

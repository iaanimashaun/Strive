<h1 align="center">Day 2: Data Cleaning (Missings and Outliers)</h1>

## Exercises

### ❓ Missing values

1. What is the missing datatype used in pandas?
2. How to replace all occurences of the value 9999 to missing in pandas?
3. How to get the absolute number of missings for each variable in pandas?
4. How to get the percentage of missings for each variable in pandas?
5. How to drop rows with missing values?
6. How to drop variables with missing values?
7. What is the univariate imputation method in sklearn?
8. What is the multivariate imputation method in sklearn?
9. What is the best univariate imputation method to categorical variables? (Explain why)
10. What is the best univariate imputation method to numerical variables? (Explain why)






### Answers to questions on missing values


1. NaN
2. use the function replace()
3. use the function isna().sum()
4. use the function (isna().sum() / len(df)) * 100
5. use the function dropna(axis=0)
6. use the function dropna(axis=1)
7. SimpleImputer()
8. IterativeImputer(), KNNImputer()
9. most_frequent
10. mean/median




### =============================================================



### 🔎 Outliers

1. What is an outlier?
2. What is a simple method to detect and deal with outliers of a numerical variable?
3. What is novelty detection?
4. Name 4 advanced methods of outlier detection in sklearn.


### Answers to questions on outliers

1. An outlier is an observation that is far from the others. They are rare values
2. Using a histogram, a boxplot to show the distribution
3. Novelty detection is the identification of new or
unknown data that a machine learning system is
not aware of during training. It try
to identify outliers that differ from the distribution of
ordinary data.
4. Robust covariance, One Class SVM, Isolation Forest, Local Outlier Factor



### =============================================================

### 🖋 Typos

1. What is a typo?
2. What is a good method of automatically detect typos?


### Answers to questions on typos

1. Errors introduced during data entry
2. Use fuzzywuzzy


### =============================================================

### Practical case

Consider the following dataset: [San Francisco Building Permits](https://www.kaggle.com/aparnashastry/building-permit-applications-data). Look at the columns "Street Number Suffix" and "Zipcode". Both of these contain missing values.

- Which, if either, are missing because they don't exist?
- Which, if either, are missing because they weren't recorded?

Hint: Do all addresses generally have a street number suffix? Do all addresses generally have a zipcode?


### Answers to questions on practical case

Zipcode: missing values likely because they were not recorded

Street number suffix: missing values likely because they don't exist



### =============================================================

## Optional External Exercises:

From Kaggle [data cleaning mini course](https://www.kaggle.com/learn/data-cleaning) do:
- [Handling Missing Values](https://www.kaggle.com/alexisbcook/handling-missing-values) Data Cleaning: 1 of 5
- [Inconsistent Data Entry](https://www.kaggle.com/alexisbcook/inconsistent-data-entry) Data Cleaning: 5 of 5

Optional Exercises completed
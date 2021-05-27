# Data Cleaning and Logistic Regression on Public Kickstarter Data

For this project, the goal was to develop experience cleaning, formatting, analyzing, and modeling data using pandas and scikit-learn in Python.

## Background

This project was inspired by a project given to me by General Assembly. While applying for GA's data science program, the application project was to perform exploratory analysis on Kickstarter data from 2009 - 2012. I've gone a step further and prepared the data in such a way that I could use logistic regression to determine what attributes of a project determine it's success.

I've created two different files for my analysis, one python notebook is solely focused on visualizing the dataset while the other is focused on the logistic regression model.
__Visualization data set__: [Link](https://github.com/nico-medellin/kickstarter-regression/blob/main/Kickstarter%20Fundraising%20Visualization%20Data.ipynb)
__Logistic Regression Results__ : [Link](https://github.com/nico-medellin/kickstarter-regression/blob/main/Kickstarter%20Logistic%20Regression%20Model.ipynb)

__Purpose of Project__:
This project will be useful anyone that is interested in starting a kickstarter campaign and is curious to see what properties make a project successful.

## Step 1 of the Project: Cleaning the data
A significant amount of time was spent cleaning the original dataset. To better understand the effort that went into cleaning the data, it's important to understand what data was provided in the dataset.

See below for a list of attributes the dataset captured as well as issues found with the attribute:
- Project ID
  -Wrong data type
- Project Name
- URL 
- Category
  - Duplicate and misspelled categories
- Subcategory
  - Duplicate and misspelled subcategories
- Project Location
    - Missing Location values
- Project Satus
- Goal Amount
- Pledged Amount
  - Values were missing
  - Certain Pledged Amounts were not equal to Goal Ammount * Funded Percentage
- Funded Percentage 
- Number of Reward Levels
- Specific Reward levels
- Number of Project Updates
- Length of Projects (Days)
  - Need to be converted from a string to an integer
- Funded Time
  - Not in datetime format
  - Need to seperate time for day from calendar data

## Fixing Missing Values
![tracking null values](https://user-images.githubusercontent.com/82164437/115604072-ef6f7380-a2ae-11eb-9e3f-c1afe9a1a831.PNG)

Missing Pledged Values could easily be calculated by using Pledged Amount data and funded percentage.
Given our limited dataset, it was impossible to deduce location and reward levels from given data, therefore, those entries were removed from our data set.
(1,381 values out of 45,957 dataset, ~3% of the entire dataset)

## Step 2: Training the Model
A quick background on logistic regression, essentially you have a statistical model decide how important a specific attribute of your data is in predicting an output. i.e. How important is a project's category when it comes to predicting the success of a kickstarter project.

A more in-depth description can be found here: [Link](https://medium.com/swlh/what-is-logistic-regression-62807de62efa)

I wanted to figure out what properties of a project determine if a kickstart project will be successful or not. 

To start, I removed atributes that I believed did not have an impact on project success or the attribute's impact would be hard to measure. For example, while a catchy project name might help drive traffic to a project, there is currently no way for me to quantify "how good" a project name is given my current skillset.

Here's a list of the following attributes I removed:
- Project ID 
  - Removed since project ID are solely used for database purposes
- Project Name
  - Difficult to quanitfy the impact of a project name
- Project URL
  - Removed since project url is just a reflection of project name
- Funded Time
  - This attribute is determined after an project is finished, which means it would do a poor job of predicting project success
 - Funded Percentage, Funding Raised
  - I removed these because these attributes actually measure how successful a project is. Including these would lead to skewed results
 - Specified funding levels
  - Difficult to quanitify
 - Category
  - Category is highly correlated with sub-categories so I decided to keep sub-categories rather than category as it offers more granular details on projects.


## Results:
Removing the number of backer parameter from our model decreases our accuracy from 92% to 83%, this somewhat expected given the large importance of the number of backers in the original logistic regression model.

There are four ways to check if the predictions are right or wrong:

1. **TN / True Negative**: the case was negative and predicted negative
2. **TP / True Positive**: the case was positive and predicted positive
3. **FN / False Negative**: the case was positive but predicted negative
4. **FP / False Positive**: the case was negative but predicted positive

To understand these results, we need to define what precision and recall are.
- **Precision**  calculates what % of your positive predictions were correct
  - Precision is equal to (TP)/ (TP + FP)
- **Recall** calculates the percentage of positivies that you correctly identified
  - Recall = TP/(TP + FN)

**F1 score — *What percent of positive predictions were correct?***

- The F1 score is a weighted harmonic mean of precision and recall such that the **best score is 1.0 and the worst is 0.0.** 
- As a rule of thumb, the weighted average of F1 should be used to compare classifier models, not global accuracy.

#### Here are the results for the initial model that included backers:

                       precision   recall  f1-score   support

           Failed        0.89      0.90      0.90      3696
           Success       0.92      0.91      0.91      4424

    accuracy                       0.91      8120
    macro avg            0.90      0.91      0.91      8120
    weighted avg         0.91      0.91      0.91      8120



#### Results for logistic regression model without backers:

                        precision    recall  f1-score   support

           Failed        0.81      0.79      0.80      3696
           Success       0.83      0.84      0.84      4424

    accuracy                                 0.82      8120
    macro avg            0.82      0.82      0.82      8120
    weighted avg         0.82      0.82      0.82      8120

As we can see here, our model that includes the amount of backers is on average 9% more accurate than our model without backers.
Granted, an average precision score and recall score of 82% is highly predictive and helpful for determining project success. 
Since the outcome for a project is either "failed" or "success" we want our model to have precision and recall higher than what we could achieve by blinding guessing, which in this case would be 50% (given the binary nature of the outcome).

The model with backers outperforming the model without backers is not surprising given how correlated the number of backers when it comes to fundraising. i.e. The more backers who have the more likely you are to raise more money and to meet your fundraising goal.

For this reason, my recommendation is to use the model without backers for prediction purposes because the inputs of that model are all inputs that the project owner has more control over (things like subcategory, goal amount, updates, etc.)






## Installation
I personally use Anaconda for all of my python package management.
Packages required for data cleaning and editing are the following
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os 
from datetime import datetime
```


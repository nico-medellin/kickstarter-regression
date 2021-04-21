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
- Project Name
- URL 
- Category
- Subcategory
- Project Location
- Project Satus
- Goal Amount
- Pledged Amount
  - Values were missing
  - Certain Pledged Amounts were not equal to Goal Ammount * Funded Percentage
- Funded Percentage 
- # of Reward Levels
- Specific Reward levels
- # of Project Updates
- Length of Projects (Days)
  - Need to be converted from a string to an integer
- Funded Time
  - Not in datetime format
  - Need to seperate time for day from calendar data




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


#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Foundation
# 
# ## Section 1, Part c: EDA Lab

# ## Introduction
# 
# We will be using the iris data set for this tutorial. This is a well-known data set containing iris species and sepal and petal measurements. The data we will use are in a file called `iris_data.csv` found in the [data](data/) directory.

# In[1]:


import os
import numpy as np
import pandas as pd


# ## Question 1
# 
# Load the data from the file using the techniques learned today. Examine it.
# 
# Determine the following:
# 
# * The number of data points (rows). (*Hint:* check out the dataframe `.shape` attribute.)
# * The column names. (*Hint:* check out the dataframe `.columns` attribute.)
# * The data types for each column. (*Hint:* check out the dataframe `.dtypes` attribute.)

# In[22]:


filepath = "iris_data.csv"
data = pd.read_csv(filepath)
data.head()


# In[23]:


### BEGIN SOLUTION
# Number of rows
print(data.shape[0])   # size matrix in matlab

# Column names
print(data.columns.tolist()) #without .tolist(), it says ([..],dtype='object')

# Data types
print(data.dtypes)
### END SOLUTION


# ## Question 2
# 
# Examine the species names and note that they all begin with 'Iris-'. Remove this portion of the name so the species name is shorter. 
# 
# *Hint:* there are multiple ways to do this, but you could use either the [string processing methods](http://pandas.pydata.org/pandas-docs/stable/text.html) or the [apply method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.apply.html).

# In[25]:


### BEGIN SOLUTION
# The str method maps the following function to each entry as a string
data['species'] = data.species.str.replace('Iris-', '')
# alternatively
# data['species'] = data.species.apply(lambda r: r.replace('Iris-', ''))

data.head()
### END SOLUTION


# ## Question 3
# 
# Determine the following:  
# * The number of each species present. (*Hint:* check out the series `.value_counts` method.)
# * The mean, median, and quantiles and ranges (max-min) for each petal and sepal measurement.
# 
# *Hint:* for the last question, the `.describe` method does have median, but it's not called median. It's the *50%* quantile. `.describe` does not have range though, and in order to get the range, you will need to create a new entry in the `.describe` table, which is `max - min`.

# In[36]:


### BEGIN SOLUTION
# One way to count each species
data.species.value_counts()


# In[50]:


# Select just the rows desired from the 'describe' method and add in the 'median'
stats_df = data.describe()
# data.describe itselfs finds count,mean,std,min,max,25-50-75 percentiles
stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']
# loc is position based slicing/locationing for rows

out_fields = ['mean','25%','50%','75%', 'range']
stats_df = stats_df.loc[out_fields]
stats_df.rename({'50%': 'median'}, inplace=True)
stats_df

### END SOLUTION


# ## Question 4
# 
# Calculate the following **for each species** in a separate dataframe:
# 
# * The mean of each measurement (sepal_length, sepal_width, petal_length, and petal_width).
# * The median of each of these measurements.
# 
# *Hint:* you may want to use Pandas [`groupby` method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html) to group by species before calculating the statistic.
# 
# If you finish both of these, try calculating both statistics (mean and median) in a single table (i.e. with a single groupby call). See the section of the Pandas documentation on [applying multiple functions at once](http://pandas.pydata.org/pandas-docs/stable/groupby.html#applying-multiple-functions-at-once) for a hint.

# In[51]:


### BEGIN SOLUTION
# The mean calculation
data.groupby('species').mean()


# In[52]:


# The median calculation
data.groupby('species').median()


# In[60]:


# applying multiple functions at once - 2 methods

data.groupby('species').agg(['mean', 'median'])  # passing a list of recognized strings


# In[61]:


data.groupby('species').agg([np.mean, np.median, np.sum] )  # passing a list of explicit aggregation functions


# In[66]:


# If certain fields need to be aggregated differently, we can do:
from pprint import pprint

agg_dict = {field: ['mean', 'median'] for field in data.columns if field != 'species'}
agg_dict['petal_length'] = 'max'  # replace pedal_length with max
pprint(agg_dict)
data.groupby('species').agg(agg_dict)
### END SOLUTION


# ## Question 5
# 
# Make a scatter plot of `sepal_length` vs `sepal_width` using Matplotlib. Label the axes and give the plot a title.

# In[67]:


### BEGIN SOLUTION
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline #to show output')


# In[70]:


# A simple scatter plot with Matplotlib
ax = plt.axes()   # plots empty

ax.scatter(data.sepal_length, data.sepal_width)

# Label the axes
ax.set(xlabel='Sepal Length (cm)',
       ylabel='Sepal Width (cm)',
       title='Sepal Length vs Width');
### END SOLUTION


# ## Question 6
# 
# Make a histogram of any one of the four features. Label axes and title it as appropriate. 

# In[71]:


### BEGIN SOLUTION
# Using Matplotlib's plotting functionality
ax = plt.axes()
ax.hist(data.petal_length, bins=25);

ax.set(xlabel='Petal Length (cm)', 
       ylabel='Frequency',
       title='Distribution of Petal Lengths');


# In[72]:


# Alternatively using Pandas plotting functionality
ax = data.petal_length.plot.hist(bins=25)

ax.set(xlabel='Petal Length (cm)', 
       ylabel='Frequency',
       title='Distribution of Petal Lengths');
### END SOLUTION


# ## Question 7
# 
# Now create a single plot with histograms for each feature (`petal_width`, `petal_length`, `sepal_width`, `sepal_length`) overlayed. If you have time, next try to create four individual histogram plots in a single figure, where each plot contains one feature.
# 
# For some hints on how to do this with Pandas plotting methods, check out the [visualization guide](http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html) for Pandas.

# In[73]:


import seaborn as sns
sns.set_context('notebook')
### BEGIN SOLUTION
# This uses the `.plot.hist` method
ax = data.plot.hist(bins=25, alpha=0.5)
ax.set_xlabel('Size (cm)');


# In[100]:


# To create four separate plots, use Pandas `.hist` method
axList = data.hist(bins=25,figsize=(8,8))

# Add some x- and y- labels to first column and last row
for ax in axList.flatten():
    if ax.is_last_row():
        ax.set_xlabel('Size (cm)')
        
    if ax.is_first_col():
        ax.set_ylabel('Frequency')
### END SOLUTION


# In[84]:


axList


# In[85]:


axList.shape


# In[95]:


axList.flatten()


# In[96]:


axList.flatten().shape


# ## Question 8
# 
# Using Pandas, make a boxplot of each petal and sepal measurement. Here is the documentation for [Pandas boxplot method](http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html#visualization-box).

# In[101]:


### BEGIN SOLUTION
# Here we have four separate plots
data.boxplot(by='species',figsize=(10,8));
### END SOLUTION


# ## Question 9
# 
# Now make a single boxplot where the features are separated in the x-axis and species are colored with different hues. 
# 
# *Hint:* you may want to check the documentation for [Seaborn boxplots](http://seaborn.pydata.org/generated/seaborn.boxplot.html). 
# 
# Also note that Seaborn is very picky about data format--for this plot to work, the input dataframe will need to be manipulated so that each row contains a single data point (a species, a measurement type, and the measurement value). Check out Pandas [stack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.stack.html) method as a starting place.
# 
# Here is an example of a data format that will work:
# 
# |   | species | measurement  | size |
# | - | ------- | ------------ | ---- |
# | 0	| setosa  | sepal_length | 5.1  |
# | 1	| setosa  | sepal_width  | 3.5  |

# In[106]:


data.head()


# In[105]:


data.set_index('species')


# In[108]:


data.set_index('species').stack().to_frame()


# In[116]:


data.set_index('species').stack().to_frame().reset_index()


# In[115]:


data.set_index('species').stack().to_frame().reset_index().rename(columns={0:'size','level_1':'measurement'})


# In[117]:


### BEGIN SOLUTION
# First we have to reshape the data so there is 
# only a single measurement in each column

plot_data = (data
             .set_index('species')
             .stack()
             .to_frame()
             .reset_index()
             .rename(columns={0:'size', 'level_1':'measurement'})
            )

plot_data.head()
### END SOLUTION


# In[126]:


### BEGIN SOLUTION
# Now plot the dataframe from above using Seaborn

sns.set_style('white')
sns.set_context('notebook')
sns.set_palette('dark')

f = plt.figure(figsize=(9,6)) # set the figure

sns.boxplot(x='measurement', y='size', 
            hue='species', data=plot_data);
### END SOLUTION


# ## Question 10
# 
# Make a [pairplot](http://seaborn.pydata.org/generated/seaborn.pairplot.html) with Seaborn to examine the correlation between each of the measurements.
# 
# *Hint:* this plot may look complicated, but it is actually only a single line of code. This is the power of Seaborn and dataframe-aware plotting! See the lecture notes for reference.

# In[131]:


### BEGIN SOLUTION
sns.set_context('talk')
plot=sns.pairplot(data, hue='species');

plot._legend.remove()  #if you dont use this, sns.pairplot(,,,,) also plots the plot.

### END SOLUTION


# ---
# ### Machine Learning Foundation (C) 2020 IBM Corporation

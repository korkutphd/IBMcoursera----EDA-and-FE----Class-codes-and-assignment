#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Foundation
# 
# ## Section 1, Part d: Feature Engineering 

# ## Feature Engineering with Linear Regression: Applied to the Ames Housing Data
# 
# Using the Ames Housing Data:
# 
# Dean De Cock
# Truman State University
# Journal of Statistics Education Volume 19, Number 3(2011), www.amstat.org/publications/jse/v19n3/decock.pdf
# 
# In this notebook, we will build some linear regression models to predict housing prices from this data. In particular, we will set out to improve on a baseline set of features via **feature engineering**: deriving new features from our existing data. Feature engineering often makes the difference between a weak model and a strong one.
# 
# We will use visual exploration, domain understanding, and intuition to construct new features that will be useful later in the course as we turn to prediction.
# 
# **Notebook Contents**
# 
# > 1. Simple EDA 
# > 2. One-hot Encoding variables
# > 3. Log transformation for skewed variables
# > 4. Pair plot for features
# > 5. Basic feature engineering: adding polynomial and interaction terms
# > 6. Feature engineering: categories and features derived from category aggregates 
# 
# ## 1. Simple EDA 

# In[3]:


get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['retina']")

import pandas as pd
import seaborn as sns
sns.set()  #setting defaults for seaborn plots


# #### Load the Data, Examine and Explore

# In[4]:


## Load in the Ames Housing Data
datafile = "Ames_Housing_Data.tsv"
df = pd.read_csv(datafile, sep='\t')


# In[5]:


## Examine the columns, look at missing data
df.info()


# In[6]:


df['Gr Liv Area'].hist()


# In[7]:


# This is recommended by the data set author to remove a few outliers

df = df.loc[df['Gr Liv Area'] <= 4000,:]
print("Number of rows in the data:", df.shape[0])
print("Number of columns in the data:", df.shape[1])
data = df.copy() # Keep a copy our original data 


# In[45]:


# A quick look at the data:
df.head()


# In[9]:


len(df.PID.unique())  #every value is unique
df.drop(['PID','Order'],axis=1,inplace=True)


# We're going to first do some basic data cleaning on this data: 
# 
# * Converting categorical variables to dummies
# * Making skew variables symmetric
# 
# ### One-hot encoding for dummy variables:

# In[10]:


# Get a Pd.Series consisting of all the string categoricals
one_hot_encode_cols = df.dtypes[df.dtypes == np.object]  # filtering by string categoricals
one_hot_encode_cols = one_hot_encode_cols.index.tolist()  # list of categorical fields

df[one_hot_encode_cols].head().T


# We're going to first do some basic data cleaning on this data: 
# 
# * Converting categorical variables to dummies
# * Making skew variables symmetric
# 
# #### One-hot encoding the dummy variables:

# In[11]:


# Do the one hot encoding
df = pd.get_dummies(df, columns=one_hot_encode_cols, drop_first=True)
df.describe().T


# In[12]:


df.select_dtypes('number')  #filter to only numbers or 'object' for object


# In[13]:


df.select_dtypes('number').columns  #filter to only numbers or 'object' for object


# ### Log transforming skew variables

# In[14]:


# Create a list of float colums to check for skewing
num_cols=df.select_dtypes('number').columns  #filter to only numbers or 'object' for object

mask = data.dtypes == np.float
float_cols = data.columns[mask]

skew_limit = 0.75 # define a limit above which we will log transform
skew_vals = data[float_cols].skew()

skew_vals


# In[15]:


# Showing the skewed columns
skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))

skew_cols


# In[16]:


# Let's look at what happens to one of these features, when we apply np.log1p visually.

# Choose a field
field = "SalePrice"

# Create two "subplots" and a "figure" using matplotlib
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))

# Create a histogram on the "ax_before" subplot
df[field].hist(ax=ax_before)

# Apply a log transformation (numpy syntax) to this column
df[field].apply(np.log1p).hist(ax=ax_after)

# Formatting of titles etc. for each subplot
ax_before.set(title='before np.log1p', ylabel='frequency', xlabel='value')
ax_after.set(title='after np.log1p', ylabel='frequency', xlabel='value')
fig.suptitle('Field "{}"'.format(field));


# In[17]:


# Perform the skew transformation:

for col in skew_cols.index.values:   #except the sale price
    if col == "SalePrice":
        continue
    df[col] = df[col].apply(np.log1p)


# In[18]:


data.isnull().sum().sort_values()


# In[19]:


# We now have a larger set of potentially-useful features
df.shape


# In[20]:


# There are a *lot* of variables. Let's go back to our saved original data and look at how many values are missing for each variable. 
df = data
data.isnull().sum().sort_values()


#  Let's pick out just a few numeric columns to illustrate basic feature transformations.

# In[21]:


smaller_df= df.loc[:,['Lot Area', 'Overall Qual', 'Overall Cond', 
                      'Year Built', 'Year Remod/Add', 'Gr Liv Area', 
                      'Full Bath', 'Bedroom AbvGr', 'Fireplaces', 
                      'Garage Cars','SalePrice']]


# In[22]:


# Now we can look at summary statistics of the subset data
smaller_df.describe().T


# In[23]:


smaller_df.info()


# In[24]:


# There appears to be one NA in Garage Cars - we will take a simple approach and fill it with 0
smaller_df = smaller_df.fillna(0)


# In[25]:


smaller_df.info()


# 
# ### Pair plot of features
# Now that we have a nice, filtered dataset, let's generate visuals to better understand the target and feature-target relationships: pairplot is great for this!

# In[26]:


sns.pairplot(smaller_df, plot_kws=dict(alpha=.1, edgecolor='none'))


# ---
# **Data Exploration Discussion**: 
# 
# 1. What do these plots tell us about the distribution of the target?   
# 
# 2. What do these plots tell us about the relationship between the features and the target? Do you think that linear regression is well-suited to this problem? Do any feature transformations come to mind?
# 
# 3. What do these plots tell us about the relationship between various pairs of features? Do you think there may be any problems here? 
# 
# ---

# #### Suppose our target variable is the SalePrice. We can set up separate variables for features and target.

# In[28]:


#Separate our features from our target

X = smaller_df.loc[:,['Lot Area', 'Overall Qual', 'Overall Cond', 
                      'Year Built', 'Year Remod/Add', 'Gr Liv Area', 
                      'Full Bath', 'Bedroom AbvGr', 'Fireplaces', 
                      'Garage Cars']]

y = smaller_df['SalePrice']


# In[29]:


X.info()


# Now that we have feature/target data X, y ready to go, we're nearly ready to fit and evaluate a baseline model using our current feature set. We'll need to create a **train/validation split** before we fit and score the model. 
# 
# Since we'll be repeatedly splitting X, y into the same train/val partitions and fitting/scoring new models as we update our feature set, we'll define a reusable function that completes all these steps, making our code/process more efficient going forward. 

# Great, let's go ahead and run this function on our baseline feature set and take some time to analyze the results.

# ### Basic feature engineering: adding polynomial and interaction terms

# One of the first things that we looked for in the pairplot was evidence about the relationship between each feature and the target. In certain features like _'Overall Qual'_ and _'Gr Liv Qual'_, we notice an upward-curved relationship rather than a simple linear correspondence. This suggests that we should add quadratic **polynomial terms or transformations** for those features, allowing us to express that non-linear relationship while still using linear regression as our model.
# 
# Luckily, pandas makes it quite easy to quickly add those square terms as additional features to our original feature set. We'll do so and evaluate our model again below.
# 
# As we add to our baseline set of features, we'll create a copy of the latest benchmark so that we can continue to store our older feature sets. 
# ### Polynomial Features

# In[30]:


X2 = X.copy()

X2['OQ2'] = X2['Overall Qual'] ** 2
X2['GLA2'] = X2['Gr Liv Area'] ** 2


# As is, each feature is treated as an independent quantity. However, there may be **interaction effects**, in which the impact of one feature may dependent on the current value of a different feature.
# 
# For example, there may be a higher premium for increasing _'Overall Qual'_ for houses that were built more recently. If such a premium or a similar effect exists, a feature that multiplies _'Overall Qual'_ by _'Year Built'_ can help us capture it.
# 
# Another style of interaction term involves feature proprtions: for example, to get at something like quality per square foot we could divide _'Overall Qual'_ by _'Lot Area'_.
# 
# Let's try adding both of these interaction terms and see how they impact the model results.
# 
# ### Feature interactions

# In[31]:


X3 = X2.copy()

# multiplicative interaction
X3['OQ_x_YB'] = X3['Overall Qual'] * X3['Year Built']

# division interaction
X3['OQ_/_LA'] = X3['Overall Qual'] / X3['Lot Area']


# -----
# **Interaction Feature Exercise**: What other interactions do you think might be helpful? Why? 
# 
# -----

# ### Categories and features derived from category aggregates 

# Incorporating **categorical features** into linear regression models is fairly straightforward: we can create a new feature column for each category value, and fill these columns with 1s and 0s to indicate which category is present for each row. This method is called **dummy variables** or **one-hot-encoding**.
# 
# We'll first explore this using the _'House Style'_ feature from the original dataframe. Before going straight to dummy variables, it's a good idea to check category counts to make sure all categories have reasonable representation.

# In[47]:


data['House Style'].value_counts()


# This looks ok, and here's a quick look at how dummy features actually appear:

# In[52]:


pd.get_dummies(df['House Style'], drop_first=True).head()


# We can call `pd.get_dummies()` on our entire dataset to quickly get data with all the original features and dummy variable representation of any categorical features. Let's look at some variable values.

# In[59]:


nbh_counts = df.Neighborhood.value_counts()
nbh_counts


# For this category, let's map the few least-represented neighborhoods to an "other" category before adding the feature to our feature set and running a new benchmark.

# In[60]:


other_nbhs = list(nbh_counts[nbh_counts <= 8].index)

other_nbhs


# In[62]:


X4 = X3.copy()

X4['Neighborhood'] = df['Neighborhood'].replace(other_nbhs, 'Other')

X4.Neighborhood.value_counts()


# #### Getting to fancier features
# 
# Let's close out our introduction to feature engineering by considering a more complex type of feature that may work very nicely for certain problems. It doesn't seem to add a great deal over what we have so far, but it's a style of engineering to keep in mind for the future.
# 
# We'll create features that capture where a feature value lies relative to the members of a category it belongs to. In particular, we'll calculate deviance of a row's feature value from the mean value of the category that row belongs to. This helps to capture information about a feature relative to the category's distribution, e.g. how nice a house is relative to other houses in its neighborhood or of its style.
# 
# Below we define reusable code for generating features of this form, feel free to repurpose it for future feature engineering work!

# In[75]:


X4.groupby('Neighborhood')['Overall Qual'].transform(lambda x: x.std())


# In[63]:


def add_deviation_feature(X, feature, category):
    
    # temp groupby object
    category_gb = X.groupby(category)[feature]
    
    # create category means and standard deviations for each observation
    category_mean = category_gb.transform(lambda x: x.mean())
    category_std = category_gb.transform(lambda x: x.std())
    
    # compute stds from category mean for each feature value,
    # add to X as new feature
    deviation_feature = (X[feature] - category_mean) / category_std 
    X[feature + '_Dev_' + category] = deviation_feature  


# And now let's use our feature generation code to add 2 new deviation features, and run a final benchmark.

# In[78]:


X5 = X4.copy()
X5['House Style'] = df['House Style']
add_deviation_feature(X5, 'Year Built', 'House Style')
add_deviation_feature(X5, 'Overall Qual', 'Neighborhood')

X5


# ## Polynomial Features in Scikit-Learn
# 
# `sklearn` allows you to build many higher-order terms at once with `PolynomialFeatures`

# In[79]:


from sklearn.preprocessing import PolynomialFeatures


# In[84]:


#Instantiate and provide desired degree; 
#   Note: degree=2 also includes intercept, degree 1 terms, and cross-terms

pf = PolynomialFeatures(degree=2)


# In[85]:


features = ['Lot Area', 'Overall Qual']
pf.fit(df[features])


# In[87]:


pf.get_feature_names()  #Must add input_features = features for appropriate names


# In[88]:


feat_array = pf.transform(df[features])
pd.DataFrame(feat_array, columns = pf.get_feature_names(input_features=features))


# ## Recap
# 
# While we haven't yet turned to prediction, these feature engineering exercises set the stage. Generally, feature engineering often follows a sort of [_Pareto principle_](https://en.wikipedia.org/wiki/Pareto_principle), where a large bulk of the predictive gains can be reached through adding a set of intuitive, strong features like polynomial transforms and interactions. Directly incorporating additional information like categorical variables can also be very helpful. Beyond this point, additional feature engineering can provide significant, but potentially diminishing returns. Whether it's worth it depends on the use case for the model. 

# ---
# ### Machine Learning Foundation (C) 2020 IBM Corporation

# In[ ]:





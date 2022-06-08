#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Foundation
# 
# ## Section 1, Part a: Reading Data 

# ### Learning Objective(s)
# 
#  - Create a SQL database connection to a sample SQL database, and read records from that database
#  - Explore common input parameters
# 
# ### Packages
# 
#  - [Pandas](https://pandas.pydata.org/pandas-docs/stable/)
#  - [Pandas.read_sql](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html)
#  - [SQLite3](https://docs.python.org/3.6/library/sqlite3.html)

# ## Simple data reads
# 
# Structured Query Language (SQL) is an [ANSI specification](https://docs.oracle.com/database/121/SQLRF/ap_standard_sql001.htm#SQLRF55514), implemented by various databases. SQL is a powerful format for interacting with large databases efficiently, and SQL allows for a consistent experience across a large market of databases. We'll be using sqlite, a lightweight and somewhat restricted version of sql for this example. sqlite uses a slightly modified version of SQL, which may be different than what you're used to. 

# In[5]:


# Imports
import sqlite3 as sq3
import pandas.io.sql as pds
import pandas as pd


# ### Database connections
# 
# Our first step will be to create a connection to our SQL database. A few common SQL databases used with Python include:
# 
#  - Microsoft SQL Server
#  - Postgres
#  - MySQL
#  - AWS Redshift
#  - AWS Aurora
#  - Oracle DB
#  - Terradata
#  - Db2 Family
#  - Many, many others
#  
# Each of these databases will require a slightly different setup, and may require credentials (username & password), tokens, or other access requirements. We'll be using `sqlite3` to connect to our database, but other connection packages include:
# 
#  - [`SQLAlchemy`](https://www.sqlalchemy.org/) (most common)
#  - [`psycopg2`](http://initd.org/psycopg/)
#  - [`MySQLdb`](http://mysql-python.sourceforge.net/MySQLdb.html)

# In[8]:


# Initialize path to SQLite database
path = 'classic_rock.db'
con = sq3.Connection(path)

# We now have a live connection to our SQL database


# ### Reading data
# 
# Now that we've got a connection to our database, we can perform queries, and load their results in as Pandas DataFrames

# In[11]:


# Write the query
query = '''
SELECT * 
FROM rock_songs;
'''

# Execute the query
observations = pds.read_sql(query, con)

observations.head()


# In[10]:


# We can also run any supported SQL query
# Write the query
query = '''
SELECT Artist, Release_Year, COUNT(*) AS num_songs, AVG(PlayCount) AS avg_plays  
    FROM rock_songs
    GROUP BY Artist, Release_Year
    ORDER BY num_songs desc;
'''

# Execute the query
observations = pds.read_sql(query, con)

observations.head()


# ## Common parameters
# 
# There are a number of common paramters that can be used to read in SQL data with formatting:
# 
#  - coerce_float: Attempt to force numbers into floats
#  - parse_dates: List of columns to parse as dates
#  - chunksize: Number of rows to include in each chunk
#  
# Let's have a look at using some of these parameters

# In[5]:


query='''
SELECT Artist, Release_Year, COUNT(*) AS num_songs, AVG(PlayCount) AS avg_plays  
    FROM rock_songs
    GROUP BY Artist, Release_Year
    ORDER BY num_songs desc;
'''

# Execute the query
observations_generator = pds.read_sql(query,
                            con,
                            coerce_float=True, # Doesn't efefct this dataset, because floats were correctly parsed
                            parse_dates=['Release_Year'], # Parse `Release_Year` as a date
                            chunksize=5 # Allows for streaming results as a series of shorter tables
                           )

for index, observations in enumerate(observations_generator):
    if index < 5:
        print(f'Observations index: {index}'.format(index))
        display(observations)


# ### Machine Learning Foundation (C) 2020 IBM Corporation

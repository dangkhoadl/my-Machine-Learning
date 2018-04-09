

import pandas as pd
import numpy as np

######################################## Series ###############################################
# create a Series object
series = pd.Series(['Dave', 'Cheng-Han', 'Udacity', 42, -1789710578])
print(series)

# custom index
series = pd.Series(['Dave', 'Cheng-Han', 359, 9001],
                    index=['Instructor', 'Curriculum Manager',
                            'Course Number', 'Power Level'])
print(series)

# Series indexing
series = pd.Series(['Dave', 'Cheng-Han', 359, 9001],
                    index= ['Instructor', 'Curriculum Manager',
                            'Course Number', 'Power Level'])
print(series['Instructor'])
print(series[['Instructor', 'Curriculum Manager', 'Course Number']])

# Boolean indexing
cuteness = pd.Series([1, 2, 3, 4, 5], index=['Cockroach', 'Fish', 'Mini Pig',
                                             'Puppy', 'Kitten'])
print(cuteness > 3)
print(cuteness[cuteness > 3])



######################################## Dataframes ###############################################
data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
        'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions',
                 'Lions', 'Lions'],
        'wins': [11, 8, 10, 15, 11, 6, 10, 4],
        'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
football = pd.DataFrame(data)
print(football.dtypes)
print(football.describe())
print(football.head())
print(football.tail())

######################################## Dataframes indexing ###############################################
data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
        'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions',
                 'Lions', 'Lions'],
        'wins': [11, 8, 10, 15, 11, 6, 10, 4],
        'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
football = pd.DataFrame(data)
print(football['year'])
print(football.year)  # shorthand for football['year']
print(football[['year', 'wins', 'losses']])


# boolean indexing in action
data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
        'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions',
                 'Lions', 'Lions'],
        'wins': [11, 8, 10, 15, 11, 6, 10, 4],
        'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
football = pd.DataFrame(data)
print(football.iloc[[0]])
print(football.loc[[0]])
print(football[3:5])
print(football[football.wins > 10])
print(football[(football.wins > 10) & (football.team == "Packers")])

######################################## vectorized methods ###############################################
data = {'one' : pd.Series([1,2,3], index=['a','b','c']),
        'two' : pd.Series([1,2,3,4], index=['a','b','c','d'])}
df = pd.DataFrame(data)
print(df)
print(df.apply(np.mean))
print(df['one'].apply(np.mean))
print(df['one'].map(lambda x: x >= 1))
print(df.applymap(lambda x: x >= 1))

#### Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sqlalchemy as sql

import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) # Jupyter Notebook only
import plotly.graph_objs as go

import cufflinks as cf
cf.go_offline()

# Python Basics


1 + 1   # addition
1 * 3   # multiplication
1 / 2   # division
2 ** 4  # potentiation
4 % 3   # remainder / %val == 0 "divisible by"

1 > 2   # greater than
1 < 2   # less than
1 >= 1  # greather or equal than
1 <= 4  # less or equal than
1 == 1  # is equal to
'hi' != 'bye'  # not equal to

(1 > 2) and (2 < 3)  # all conditions must be true for TRUE
(1 == 1) or (2 == 3) or (3 == 2)  # only one conditions needs to be true for TRUE

## Demonstration of variable info vs printing actual output
var = 20
var
print(var)
print('The answer is {}'.format(var))

var_array = np.array([0, 1, 2, 3])
var_array
print(var_array)
print('The answer {} is not contained in {}'.format(var, var_array))


## Locating elements in strings


name = 'bruno'
name[0]
name[1:2]
name[2:]


## Lists & locating elements in lists


[2, 'fast', 4, 'you']

my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
my_list.pop(2)
my_list.pop()  # removal is permanent
my_list.append(15)
my_list

my_list[0]
my_list[0] = 'Zero is the new black'
my_list
my_list[0][0]

'Zero' in my_list
5 in my_list


## Tuples (values can't be redefined unlike lists)


not_a_list = (1, 2, 3)
not_a_list[0]

water_properties = [('Tc', -20), ('Tm', 0), ('Tg', -64)]
water_properties[2][1]

for data in water_properties:
    print(data)

for (a, b) in water_properties:
    print(a)  # this is called tuple unpacking


## Nesting arrays & multi-level locating


nest = [my_list, 5, 6]
nest
nest[0][2]  # searches index 2 inside index 0 of nest variable


## Dictionaries


polymerA = {'structure': 'branched', 'function': 'cryoprotective'}
polymerA['function']
polymerA.keys()
polymerA.values()
polymerA.items()

for relationships in polymerA:
    print(relationships)  # would need tuples, not dictionary


## Sets (values with non-repetition)


bulkdata = {1, 2, 3, 2, 1}
bulkdata


## IF, ELIF, ELSE statements


if 1 == 2:
  print('You just broke math')
elif 2 == 'cat':
  print('What an absolute madlad')
else:
  print('stonks')


## FOR Loops created with a range generator


range_list = range(5)

for value in range_list:
  print('Yep, that is a {}'.format(value))


## WHILE Loops with an operator inside the print


i = 1
while i < 5:
  print(i + i)
  i = i + 1


## Saving output in lists


output = []
for item in range_list:
  output.append(item**2)
  print(output)


## FUNCTIONS (for storage)


def duplicate(x):
  return x * 2

function_output = duplicate(var_array)
function_output


## LAMBDA expressions (applied directly at line of code)


lambda x: x * 2


## MAP function (applies function type to var, stores in a list)


map(duplicate, var_array)
list(map(duplicate, var_array))
list(map(lambda x: x * 2, var_array))


## FILTER function only outputs True conditions in the function


divisible_by_two = filter(lambda x: x % 2 == 0, var_array)
list(divisible_by_two)


## String methods


edgymeme = 'OmG, i, KnOw, HoW, tO, cOdE'
edgymeme.lower()
edgymeme.upper()
edgymeme.split()
edgymeme.split(',')
edgymeme.split(',')[5]
edgymeme


# NumPy

## Vectors (1d) and Matrices (2d+)


list3d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matrix = np.array(list3d)
matrix


## Generators


##### discrete
empty = np.zeros(50)
np.zeros((5, 5))
np.ones([3, 3])
identity = np.eye(5)
np.arange(0, 100, 20)
np.linspace(0, 100, 20)

##### random
np.random.randint(100)
np.random.randint(0, 100, 10)
np.random.seed(100)

##### random (uniform distribution, values 0 to 1)
np.random.rand(10)
np.random.rand(5, 5)

##### random (normal distribution, values -1 to 1)
np.random.randn(2, 2)
np.random.randn(2, 2, 3)


## Array modifiers


discrete_set = np.arange(10)
discrete_set

random_set = np.random.randint(0, 10, 10)
random_set

matrix_discrete_set = discrete_set.reshape(5, 2) # must be stored, wont save
matrix_discrete_set


## Array probes


random_set.max()
random_set.min()
random_set.argmax() # index location of max
random_set.argmin() # index location of min

matrix_discrete_set.shape
matrix_discrete_set.dtype


## Broadcasting (permanent multi-append)


discrete_set
discrete_set[2:5] = 100
discrete_set


## Array slicing (an instance of original array, but still changes it)


discrete_set
discrete_slice = discrete_set[2:5]
discrete_slice

discrete_slice[:] = 99
discrete_slice
discrete_set


## Array copy (work independently of original array)


discrete_copy = discrete_slice.copy()
discrete_copy


## 2D arrays


##### Indexing
matrix
matrix[1][0]
matrix[1, 0]
matrix[1, :]
matrix[:2, 1:]

empty2d = np.zeros((10, 5))
empty2d_nrows = empty2d.shape[0]
empty2d_ncols = empty2d.shape[1]

print('The array \n{} \nhas {} rows \nand {} columns.'.format( empty2d, empty2d_nrows, empty2d_ncols))


##### Fancy indexing
for i in range(empty2d_nrows):
  empty2d[i] = i

empty2d
empty2d[[8, 5, 1, 9]]


##### Selection
empty2d > 4
empty2d[empty2d > 4]

outcome = 8
empty2d[empty2d >= outcome][2:7]


## Array operators

my_list + my_list # not numpy, just merges arrays
var_array + var_array
var_array + 100

var_array / var_array # 0 shows as 'nan'
1 / var_array # infinity shows as 'inf'

##### modifiers need 'np.' callout
np.sqrt(var_array)
np.exp(var_array)
np.log(var_array)
np.sin(var_array)

##### locators can be '.function' applied to array
np.max(var_array)
var_array.max()
var_array.min()
var_array.sum()
var_array.std()

matrix
matrix.sum(axis=0) # column sum
matrix.sum(axis=1) # row sum


# Pandas

## Series


var_array # data can be: list, dictionary, numpy array
series_labels = ['a', 'b', 'c', 'd']

##### automatic indexing
var_series = pd.Series(var_array)
var_series

##### custom indexing
var_series = pd.Series(data=var_array, index=series_labels)
var_series2 = pd.Series(data=[0, 1, 2, 3], index=['a','b','c','e'])

##### locating
var_series['b']

##### operations
var_series + var_series2


## DataFrames (merging of series with same index: spreadsheet)


np.random.seed(1337) # always obtain same numbers
dataframe = pd.DataFrame(data=np.random.randn(5,4), index='Cryoprotection Antioxidant TH IRI CrystalSize'.split(), columns='Water C+ C- Polymer'.split())
dataframe

type(dataframe)
type(dataframe['Water'])


##### locating columns
dataframe['Water']
dataframe[['Water', 'Polymer']]

##### locating rows
dataframe.loc['CrystalSize']
dataframe.iloc[4]

##### locating data pairs (useful for comparisons)
dataframe.loc['Cryoprotection','Polymer']
dataframe.loc[['Cryoprotection', 'IRI'], ['Water', 'Polymer']]

##### create new column
dataframe['Antifreeze Protein'] = np.zeros((5, 1))
dataframe

##### remove row
dataframe.drop('IRI', axis=0)

##### remove column
dataframe.drop('C+', axis=1, inplace=True)  # permanent dataframe


## Conditional Selection


##### one condition
dataframe > 0
dataframe[dataframe > 0]
dataframe[dataframe['Water'] > 0]
dataframe[dataframe['Water'] > 0]['polymer']
dataframe[dataframe['Water'] > 0][['Polymer', 'Antifreeze Protein']]

##### two conditions (must be in parentheses)
dataframe[(dataframe['Water'] > 0) & (dataframe['Polymer'] > 0)]
dataframe[(dataframe['Water'] > 0) | (dataframe['Polymer'] > 0)]


## Reset index


dataframe.reset_index()
newindex = 'Outcome Function1 Function2 Function3 Function4'.split()
dataframe['Class'] = newindex
dataframe.set_index('Class', inplace=True)
dataframe


## Multi-index and hierarchy


stats = pd.DataFrame(np.zeros((8,5)), columns = ['FucoPol', 'STB1', 'REST10', 'ACM1', 'V2'])
stats

outside_index = ['Structure', 'Structure', 'Structure', 'Structure', 'Dynamics', 'Dynamics', 'Dynamics', 'Dynamics']
inside_index = ['%Glc', '%Gal', '%Fuc', '<other>', 'Flexibility', 'Relaxation rate', 'Brownian coeff', '<other>']

multi_index = pd.MultiIndex.from_tuples(list(zip(outside_index, inside_index)))
stats.set_index(multi_index, inplace=True)
stats


## Multi-index with pivot table


stats_sheet = {'outside':['foo','foo','foo','bar','bar','bar'],
               'inside':['one','one','two','two','one','one'],
               'vars':['x','y','x','y','x','y'], 'data':[1,3,2,5,4,1]}

stats_df = pd.DataFrame(stats_sheet)
stats_df.pivot_table(values='data',
                     index=['outside', 'inside'],
                     columns=['vars'])  # if no combination of val/index/cols is found, a NaN is placed


## Naming inside and outside indices


stats.index.names = ['Class', 'Parameter']
stats.index.names


## Multi-index Locating


stats.loc['Structure']
stats.xs('Structure')  # only works in multi-index

stats.loc['Structure'].loc['%Fuc']
stats.xs(['Structure', '%Fuc'])  # avoids using multiple .loc[]

stats.xs('Dynamics', level='Class')  # go straight to what you want
stats.xs('<other>', level='Parameter')  # in case of repeat, shows in both


## Missing Data


incomplete = pd.DataFrame({'Tc':[1,2,np.nan],
                           'Tm':[5,np.nan,np.nan],
                           'Tg':[1,2,3] })
incomplete

incomplete.dropna()
incomplete.dropna(axis=1)
incomplete.dropna(thresh=2, inplace=True) # if number of NaN >= 2, drop incomplete

incomplete.fillna('null')
incomplete['Tm'].fillna(incomplete['Tm'].mean())
incomplete


## Groupby


inventory = {'Item':['Cell lines', 'Tips', 'Filters', 'Baloons', 'Microplates'],
             'Lab':['GREAT', 'BIOENG', 'PHOTOCHEM', 'BIOENG', 'GREAT'],
             'Amount':[14, 1000, 250, 30, 120]}

inventory_sheet = pd.DataFrame(inventory)

inventory_sheet.groupby('Lab')
inventory_sheet.groupby('Lab').sum()
inventory_sheet.groupby('Lab').count() # also counts strings
inventory_sheet.groupby('Lab').describe() # full statistical analysis
inventory_sheet.groupby('Lab').describe().transpose()
inventory_sheet.groupby('Lab').describe().transpose()['BIOENG']


## Merging, Joining and Concatenating


replicate1 = pd.DataFrame({'Tm': ['A0', 'A1', 'A2', 'A3'],
                           'Tf': ['B0', 'B1', 'B2', 'B3'],
                           'Tg': ['C0', 'C1', 'C2', 'C3'],
                           'Aging': ['D0', 'D1', 'D2', 'D3']},
                           index=[0, 1, 2, 3])

replicate2 = pd.DataFrame({'Tm': ['A4', 'A5', 'A6', 'A7'],
'Tf': ['B4', 'B5', 'B6', 'B7'],
'Tg': ['C4', 'C5', 'C6', 'C7'],
'Aging': ['D4', 'D5', 'D6', 'D7']},
index=[4, 5, 6, 7])

replicate3 = pd.DataFrame({'Tm': ['A8', 'A9', 'A10', 'A11'],
                           'Tf': ['B8', 'B9', 'B10', 'B11'],
                           'Tg': ['C8', 'C9', 'C10', 'C11'],
                           'Aging': ['D8', 'D9', 'D10', 'D11']},
                           index=[8, 9, 10, 11])

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

example1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                         'B': ['B0', 'B1', 'B2']},
                         index=['K0', 'K1', 'K2'])

example2 = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                         'D': ['D0', 'D2', 'D3']},
                         index=['K0', 'K2', 'K3'])

##### Concatenate
pd.concat([replicate1, replicate2, replicate3])
pd.concat([replicate1, replicate2, replicate3], axis=1) # for diff cols


##### Merge Inner (if common cells dont match, partially merged)
pd.merge(left, right, on=['key1', 'key2']) # works like "and"

##### Merge Outer (full merge but with NaNs)
pd.merge(left, right, how='outer', on=['key1', 'key2']) # works like "or"

##### Merge Sides (only merges common columns from the right/left dataframe)
pd.merge(left, right, how='right', on=['key1', 'key2'])
pd.merge(left, right, how='left', on=['key1', 'key2'])


##### Join (dataframes with different indexes)
example1.join(example2) # preserve 1, join 2 filtered by index of 1
example1.join(example2, how='outer') # unfiltered join


## General operations


mockdf = pd.DataFrame({'col1':[1,2,3,np.nan],
                       'col2':[444,555,666,444],
                       'col3':['abc','def','ghi','xyz']})
mockdf

mockdf.head(1)
mockdf.columns
mockdf.index
mockdf.info()
mockdf.describe()

mockdf['col2'].unique()
mockdf['col2'].nunique()  # same as len(mockdf['col2'].unique())
mockdf['col2'].count()
mockdf['col2'].value_counts()
mockdf.sort_values(by='col2')
mockdf.isnull()

mockdf.apply(duplicate)
mockdf.apply(lambda x: x*2)
mockdf.apply(len)
mockdf['col2'].sum()
mockdf[['col1','col2']].corr()

mockdf['col2'].max()
mockdf['col2'].idxmax()
mockdf.loc[mockdf['col2'].idxmax()]

mockdf.dropna() # drop rows with NaN values
mockdf.fillna('FILL')
del mockdf['col1'] # same as mockdf.drop('col1', axis=1, inplace=True)


## Input and Output


pwd # where i am. any csv/excels must be where

##### CSV
importedcsv = pd.read_csv('csvfile', delimiter=',')
importedcsv.to_csv('csvfile, index=False') # after work, save. index not copied

##### Excel
importedcsv.read_excel('excel.xlsx', sheetname="Sheet1") # read
importedcsv.to_excel('excel.xlsx', sheetname="Sheet1") # save/overwrite
importedcsv.to_excel('excel.xlsx', sheet_name="New Sheet") # create new

##### HTML
importedcsv = pd.read_html('<http://www.fdic.gov/bank/individual/failed/banklist.html>')
importedcsv
importedcsv[0]

##### SQL
engine = sql.create_engine('sqlite:///:memory:')
stats_df.to_sql('stats_sheet', engine)
statsSQL = pd.read_sql('stats_sheet', con='engine') # 'con' is connection


# matplotlib
##### https://matplotlib.org/gallery/index.html for plots, animations, apis, etc.

## Functional Plotting


x = np.linspace(0, 5, 11)
y = x ** 2

plt.plot(x, y, 'b.-')

plt.xlabel('Time passed (d)')
plt.ylabel('Increase in knowledge (brain cells)')
plt.title('My first matplotlib graph!')
plt.show()


## Object-oriented Plotting


fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]
axes.plot(x, y)

axes.set_xlabel('Concentration')
axes.set_ylabel('Absorbance')
axes.set_title('UV-VIS')

##### Multiplots
plt.subplot(1, 2, 1)  # nr rows, nr cols, graph numbers
plt.plot(x,y,'r')

plt.subplot(1,2,2)
plt.plot(y, x, 'gd-')

##### Plots with insets
firstgraph = plt.figure()
axes1 = firstgraph.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = firstgraph.add_axes([0.55, 0.55, 0.3, 0.3])
axes1.plot(x,y)
axes2.plot(y,x)
axes1.set_title('Main')
axes2.set_title('Inset')


## Multiplots (+++ complexity and control)


##### Axes as a discrete list
firstgraph, ([ax1,ax2,ax3],
      [ax4,ax5,ax6]) = plt.subplots(nrows = 2, ncols = 3)

firstgraph.tight_layout()  # solves axis superposition

im1 = ax1.plot(x,y)
im2 = ax2.plot(x,y*y)
im3 = ax3.plot(x*x,y)
im4 = ax4.plot(y, x)
im5 = ax5.plot(30*x,y)
im6 = ax6.plot()
ax1.set_title('normal')
ax2.set_title('greater exponential')
ax3.set_title('linear')
ax4.set_title('inverse')
ax5.set_title('scalar')
ax6.set_title('empty')

##### Axes as tuple unpacked
firstgraph,axes = plt.subplots(nrows = 2, ncols = 3, figsize=(8, 4), dpi=600)
firstgraph.tight_layout()
axes[0, 0].plot(x,y)
axes[0, 1].plot(x,y*y)
axes[0, 2].plot(x*x,y)
axes[1, 0].plot(y, x)
axes[1, 1].plot(30*x,y)
axes[1, 2].plot()
axes[0, 0].set_title('normal')
axes[0, 1].set_title('greater exponential')
axes[0, 2].set_title('linear')
axes[1, 0].set_title('inverse')
axes[1, 1].set_title('scalar')
axes[1, 2].set_title('empty')


## Figure Size, Aspect Ratio and DPI

emptygraph = plt.figure(figsize=(8, 2), dpi=600)  # figsize in width, height
ax = emptygraph.add_axes([0,0,1,1])


## Save a graph

firstgraph.savefig('firstgraph.png', dpi=600)
firstgraph.savefig('firstgraph.svg', dpi=600)
firstgraph.savefig('firstgraph.pdf', dpi=600)


## Graph detailing

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])

ax.set_title('Title')                                     # Title
ax.set_ylabel('Y')                                        # Dependent variable
ax.set_xlabel('X')                                        # Independent variable
ax.plot(y,x**2, 'r.-', label='X squared')                 # Customised red line
ax.plot(y,x**3, 'b.-', label='X cubed')                   # Customised blue line
ax.legend()                                               # Legend. requires label="" attribute in plots

ax.legend(loc=0)                              # default is loc=0 -> "best"
ax.legend(loc=2)                              # upper left corner
ax.legend(loc=3)                              # lower left corner
ax.legend(loc=4)                              # lower right corner
ax.legend(loc=(0.1, 0.4))                     # customised position


## Graph appearance

fig = plt.figure(dpi=600)
ax = fig.add_axes([0,0,1,1])
ax.plot(x,x)

##### Color plot
ax.plot(x,x+1, color='green')
ax.plot(x,x+2, color="blue", alpha=0.5)
ax.plot(x,x+3, color="#FF8C00")

##### Linewidth (lw)
ax.plot(x,x+8, color="red", linewidth=0.25)
ax.plot(x, x+9, color="red", lw=0.50)
ax.plot(x, x+10, color="red", lw=1.00)
ax.plot(x, x+11, color="red", lw=2.00)   # linewidth/lw same thing

##### Linestyle (ls)
ax.plot(x, x+16, color="black", lw=3, linestyle='-')
ax.plot(x, x+17, color="black", lw=3, ls='-.')
ax.plot(x, x+18, color="black", lw=3, ls=':')
ax.plot(x, x+19, color="black", lw=3, ls='steps')

line, = ax.plot(x, x+20, color="black", lw=3)
line.set_dashes([1, 10, 10, 3]) # custom dash (line length, space length, ...)

##### Markers
ax.plot(x, x+25, color="blue", lw=0, ls='-', marker='+')
ax.plot(x, x+26, color="blue", lw=0, ls='--', marker='o')
ax.plot(x, x+27, color="blue", lw=0.1, ls='-', marker='s')
ax.plot(x, x+28, color="blue", lw=0.1, ls='--', marker='1')

##### Marker: size, facecolor (inner), edgecolor (outer), edgewidth
ax.plot(x, x+33, color="black", lw=1, ls='-', marker='o', markersize=2)
ax.plot(x, x+36, color="black", lw=1, ls='-', marker='o', markersize=4)
ax.plot(x, x+39, color="black", lw=1, ls='-', marker='o', markersize=8, markerfacecolor="red")
ax.plot(x, x+42, color="black", lw=1, ls='-', marker='s', markersize=8,
        markerfacecolor="yellow", markeredgewidth=3, markeredgecolor="green")

## Plot range

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x,y, color="black")

##### Crop graph in x and y
ax.set_xlim([0,2])
ax.set_ylim([0,1])

##### Adjust to window
ax.axis('tight')


# Seaborn
##### http://seaborn.pydata.org/

tips = sns.load_dataset('tips') # seaborn has built-in datasets as template
tips

## Distribution Plots
### Histogram plot
##### univariate set of observations
sns.distplot(tips['total_bill'])

sns.distplot(tips['total_bill'], kde=False) # KDE: kernel density estimate
sns.distplot(tips['total_bill'], kde=False, bins=30)

### Joint plot
##### join two dist plots with bivariant data
sns.jointplot(x='total_bill', y='tip', data= tips) # default is kind="scatter"
sns.jointplot(x='total_bill', y='tip', data= tips, kind='hex')
sns.jointplot(x='total_bill', y='tip', data= tips, kind='reg')
sns.jointplot(x='total_bill', y='tip', data= tips, kind='kde')

### Pair plot
##### pair-wise relationship across an entire dataframe
sns.pairplot(tips)
sns.pairplot(tips, hue='sex') # for categorical columns
sns.pairplot(tips, hue='sex', palette='coolwarm')

### Rug plot
##### data is unbinned, intensity is seen by color density
sns.rugplot(tips['total_bill'])

### KDE Plot
##### built from sums of rug plot gaussian curves
sns.kdeplot(tips['total_bill'])


## Categorical Plots
### Box plot
##### distribution of quantitative data, comparing qualitative factors
sns.boxplot(x='day', y='total_bill', data=tips)
sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker')

sns.boxplot(data=tips)
sns.boxplot(data=tips, orient='h') # horizontal

### Violin plot
##### similar to box, but also shows density of underlying distribution and includes outliers
sns.violinplot(x='day', y='total_bill', data=tips)
sns.violinplot(x='day', y='total_bill', data=tips, hue='sex')
sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', split=True) # adds a third category visually

### Strip plot
##### scatter plot where one var is categorical
sns.stripplot(x='day', y='total_bill', data=tips) # default is: jitter=True
sns.stripplot(x='day', y='total_bill', data=tips, hue='sex')
sns.stripplot(x='day', y='total_bill', data=tips, hue='sex', split=True)

### Swarm plot
##### a combination of strip+violin. don't use in very large datasets
sns.swarmplot(x='day', y='total_bill', data=tips)

sns.violinplot(x='day', y='total_bill', data=tips)
sns.swarmplot(x='day', y='total_bill', data=tips, color='black') # run this together

### Bar plot
##### aggregate data based on some function. default: mean
sns.barplot(x='sex', y='total_bill', data= tips)
sns.barplot(x='sex', y='total_bill', data= tips, estimator=np.std)

### Count plot
##### bar plot, but the estimator is counting occurrences
sns.countplot(x='sex', data=tips)

### Factor plot
##### you can express all plots above with the kind attribute
sns.factorplot(x='day', y='total_bill', data=tips)
sns.factorplot(x='day', y='total_bill', data=tips, kind='bar')
sns.factorplot(x='day', y='total_bill', data=tips, kind='violin')


## Matrix Plots
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')

### Heatmap
##### rows and columns MUST be variable assigned, not just row ID'd
tc = tips.corr()
sns.heatmap(tc)
sns.heatmap(tc, annot=True)
sns.heatmap(tc, annot=True, cmap='coolwarm')

flights
fp = flights.pivot_table(index='month', columns='year', values='passengers')
sns.heatmap(fp)
sns.heatmap(fp, cmap='magma')
sns.heatmap(fp, cmap='magma', linecolor='white', linewidths=.1)

### Clustermap
##### hierarchical clustering of a heatmap based on similarity
sns.clustermap(fp)
sns.clustermap(fp, cmap='coolwarm')
sns.clustermap(fp, cmap='coolwarm', standard_scale=1)


## Grids
iris = sns.load_dataset('iris')
iris

### Pair Grid
###### basically the pair plot functionality, but customisable
sns.PairGrid(iris)

g = sns.PairGrid(iris)  # map everywhere
g.map(plt.scatter)

g = sns.PairGrid(iris)  # map locally
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)

### Facet Grid
###### plot two variables, but graphs separated by conditional pair col/row
g = sns.FacetGrid(data=tips, col='time', row='smoker')
g.map(sns.distplot, 'total_bill')

g = sns.FacetGrid(data=tips, col='time', row='smoker')
g.map(sns.scatterplot, 'total_bill', 'tip')

### Joint Grid
###### customisable version of joint plot
g = sns.JointGrid(x="total_bill", y="tip", data=tips)
g = g.plot(sns.regplot, sns.distplot)


## Regression Plots
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', markers=['o','v'],
           scatter_kws={'s':100})   # edit mpl under the hood with scatter_kws

sns.lmplot(x='total_bill', y='tip', data=tips, col='sex', row='time')  # like hue, but diff graphs


sns.lmplot(x='total_bill', y='tip', data=tips, col='sex', row='time', hue='smoker') # usually too much info, but also possible

sns.lmplot(x='total_bill', y='tip', data=tips, col='sex', row='time', hue='smoker',
           aspect=0.6, size=8) # change size of graph


## Aesthetics (http://seaborn.pydata.org/tutorial/aesthetics.html)
plt.figure(figsize=(16,9)) # change aspect ratio
sns.set_style('ticks')  # background style: white, dark, whitegrid, darkgrid, ticks
sns.countplot(x='sex',data=tips, palette='nipy_spectral') # change color
sns.despine(top=True, right=True)  # remove graph borders

##### set aspect ratio & size in templates using set_context
sns.set_context('poster', font_scale=.8) # context: paper, notebook, talk, poster
sns.set_style('ticks')
sns.countplot(x='sex',data=tips, palette='nipy_spectral')
sns.despine(top=True, right=True)


# Pandas Built-in Visualization
df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])

df.plot.hist()
df.plot.area()
df.plot.bar(stacked=True)
df.plot.line(x='a', y='b')
df.plot.scatter(x='a', y='b', c='b', cmap='coolwarm')
df.plot.hexbin(x='a',y='b',gridsize=25,cmap='Oranges')
df.plot.kde()
df.plot.density()


# Plotly and Cufflinks
##### for advanced stuff like moving average etc (github.com/santosjorge/cufflinks)
df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())
df2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})
df3 = pd.DataFrame({'x':[1,2,3,4,5],'y':[10,20,30,20,10],'z':[5,4,3,2,1]})

## types
df.iplot()
df.iplot(kind='scatter', x='A', y='B', mode='markers')
df2.iplot(kind='bar', x='Category', y='Values')
df.iplot('box')
df3.iplot('surface', colorscale='rdylbu')
df.iplot('hist')
df[['A', 'B']].iplot('spread')
df.iplot('bubble', x='A', y='B', size='C')
df.iplot('area', fill='True', opacity=1)
df.scatter_matrix()


## pass operations to data and plot them
df2.count().iplot(kind='bar')
df2.sum().iplot(kind='bar')

# Geographical Plotting with Plotly
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

data = dict(type = 'choropleth',
            locations = ['AZ','CA','NY'],
            locationmode = 'USA-states',
            colorscale= 'Portland',
            reversescale=False,
            text= ['text1','text2','text3'],  # what hovers over locations
            z=[1.0,2.0,3.0],    # values shown on colorscale
            colorbar = {'title':'Colorbar Title'},
            marker = dict(line = dict(color = 'rgb(0, 0, 0)', width=2))
            )
# National
layout = dict(title = 'Test geo graph',
              geo = dict(scope = 'usa',
                         showlakes = True,
                         lakecolor = 'rgb(85, 173, 240)',
                         #showframe = False
                         )
              )

choromap = go.Figure(data= [data], layout = layout)
iplot(choromap)
plot(choromap) # full size window

# World
layout = dict(title = 'Test geo graph',
              geo = dict(#scope = 'world',
                         showlakes = True,
                         #locationmode = "country names",
                         lakecolor = 'rgb(85, 173, 240)',
                         showframe = False,
                         projection = dict(type='mollweide')
                         )
              )

choromap = go.Figure(data= [data], layout = layout)

iplot(choromap)
plot(choromap) # full size window

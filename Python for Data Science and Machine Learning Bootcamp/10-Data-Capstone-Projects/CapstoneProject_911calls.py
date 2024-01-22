import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Asus\\github\\testspace\\Learning\\Python for Data Science and Machine Learning Bootcamp\\10-Data-Capstone-Projects\\911.csv")
df
df.info()
df.head()

# What are the top 5 zipcodes for 911 calls?
df['zip'].value_counts().head(5)


# What are the top 5 townships (twp) for 911 calls?
df['twp'].value_counts().head(5)


# Take a look at the 'title' column, how many unique title codes are there?
df['title'].nunique()


# In the titles column there are "Reasons/Departments" specified before the
# title code. These are EMS, Fire, and Traffic. Use .apply() with a custom
# lambda expression to create a new column called "Reason" that contains this string value.
df['title']
reason = df['title'].apply(lambda x: x.split(':')[0])
reason = np.array(reason)
reasoncol = pd.DataFrame(data=reason, index=None, columns=['reason'])
reasoncol
df2 = df.join(reasoncol)
df2


# What is the most common Reason for a 911 call based off of this new column?
df2['reason'].value_counts()


# Now use seaborn to create a countplot of 911 calls by Reason.
sns.countplot(data=df2, x='reason')


# Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column?
type(df2['timeStamp'][0])


# You should have seen that these timestamps are still strings. Use pd.to_datetime to convert the column from strings to DateTime objects.
df2['timeStamp'] = pd.to_datetime(df2['timeStamp'])

time = df2['timeStamp'].iloc[0]
time.hour


# Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week.
hour = df2['timeStamp'].apply(lambda x: x.hour)
month = df2['timeStamp'].apply(lambda x: x.month)
weekday = df2['timeStamp'].apply(lambda x: x.weekday_name)

hourcol = pd.DataFrame(data=np.array(hour), index=None, columns=['hour'])
monthcol = pd.DataFrame(data=np.array(month), index=None, columns=['month'])
weekdaycol = pd.DataFrame(data=np.array(weekday), index=None, columns=['weekday'])
df3 = df2.join([hourcol, monthcol, weekdaycol])
df3


# Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.

sns.countplot(data=df3, x='weekday', hue='reason')
sns.countplot(data=df3, x='month', hue='reason')
### NOTE: to relocate the legend
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)


# create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame.
byMonth = df3.groupby('month').count()



# Now create a simple plot off of the dataframe indicating the count of calls per month.
sns.lineplot(data=df3, x=byMonth.index, y=byMonth['timeStamp'])


# Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column.
byMonth.reset_index(inplace=True)
byMonth

sns.lmplot(data=byMonth, x='month', y='e')


# Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method.
df3['date'] = df3['timeStamp'].apply(lambda x: x.date())
df3


# Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.
byDate = df3.groupby('date').count()
byDate
sns.lineplot(data=byDate, x=byDate.index, y='e')


# Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call
byEMS = df3[df3['reason'] == 'EMS'].groupby('date').count()
byFire = df3[df3['reason'] == 'Fire'].groupby('date').count()
byTraffic = df3[df3['reason'] == 'Traffic'].groupby('date').count()

sns.lineplot(data=byEMS, x=byEMS.index, y='e').set_title('EMS')
sns.lineplot(data=byFire, x=byFire.index, y='e').set_title('Fire')
sns.lineplot(data=byTraffic, x=byTraffic.index, y='e').set_title('Traffic')


# restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week.
# I would recommend trying to combine groupby with an unstack method.
byWeekdayAndHour = df3.groupby(['weekday', 'hour']).count()
byWeekdayAndHour
df4 = byWeekdayAndHour.unstack(level=-1)['e']
df4


# Now create a HeatMap using this new DataFrame.
sns.heatmap(df4)


# Now create a clustermap using this DataFrame.
sns.clustermap(df4)


# Now repeat these same plots and operations, for a DataFrame that shows the Month as the column.
byWeekdayAndMonth = df3.groupby(['weekday', 'month']).count()
df5 = byWeekdayAndMonth.unstack(level=-1)['e']
sns.heatmap(df5)
sns.clustermap(df5)

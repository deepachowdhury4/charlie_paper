import xml.etree.ElementTree as ET
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import csv
from csv import reader

dates=[]
values=[]
cals_sum=0
official_days=[]
official_cals=[]
tot=0

#print(df['value'].head())

with open('wrist_data/step_count.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    next(csv_reader)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        #print(row[-1])
        if(row[-1]=='2020-11-01' or row[-1]=='2020-11-02' 
        or row[-1]=='2020-11-03' or row[-1]=='2020-11-04' or row[-1]=='2020-11-05'
        or row[-1]=='2020-11-06' or row[-1]=='2020-11-07' or row[-1]=='2020-11-08' or row[-1]=='2020-11-09' or
        row[-1]=='2020-11-10' or row[-1]=='2020-11-12'):
            dates.append(row[-1])
            values.append(row[-2])

first=dates[0]
for i in range(0,len(dates)):
    if(dates[i]!=first):
        first=dates[i]
        #print(first)
        #print(cals_sum)
        official_days.append(first)
        official_cals.append(cals_sum)
        cals_sum=0
    else:
        cals_sum+=float(values[i])

#print(official_cals)
#print(official_days)


ax=plt.bar(official_days[:7], official_cals[:7], color ='purple',
        width = 0.5)

# Set tick font size
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
 
plt.xlabel("Dates", fontsize=30)
plt.ylabel("Steps Walked/Ran (steps)", fontsize=30)
plt.title("Steps Walked/Ran vs. Days", fontsize=30)
plt.show()

import xml.etree.ElementTree as ET
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# create element tree object 
tree = ET.parse('wrist_data\export.xml') 

# for every health record, extract the attributes into a dictionary (columns). Then create a list (rows).
root = tree.getroot()
record_list = [x.attrib for x in root.iter('Record')]

# create DataFrame from a list (rows) of dictionaries (columns)
data = pd.DataFrame(record_list)



#print(data['startDate'][1])
#print(data['creationDate'][1])
#print(data['endDate'][1])
#print(len(data))

# proper type to dates
for col in ['creationDate', 'startDate', 'endDate']:
    data[col] = pd.to_datetime(data[col])
    

# value is numeric, NaN if fails
data['value'] = pd.to_numeric(data['value'], errors='coerce')


# some records do not measure anything, just count occurences
# filling with 1.0 (= one time) makes it easier to aggregate
data['value'] = data['value'].fillna(1.0)

# shorter observation names: use vectorized replace function
data['type'] = data['type'].str.replace('HKQuantityTypeIdentifier', '')
data['type'] = data['type'].str.replace('HKCategoryTypeIdentifier', '')

#print(data.type.unique())

basal_data=pd.DataFrame(data[data['type'].str.contains("BasalEnergyBurned")])
calorie_data=pd.DataFrame(data[data['type'].str.contains("ActiveEnergyBurned")])
#basal and active calories sum
new_basal_data=basal_data.iloc[30000:]
new_calorie_data=calorie_data.iloc[30000:]
#new_basal_data.to_csv('basal_data.csv')
#new_calorie_data.to_csv('calorie_data.csv')

#step count sum
step_count=pd.DataFrame(data[data['type'].str.contains("StepCount")])
new_step_count=step_count.iloc[30000:]
#new_step_count.to_csv('step_count.csv')

#distance walking and running sum
distance=pd.DataFrame(data[data['type'].str.contains("DistanceWalkingRunning")])
new_distance=distance.iloc[30000:]
#new_distance.to_csv('distance.csv')

#exercise time sum
exercise=pd.DataFrame(data[data['type'].str.contains("AppleExerciseTime")])
new_exercise=exercise.iloc[30000:]
#new_exercise.to_csv('exercise.csv')
#exercise.to_csv('exercise.csv')


#print(new_basal_data.keys())
# pivot and resample
pivot_df = pd.pivot_table(new_basal_data, index='endDate', columns='type', values='value')
df = pivot_df.resample('W').agg({"BasalEnergyBurned": sum})

#print(pivot_df.head())

# create plot for basal and active calories lost
fig = plt.figure(figsize=(8,4)) 
sns.barplot(data=df, x='type', y='BasalEnergyBurned')
#sns.lineplot(data=df['ActiveEnergyBurned'], color='purple', linewidth=1)
#sns.lineplot(data=df['StepCount'], color='purple', linewidth=1)
#sns.lineplot(data=df['DistanceWalkingRunning'], color='purple', linewidth=1)
#sns.lineplot(data=df['AppleExerciseTime'], color='purple', linewidth=1)
#plt.bar(df['endDate'], 6, 5, df['BasalEnergyBurned'])
#plt.xlabel()
#plt.ylabel()
#plt.show()



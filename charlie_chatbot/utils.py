from datetime import time
import random #random choice from possible answers
import json
import pickle
import pandas as pd
import torch
from torch.functional import meshgrid

df = pd.read_pickle('data_processed.pickle')

df2=pd.read_pickle('recipes.pickle')


with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

def get_calendar(num,day):
    day_list=(df[df["Calendar #"]==int(num)]).values.tolist()
    for i in range(len(day_list)):
        if day==day_list[i][1]:
            return (day_list[i][2], day_list[i][3])

def get_dataframe(a,b):
    recipes=pd.read_pickle('recipes.pickle') 
    cals_weight_hours=pd.read_pickle('data_processed.pickle')
    exercise=pd.read_pickle('exercise_data.pickle')
    recipe_nums=recipes.copy()
    id_list=recipes['ID'].to_list()
    name_list=recipes['Recipe Name'].to_list()
    url_list=recipes['Url'].to_list()
    del recipes['ID']
    del recipes['Recipe Name']
    del recipes['Url']
    def colors_nums(x):
        if str(x) == 'green':
            return 3
        elif str(x) == 'orange':
            return 2
        else:
            return 1
    #daily cals and hours for each day after exercise
    user_weight=(cals_weight_hours.loc[cals_weight_hours['Calendar #'] == a, 'Weight'].iloc[b-1])
    user_cals=(cals_weight_hours.loc[cals_weight_hours['Calendar #'] == a, 'Calories Lost Daily'].iloc[b-1])
    user_hours=(cals_weight_hours.loc[cals_weight_hours['Calendar #'] == a, 'Hours Left'].iloc[b-1])
    multiplier1=float(exercise.loc[exercise['Activity, Exercise or Sport (1 hour)']=='Walking 3.0 mph, moderate', 'Calories per Ib'])
    multiplier2=float(exercise.loc[exercise['Activity, Exercise or Sport (1 hour)']=='Running, general', 'Calories per Ib'])
    if 24 > user_hours > 4:
        lost_cals=multiplier1*user_weight
        total_cals=user_cals+lost_cals
        total_hours=user_hours-1
        cals_weight_hours['Calories Lost Daily'] = cals_weight_hours['Calories Lost Daily'].replace([user_cals],total_cals)
        cals_weight_hours['Hours Left'] = cals_weight_hours['Hours Left'].replace([user_hours],total_hours)
        t=1
        #make sure ot recommend walking
    else:
        lost_cals=multiplier2*user_weight*0.5
        total_cals=user_cals+lost_cals
        total_hours=user_hours-0.5
        cals_weight_hours['Calories Lost Daily'] = cals_weight_hours['Calories Lost Daily'].replace([user_cals],total_cals)
        cals_weight_hours['Hours Left'] = cals_weight_hours['Hours Left'].replace([user_hours],total_hours)
        t=2
        #make sure to recommend running
    recipe_nums['FSA Lights: Fat']=recipes['FSA Lights: Fat'].apply(colors_nums)
    recipe_nums['FSA Lights: Salt']=recipes['FSA Lights: Salt'].apply(colors_nums)
    recipe_nums['FSA Lights: Saturates']=recipes['FSA Lights: Saturates'].apply(colors_nums)
    recipe_nums['FSA Lights: Sugar']=recipes['FSA Lights: Sugar'].apply(colors_nums)
    daily_cals=[total_cals]*len(name_list)
    hours_left=[total_hours]*len(name_list)
    recipe_nums.insert(14,'Daily Cals Lost', daily_cals)
    recipe_nums.insert(15,'Hours Left', hours_left)
    recipe_nums['Total Calories']=recipes['Total Calories'].apply(lambda x: x-int(total_cals/3))
    recipe_nums.reset_index(inplace=True)
    #recipes organizaton to use for training data
    recipes.reset_index(inplace=True)
    recipes['Total Calories']=recipes['Total Calories'].apply(lambda x: x-int(total_cals/3))
    recipes['FSA Lights: Fat']=recipes['FSA Lights: Fat'].apply(colors_nums)
    recipes['FSA Lights: Salt']=recipes['FSA Lights: Salt'].apply(colors_nums)
    recipes['FSA Lights: Saturates']=recipes['FSA Lights: Saturates'].apply(colors_nums)
    recipes['FSA Lights: Sugar']=recipes['FSA Lights: Sugar'].apply(colors_nums)
    cal_list=recipes['Total Calories'].to_list()
    exercise_list=[t]*len(cal_list)
    ye_nolist=[]
    for i in cal_list:
        if float(i)>600 and float(i)<700:
            ye_nolist.append(1.0)
        #divide serving by 2
        elif (float(i)/2)>600 and (float(i)/2)<700:
            ye_nolist.append(0.5)            
        #divide serving by 3
        elif (float(i)/3)>600 and (float(i)/3)<700:
            ye_nolist.append(0.3)            
        else:
            ye_nolist.append(0.0)            
    recipes.insert(11,'Cals Usable', ye_nolist)
    recipes.insert(13,'Hours Left', hours_left)
    recipes.insert(14,'Exercise',exercise_list)
    return recipes

def weighted_rating(x):
    #using FSA Lights
    fat_fsa=x['FSA Lights: Fat']
    salt_fsa=x['FSA Lights: Salt']
    saturate_fsa=x['FSA Lights: Saturates']
    sugar_fsa=x['FSA Lights: Sugar']
    percent_healthy=((fat_fsa+salt_fsa+saturate_fsa+sugar_fsa)/12)*100
    
    #using actual numbers, each recipe should be equal to about 30-40% of nutrition 
    fat_100g=x['Fats per 100g'] #60.5 
    protein_100g=x['Proteins per 100g'] #50 g
    salt_100g=x['Salts per 100g'] #2.3 g
    saturate_100g=x['Saturates per 100g'] #20 g
    sugar_100g=x['Sugars per 100g'] #50 g
    #need to encorporate if statements to determine actual fats user would use
    if x['Cals Usable']==1.0:
        percent_nutrition=100*((fat_100g/60.5)+(protein_100g/50)+(salt_100g/2.3)+(saturate_100g/20)+(sugar_100g/50))/5
    elif x['Cals Usable']==0.5:
        percent_nutrition=100*((fat_100g/60.5)+(protein_100g/50)+(salt_100g/2.3)+(saturate_100g/20)+(sugar_100g/50))/10
    elif x['Cals Usable']==0.3:
        percent_nutrition=100*((fat_100g/60.5)+(protein_100g/50)+(salt_100g/2.3)+(saturate_100g/20)+(sugar_100g/50))/15
    percent_nutrition=round(percent_nutrition,4)
    #percent nutrition wont be used until later but both should be above 80%
    return pd.Series([percent_healthy, percent_nutrition])

def get_choices(a,b):
    x=get_dataframe(a,b)
    m=x['FSA Lights: Fat'].quantile(0.90)
    x = x[x['Cals Usable'] != 0]
    y=x.iloc[0]['Hours Left']
    q_fats = x.copy().loc[x['FSA Lights: Fat'] >= m]
    q_salts=x.copy().loc[x['FSA Lights: Salt'] >= m]
    q_saturates=x.copy().loc[x['FSA Lights: Saturates'] >= m]
    q_sugar=x.copy().loc[x['FSA Lights: Sugar'] >= m]
    frames = [q_fats, q_salts, q_saturates, q_sugar]
    result = pd.concat(frames)
    result['FSA score'], result['score']= result.apply(weighted_rating, axis=1, result_type='expand').T.values

    result = result.sort_values('score', ascending=False)
    result.drop_duplicates(subset ="index",
                        keep = False, inplace = True)


    #drops all unecessary scores that don't fit range of nutritional values 
    #1 percent error for nutritional value and >50 for fsa lights
    incorrect_score=result[result['score']>33].index
    result.drop(incorrect_score, inplace = True)
    incorrect1_score=result[result['score']<27].index
    result.drop(incorrect1_score, inplace = True)
    incorrect_fsascore=result[result['FSA score']<50].index
    result.drop(incorrect_fsascore, inplace = True)

    official_recipes=pd.read_pickle('recipes.pickle') 
    official_recipes['Time'] = official_recipes['Time'].apply(lambda x: 'True' if (x < y and x!=0) else 'False')
    #will add option for time as more data starts to come in
    official_recipes = official_recipes.loc[result.index, :]
    official_recipes['exercise']=pd.Series(x['Exercise'])
    official_recipes['cals usable']=pd.Series(x['Cals Usable'])
    return official_recipes

def get_time(hours1,hours2,hours3, id1, id2, id3):
    df2.loc[df2.ID == id1, 'Time'] = hours1
    df2.loc[df2.ID == id2, 'Time'] = hours2
    df2.loc[df2.ID == id3, 'Time'] = hours3
    df2.to_pickle('recipes.pickle')




import random #random choice from possible answers
import json
import csv
import torch
from torch.functional import meshgrid
from random import choice

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents2.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data2.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
convos=[]
charlie=[]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "CHARLIE"

#user will be asked about calendar day and number they want to look at and then will be recommended food based on nlp 
#make list for breakfast, lunch, and dinner and then choose one randomly 

def get_foods(a,b):
    official=get_choices(a,b)
    recipes=pd.read_pickle('recipes.pickle') 
    recipe_name=official['Recipe Name'].to_list()
    breakfast_list=[]
    lunch_list=[]
    dinner_list=[]
    idk_list=[]
    snack_list=[]
    dessert_list=[]
    x,y,z,s,d=[],[],[],[],[]
    breakfast,lunch,dinner,snack,dessert=[],[],[],[],[]
    for msg in recipe_name:
        lang=msg
        sentence = tokenize(msg)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    if intent["tag"] == "breakfast":
                        breakfast_list.append(lang)
                        #do all the functions here and utils stuff
                    elif intent["tag"] == "lunch":
                        lunch_list.append(lang)
                    elif intent["tag"] == "dinner":
                        dinner_list.append(lang)
                    elif intent["tag"] == "snack":
                        snack_list.append(lang)
                    elif intent["tag"] == "dessert":
                        dessert_list.append(lang)   
                    else:
                        idk_list.append('Unknown')
    if len(breakfast_list) !=0:
        x=choice(breakfast_list)
        breakfast=official.loc[official['Recipe Name']==x].values.tolist()
    if len(lunch_list) !=0:
        y=choice(lunch_list)
        lunch=official.loc[official['Recipe Name']==y].values.tolist()
    if len(dinner_list) !=0:
        z=choice(dinner_list)
        dinner=official.loc[official['Recipe Name']==z].values.tolist()
    if len(snack_list) !=0:
        s=choice(snack_list)
        snack=official.loc[official['Recipe Name']==s].values.tolist()
    if len(dessert_list) !=0:
        d=choice(dessert_list)
        dessert=official.loc[official['Recipe Name']==d].values.tolist()
    return breakfast, lunch, dinner, snack, dessert

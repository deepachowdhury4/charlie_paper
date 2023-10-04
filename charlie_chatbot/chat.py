import random #random choice from possible answers
import json
import csv
from fractions import Fraction
import torch
from torch.functional import meshgrid

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from utils import *
from recipes_chat import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
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

list=[]

def get_response(msg):
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
                if intent["tag"] == "introduction/help":
                    return (random.choice(intent['responses']))
                    #do all the functions here and utils stuff
                if intent["tag"] == "calendar information":
                    info=str(lang)[14:]
                    new_info=info.split(":",1)
                    caleninfo=int(new_info[0]) #which calendar to look at
                    dayinfo=int(new_info[1])
                    hours, cals=get_calendar(caleninfo,dayinfo)
                    #add to pickle file???
                    #maybe add something like based on hours left and nutritional information, here is the best recipe... will you have neough time to make it?
                    return ("You have "+str(hours)+" hours left in the day for meal and fitness prep. You have also lost "+str(cals)+" calories today based on your schedule.")
                elif intent['tag']=="food information": #this will be for choosing a recipe that will be in fat and salts
                    info=str(lang)[28:]
                    new_info=info.split(":",1)
                    caleninfo=int(new_info[0]) #which calendar to look at
                    dayinfo=int(new_info[1])
                    # need tp be 4 lists and then choose random recipe from those four lists
                    breakfast, lunch, dinner, snack, dessert=get_foods(caleninfo,dayinfo)
                    nameb, urlb,idb,servingb=breakfast[0][1], breakfast[0][12], breakfast[0][0], breakfast[0][15]
                    namel, urll, idl, servingl=lunch[0][1], lunch[0][12], lunch[0][0], lunch[0][15]
                    named, urld, idd, servingd=dinner[0][1], dinner[0][12], dinner[0][0], dinner[0][15]
                    if servingb==0.3:
                        servingb=Fraction(1,3)
                    if servingb==0.5:
                        servingb=Fraction(1,2)
                    if servingl==0.3:
                        servingl=Fraction(1,3)
                    if servingl==0.5:
                        servingl=Fraction(1,2)
                    if servingd==0.3:
                        servingd=Fraction(1,3)
                    if servingd==0.5:
                        servingd=Fraction(1,2)
                    if breakfast[0][14]==1:
                        exercise = 'Walk for 1 hour.'
                    else:
                        exercise='Run for 30 minutes.'
                    return ("\nCHARLIE recommend this fitness and diet plan:\nEXERCISE:"+exercise+"\nBREAKFAST:"+nameb+
                                "\nURL:"+urlb+"\nServing Size:"+str(servingb)
                                +"\n\nLUNCH:"+namel+"\nURL:"+urll+"\nServing Size:"
                                +str(servingl)+"\n\nDINNER:"+named+"\nURL:"+urld+
                                "\nServing Size:"+str(servingd))
                elif intent['tag']=="time information":
                    info=str(lang)[7:]
                    new_info=info.split(",",2)
                    hours=new_info[0].split(":",3)
                    ids=new_info[1].split(":",3)
                    get_time(hours[0], hours[1], hours[2], ids[0], ids[1], ids[2])
                    return (random.choice(intent['responses']))
                else:
                    return (random.choice(intent['responses']))
                
                
                
    else:
        return "I do not understand..."





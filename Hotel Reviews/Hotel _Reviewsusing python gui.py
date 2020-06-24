import tkinter
 
window = tkinter.Tk()
 
# to rename the title of the window window.title("GUI")
 
# pack is used to show the object in the window
 
label = tkinter.Label(window, text = "Hello World!").pack()
 
window.mainloop()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
dataset=pd.read_csv('hotel-reviews.csv')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
def cleanData(sentence):
    processedList = ""
    
    # convert to lowercase, ignore all special characters - keep only alpha-numericals and spaces (not removing full-stop here)
    sentence = re.sub(r'[^A-Za-z0-9\s.]',r'',str(sentence).lower())
    sentence = re.sub(r'\n',r' ',sentence)
    
    # remove stop words
    sentence = " ".join([word for word in sentence.split() if word not in stopWords])
    
    return sentence
corpus=[]
for i in range(38932):
    x=cleanData(dataset['Description'][i])
    corpus.append(x)
from sklearn.model_selection import train_test_split
y=dataset['Is_Response']

X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LogisticRegression 
from sklearn.feature_extraction.text import TfidfVectorizer
tvec=TfidfVectorizer()
clf2=LogisticRegression(solver="lbfgs")
from sklearn.pipeline import Pipeline
model = Pipeline([('vectorizer',tvec),('classifier',clf2)])
model.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
confusion_matrix(y_pred,y_test)
example=['Awesome']
ans=model.predict(example)
print(ans)
import tkinter as tk


def show_entry_fields():
    print(e1.get())
    temp=[]
    temp.append(e1.get())
    ans1=model.predict(temp)
    print(ans1)

master = tk.Tk()
tk.Label(master, 
         text="Enter your comments").grid(row=0) 

e1 = tk.Entry(master)


e1.grid(row=0, column=1)


tk.Button(master, 
          text='Quit', 
          command=master.quit).grid(row=3, 
                                    column=0, 
                                    sticky=tk.W, 
                                    pady=4)
tk.Button(master, 
          text='Show', command=show_entry_fields).grid(row=3, 
                                                       column=1, 
                                                       sticky=tk.W, 
                                                       pady=4)

tk.mainloop()
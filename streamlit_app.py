#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


import pickle
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer=TfidfVectorizer()

#svm_model = joblib.load('svm_model.pkl')

# Import the Flask library
from flask import Flask, render_template, request, jsonify
#with open('model_svm1.pkl','rb') as f:
    #model_svm1=pickle.load(f)
with open ('svm_model2.pkl','rb') as f:
    svm_model2=pickle.load(f)
with open('tfidf_vec.pkl', 'rb') as f:
    tfidf_vec= pickle.load(f)
#vectorizer=TfidfVectorizer()
#vectorizer.fit(svm_model2)
# Create a Flask application
app = Flask(__name__,template_folder='templates')

# Define the 



@app.route('/')
def home():
    # Define an initial welcome message
    messages = [{'type': 'bot', 'text': 'Hi there, how can I help you today?'}]
    return render_template('layout.html', messages=messages)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get the user's question from the form
    question = request.form['question']
    
    # Use the vectorizer to transform the question into a vector
    question_vec = tfidf_vec.transform([question])
    
    # Use the SVM model to generate an answer
    answer = svm_model2.predict((question_vec)[0])
    print('Predicted class:',answer)
    print('Answer:',get_answer(answer))
    # Add the user's question and the chatbot's answer to the chatlog
    messages = [
        {'type': 'user', 'text': question},
        {'type': 'bot', 'text': get_answer(answer)}
    ]
    
    # Render the HTML template with the updated chatlog
    return render_template('layout.html', messages=messages)
    #return render_template('layout.html', messages=messages)
    
    
def get_answer(category):
    
    
        
    return'this is the answer for category {}'.format(category)
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




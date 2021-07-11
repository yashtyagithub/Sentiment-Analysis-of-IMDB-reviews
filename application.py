from flask import Flask, render_template, request
import re
import pickle
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification

application = Flask(__name__)




global model
global tokenizer
global sentiment
global score

model=TFDistilBertForSequenceClassification.from_pretrained('bert_model')
with open('bert_tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

def prep(text):
    tokens =tokenizer.encode_plus(text,                      
                add_special_tokens = True, 
                max_length = 128, 
                pad_to_max_length = True, 
                truncation=True,
                return_attention_mask = True, 
                return_token_type_ids=False,
                return_tensors='tf'
            )
    return{
        'input_ids':tf.cast(tokens['input_ids'],tf.int64),
        'attention_mask':tf.cast(tokens['attention_mask'],tf.int64)
        }

    
@application.route('/',methods=['Get'])
def index():
    return render_template('index.html')

@application.route('/', methods = ['POST'])
def predict():
    if request.method=='POST':
        text = request.form['text']
        text1=prep(text)
        prob=model.predict(text1)
        neg_prob=tf.nn.sigmoid(prob.logits)[0][0]
        pos_prob=tf.nn.sigmoid(prob.logits)[0][1]
        if(pos_prob>neg_prob):
            score=round(pos_prob.numpy()*100,2)
            sentiment=1
        else:
            score=round(neg_prob.numpy()*100,2)
            sentiment=0
                
    return render_template('index.html',sentiment=sentiment,text=text,score=score)




if __name__ == "__main__":
    application.run(debug=True)
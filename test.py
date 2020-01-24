import numpy as np
import pandas as pd

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from keras.models import load_model
from keras.preprocessing import image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
import torch
import pandas as pd
import io

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras import backend as K
# Define a flask app
# Model saved with Keras model.save()
# MODEL_PATH = 'model4.h5'
# # Load your trained model
# model = TheModelClass(*args, **kwargs);
model = torch.load(r"C:\Users\Anup\Desktop\NLP\keras-flask-deploy-webapp\model_bert_old.pt", map_location=torch.device('cpu'))
model.eval()
# #data
df = pd.read_csv(r"C:\Users\Anup\Desktop\NLP\keras-flask-deploy-webapp\data.csv")
t5 = df['sentence']
h2 = df['label']
bert_t5 = ["[CLS] " + sentence + " [SEP]" for sentence in t5]
device = torch.device('cpu')


# char_dict_csv = pd.read_csv('./hello.csv')
# char_dict = dict()
# # convert string to lower case 
# for i in range(len(char_dict_csv['target'])):
#     char_dict[char_dict_csv['target'][i]] = i+1

# # convert string to lower case 

# train_texts, test_texts, train_classes, test_classes = train_test_split(t5, h2, test_size=0.1, random_state=1)

# train_texts = [s.lower() for s in train_texts] 
# test_texts = [s.lower() for s in test_texts] 

# # Tokenizer
# tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
# tk.word_index = char_dict.copy() 
# tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

# print(char_dict)


# 'मेरा मोदी-मीटर है। आपका मोदी-मीटर क्या है इस लिंक पर अभी जांचें और नमो टी-शर्ट जीतने का मौका पाएं'
print("hi")



def model_predict(chat, model):
   
    sen=[chat]
    df5=pd.DataFrame()
    df5["sentence"]=sen

    sentences = df5.sentence.values

    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    MAX_LEN = 128
    # batch_size = 32

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    validation_inputs = torch.tensor(input_ids)
    print(validation_inputs)
    attention_masks = torch.tensor(attention_masks)
    print(validation_inputs)

    batch = (validation_inputs.to(device),attention_masks.to(device))

    print(batch)
    b_input_ids, b_input_mask = batch
    with torch.no_grad():
        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = logits.detach().cpu().numpy()
    # print(logits)

    return np.argmax(logits,axis=1).flatten()[0]

print("hi")
print(model_predict('पुलवामा झड़प में पत्थरबाजों की मौत की हालत गंभीर से ज्यादा घायल आतंकी भी ढेर डाउनलोड करें', model))
accuracy = 0

for i in range(len(df)):
    if (model_predict(bert_t5[i], model) == h2[i]):
        if(accuracy%100 ==0):
            print(accuracy/(i+1))
        accuracy += 1

print(accuracy/len(df))
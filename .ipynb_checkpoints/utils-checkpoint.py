import sys, time, os, warnings 

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

import string
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
from collections import OrderedDict

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16

import time
import re
import gensim, logging
import PIL

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Location of the Flickr8k images and caption files
dataset_image_path ="flickr8k/Images/"
dataset_text_path  ="flickr8k/captions.txt" 
wanted_shape = (224,224,3)

# To obtain the text dataset corresponding to images
df_texts = pd.read_csv(dataset_text_path, sep=",") #["image","caption"] 
n_img = df_texts.count()/5 # 40455/5 
unique_img = pd.unique(df_texts["image"])# 8091 unique images

# Function to crop images
def crop_center(img):
    cropx, cropy, _ = wanted_shape
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

# Images saved to df_images as (224,224,3) for each unique img in the dataset
def load_img_from_ds(image_name):
    #PREPROCESSING
    img =  img_to_array(load_img(dataset_image_path+image_name, target_size=wanted_shape))
    #img = crop_center(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    #print(f"shape img input {image_name} : {img.shape}")
    return img

#Text processing
def process_sentence(sentence):
    #Text preparation
    def remove_stopwords(caption, stop_words=stop_words):
        word_tokens = word_tokenize(caption)  
        filtered_caption = " ".join([w for w in word_tokens if w not in stop_words])
        return filtered_caption

    # Add start and end sequence token
    def add_start_end_seq_token(txt):
        return 'startseq ' + txt + ' endseq'

    def text_pipeline(caption):
        # lowercase
        caption = caption.lower()
        caption = remove_stopwords(caption)
        # remove some punctuations
        caption = caption.replace(".", "")
        caption = caption.replace(",", "")
        # remove numeric values
        caption = re.sub("\d+", "", caption)
        caption = add_start_end_seq_token(caption)
        return caption
    
    return text_pipeline(sentence)


# Change character vector to integer vector
# Embedding
def character_to_integer_vector(df_texts):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df_texts) #list of captions to train on
    #print("size of the dictionary : " + len(tokenizer.word_index) +1)
    return tokenizer.texts_to_sequences(df_texts)
'''
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
'''

# train word2vec on the two sentences
def word2vec(df_texts,model):
    # Need a cleaned_tokenized category in df_texts
    return np.array([model[w] for w in df_texts["cleaned_tokenized"]])

#START WORD2VEC
#print(model["girl"].shape)
#print(model["boy"].shape)
#print(f"Similarité : {model.similarity('girl', 'boy')}")
#END WORD2VEC

def finalpreprocessing(dftext, dfimage, vocab_size):
    print("# captions/images = {}".format(len(dftext)))
    assert(len(dftext)==len(dfimage)) # return error if len(text) != len(image)

    maxlen = np.max([len(text) for text in dftext])
    Xtext, Ximage, ytext = [], [], []
    for text, image in zip(dftext, dfimage):
        for i in range(1, len(text)):
            in_text, out_text = text[:i], text[i]
            in_text = pad_sequences([in_text], maxlen=maxlen).flatten()
            out_text = to_categorical(out_text, num_classes = vocab_size)
            Xtext.append(in_text)
            Ximage.append(image)
            ytext.append(out_text)
    Xtext = np.array(Xtext)
    Ximage = np.array(Ximage)
    ytext = np.array(ytext)
    return(Xtext, Ximage, ytext)

# Split dataset
def split_test_val_train(df, Ntest, Nval):
    return(df[:Ntest],
           df[Ntest:Ntest+Nval],
           df[Ntest+Nval:])
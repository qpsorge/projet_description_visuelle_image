from utils import *
from keras import layers 
from tensorflow.keras.models import Sequential, Model, Model
from tensorflow.keras.layers import LSTM, Embedding, Input, add

"""
CHARGEMENT DES DONNES ET CONFIGURATION
"""
# Location of the Flickr8k images and caption files
dataset_image_path ="flickr8k/Images/"
dataset_text_path  ="flickr8k/captions.txt" 
# Wanted shape for images
wanted_shape = (224,224,3)

# To obtain the text dataset corresponding to images
df_texts = pd.read_csv(dataset_text_path, sep=",") #["image","caption"] 
n_img = df_texts.count()/5 # 40455/5 
unique_img = pd.unique(df_texts["image"])# 8091 unique images

"""
PARTIE IMAGE AVEC GENERATION DU VGG16 PRE ENTRAINE : FEATURE MAPS 4096
"""
base_model = VGG16(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=wanted_shape, pooling=None, classes=1000,
    classifier_activation='softmax'
)
# Feature extraction
vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output) #end the modèle with a 4096 feature layer

charge_image, one_by_one = False, False # false to gain time when testing other parts
# To obtain the feature maps
if charge_image:
    feature_maps = np.array([model.predict(load_img_from_ds(unique_img[i])) for i in range(len(unique_img))])
    print(f"Shape des fm {feature_maps.shape}")
elif one_by_one:
    feature_maps=[]
    #for i in range(len(unique_img)):
    #    img = load_img_from_ds(unique_img[i])
    #    feature_map = model.predict(img)
    #    print(feature_map.shape)
    #    feature_maps.append(feature_map)
    #feature_maps=np.array(feature_maps)

"""
PARTIE TEXTE AVEC PREPROCESSING & WORD2VEC : EMBEDDINGS 4096 
"""
df_texts["cleaned"]=[process_sentence(s) for s in df_texts["caption"]]
#df_texts["tokenized"]=character_to_integer_vector(df_texts["cleaned"])
word2vec_model = gensim.models.Word2Vec([word_tokenize(w) for w in df_texts["cleaned"]], min_count=1, size=4096)
df_texts["embedded"] = word2vec(df_texts,word2vec_model)

# ACP pour faire correspondre les dimensions du texte et image > on laisse tomber for now

#dimages = np.array([np.array(load_img_from_ds(unique_img[i])) for i in range(len(unique_img))])
#dimages = np.array([np.array(load_img_from_ds(df_texts["image"][i])) for i in range(len(df_texts["image"]))])
dimages = feature_maps

# Split du dataset
prop_test, prop_val = 0.2, 0.2
N = len(df_texts["embedded"])
Ntest, Nval = int(N*prop_test), int(N*prop_val)

def split_test_val_train(df, Ntest, Nval):
    return(df[:Ntest],
           df[Ntest:Ntest+Nval],
           df[Ntest+Nval:])

# dt = true image caption cleaned
dt_test, dt_val, dt_train = split_test_val_train(df_text["embedded"], Ntest, Nval)
# di = true image array
di_test, di_val, di_train = split_test_val_train(dimages, Ntest, Nval)
# fnm = image_name
fnm_test, fnm_val, fnm_train = split_test_val_train(df_text["image"], Ntest, Nval)

vocab_size = len(model.wv.vocab)
print("vocab size : ", vocab_size)

Xtext_train, Ximage_train, ytext_train = finalpreprocessing(dt_train, di_train, vocab_size) 
Xtext_val, Ximage_val, ytext_val = finalpreprocessing(dt_val, di_val, vocab_size)

print(f"Training set : \n \tInput image : {Ximage_train.shape}\n\tInput text : {Xtext_train.shape}\n\tOutput text : {ytext_train.shape}")
'''
MODEL
'''
dim_embedding=64

# image input
input_img = Input(shape=(Ximage_train.shape[1],), name="InputImage") 
input_img = ( Dense(units=256,activation='relu',name="CompressedImageFeatures") )(input_img)
# text input
input_txt = Input(shape=(maxlen,), name="InputSequence")
input_txt = ( Embedding(vocab_size,dim_embedding, mask_zero=True))(input_txt)
input_txt = ( LSTM(units=8, activation="relu", name="CaptionFeatures") )(input_txt)

# Common part
common = add(input_txt, input_img)
common = Dense(256, activation='relu') (common)
common = Dense(vocab_size, activation='softmax')(common)

#Model
total_model  = Model(inputs=[input_image, input_txt],outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='adam')
print(model.summary())
'''
MODEL TRAINING
'''
hist = model.fit([Ximage_train, Xtext_train], ytext_train, epochs=5, verbose=2, batch_size=64, validation_data=([Ximage_val, Xtext_val], ytext_val))

'''
MODEL EVALUATION (VALIDATION AND TRAINING LOSS OVER EPOCHS)
'''
for label in ["loss", "val_loss"]:
    plt.plot(hist.history[label], label=label)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

'''
PREDICTION
'''

# 1 couche 256 LSTM ?
# A partir de combien de couce=hes c est ok 1 8 16 32 256 
# Temps d entrainement : compromis 
# Voir si dimensions pas trop grandes ?
# GRU ! :D mieux (3 params au lieu de 4)
# simpleRNN ? 
# Etude comparative : 3 RNN (simple, LSTM, GRU & Etude de perf)
# Limiter Dataset ! => entrainements en O(heure)

#tf.keras.utils.get_file(origin="lien", fname="nom_que_tu_veux_donner_au_fichier.zip", extract=True)
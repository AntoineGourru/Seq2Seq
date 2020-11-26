#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!curl -O http://www.manythings.org/anki/fra-eng.zip
#!unzip fra-eng.zip


# In[154]:


import pandas as pd

data = pd.read_csv("fra.txt",sep='\t',header= None)
ang = list(data[0])
fra = list(data[1])

from spacy.lang.en import English
from spacy.lang.fr import French

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)
nlp_f = French()
tokenizer_f = nlp_f.Defaults.create_tokenizer(nlp_f)

ang = [['<S>'] + [token.string.strip() for token in tokenizer(text.lower())] + ['</S>'] for text in ang]

fra = [['<S>'] + [token.string.strip() for token in tokenizer_f(text.lower())] + ['</S>'] for text in fra]

# In[155]:


from gensim.models import Word2Vec
import numpy as np

EMBEDDING_SIZE = 120
w2v = Word2Vec(ang, size=EMBEDDING_SIZE, window=10, min_count=1, negative=10, workers=10)
word_map = {}
word_map["<PAD>"] = 0
word_vectors = [np.zeros((EMBEDDING_SIZE,))]
for i, w in enumerate([w for w in w2v.wv.vocab]):
    word_map[w] = i+1
    word_vectors.append(w2v.wv[w])
word_vectors = np.vstack(word_vectors)

w2v = Word2Vec(fra, size=EMBEDDING_SIZE, window=10, min_count=1, negative=10, workers=10)
word_map_fr = {}
word_map_fr["<PAD>"] = 0
word_vectors_fr = [np.zeros((EMBEDDING_SIZE,))]
for i, w in enumerate([w for w in w2v.wv.vocab]):
    word_map_fr[w] = i+1
    word_vectors_fr.append(w2v.wv[w])
word_vectors_fr = np.vstack(word_vectors_fr)
i2w = dict(zip([*word_map_fr.values()],[*word_map_fr]))


# In[156]:


def pad(a):
    shape = len(a)
    max_s = max([len(x) for x in a])
    print(max_s)
    token = np.zeros((shape,max_s+1),dtype = np.int)
    mask  =  np.zeros((shape,max_s+1),dtype = np.int)
    for i,o in enumerate(a):
        token[i,:len(o)] = o
        mask[i,:len(o)] = 1
    return token,mask


# In[181]:


ang_tok,ang_mask = pad([[word_map[w] for w in text] for text in ang])
fra_tok,fra_mask = pad([[word_map_fr[w] for w in text] for text in fra])

fra_tok_t = fra_tok[:,1:]
fra_mask = fra_mask[:,1:]

fra_tok = fra_tok[:,:-1]


# In[182]:


import tensorflow as tf

batch_size = 64
train_data = tf.data.Dataset.from_tensor_slices((ang_tok,ang_mask,fra_tok,fra_tok_t,fra_mask)).batch(batch_size)


# In[190]:


from tensorflow.keras import layers,Model
from tensorflow.keras.initializers import Constant
import tensorflow as tf

class S2S(tf.keras.Model):
    def __init__(self,Win,Wout,i2w):
      
        super(S2S, self).__init__() 
        
        self.nv_in = Win.shape[0]
        self.r = Win.shape[1]
        self.nv_out = Wout.shape[0]
        
        self.i2w = i2w
        
        self.Win = layers.Embedding(self.nv_in,self.r)
        self.Win.build((None, ))
        self.Win.set_weights([Win])
        self.Win.trainable = False

        self.Wout = layers.Embedding(self.nv_out,self.r)
        self.Wout.build((None, ))
        self.Wout.set_weights([Wout])
        self.Wout.trainable = False
        
        self.encoder = layers.GRU(self.r, return_sequences=True, return_state=True,dropout=0.2)
        
        self.decoder = layers.GRU(self.r, return_sequences=True, return_state=True,dropout=0.2)
        
        self.mapper = layers.Dense(self.nv_out,activation = "softmax")

        self.attention = layers.Attention()

        self.H = layers.Dense(self.r*2,activation = "tanh")

        self.W = layers.Dense(self.r,activation = "relu")

    @tf.function
    def encode(self,x,x_mask):
        
        x = self.Win(x)
        x_mask = tf.cast(x_mask,dtype=bool)
    
        hidden_seq,hidden_last = self.encoder(x,mask=x_mask)

        return hidden_seq,hidden_last
    
    @tf.function
    def decode(self,encoder_seq,encoder_mask,decoder_last,context_last,x_out):

        x_out = self.Wout(x_out)

        input_decoder = tf.concat([x_out,context_last],2)

        encoder_mask = tf.cast(encoder_mask,dtype=bool)

        _,decoder_last = self.decoder(input_decoder, initial_state=decoder_last)

        decoder_last = tf.expand_dims(decoder_last,1)
        
        query = self.W(decoder_last)

        key = encoder_seq 
        value = encoder_seq      
        context_vector = self.attention([query,key,value],mask = [None,encoder_mask])

        probs = self.mapper(self.H(tf.concat([decoder_last,context_vector],2)))

        decoder_last = tf.squeeze(decoder_last)
        
        return probs,decoder_last,context_vector

 
    def generate(self,start_emb,x,x_mask):
        aout = []
        
        x = tf.expand_dims(x,axis=0)
        x_mask = tf.expand_dims(x_mask,axis=0)

        encoder_seq,hidden_last = model.encode(x,x_mask)
        context_last = tf.zeros([x.shape[0],model.r])
        context_last = tf.expand_dims(context_last,1)
       
        input_0 = tf.expand_dims(tf.expand_dims(start_emb,axis=0),axis=0)

        probs,hidden_last,context_last = model.decode(encoder_seq,x_mask,hidden_last,context_last,input_0)

        val,argval = tf.nn.top_k(tf.squeeze(probs), k=2, sorted=True, name=None)
        x_out = argval.numpy()[0]
        aout.append(self.i2w[x_out])
    
        for t in range(10):
            hidden_last = tf.expand_dims(hidden_last,axis=0)
            input_0 = tf.expand_dims(tf.expand_dims(x_out,axis=0),axis=0)
            probs,hidden_last,context_last = model.decode(encoder_seq,x_mask,hidden_last,context_last,input_0)
            
            val,argval = tf.nn.top_k(tf.squeeze(probs), k=2, sorted=True, name=None)
            x_out = argval.numpy()[0]
            aout.append(self.i2w[x_out])
                
        return aout

        



@tf.function
def compute_loss(model,loss_f,x,x_mask,x_out,y_out,y_out_mask):
    pro = []
    
    encoder_seq,hidden_last = model.encode(x,x_mask)
    context_last = tf.zeros([x.shape[0],model.r])
    context_last = tf.expand_dims(context_last,1)
    
    input_0 = tf.gather(x_out, [0], axis=1)
    probs,hidden_last,context_last = model.decode(encoder_seq,x_mask,hidden_last,context_last,input_0)

    pro.append(probs)

    for t in range(1,y_out.shape[1]):
        input_0 = tf.gather(x_out, [t], axis=1)
        probs,hidden_last,context_last = model.decode(encoder_seq,x_mask,hidden_last,context_last,input_0)
        pro.append(probs)
        
    pro = tf.concat(pro,1)

    y_true= tf.boolean_mask(y_out,y_out_mask)
    y_pred = tf.boolean_mask(pro,y_out_mask)
    
    
    return loss_f(y_true,y_pred),y_true,y_pred


@tf.function
def compute_apply_gradients(model, x,x_mask,x_out,y_out,y_out_mask, optimizer):
    loss_f = tf.keras.losses.CategoricalCrossentropy()

    with tf.GradientTape() as tape:
        
        loss = compute_loss(model, loss_f,x,x_mask,x_out,y_out,y_out_mask)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss,label,prediction


# In[191]:

model = S2S(word_vectors,word_vectors_fr,i2w)

from tqdm import tqdm 

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
epochs = 30
for epoch in range(1, epochs + 1):
    print(epoch,flush=True)
    for x,x_mask,x_out,y_out,y_out_mask in tqdm(train_data):
        y_out_onehot = tf.one_hot(y_out,depth = word_vectors_fr.shape[0])
        compute_apply_gradients(model,x,x_mask,x_out,y_out_onehot,y_out_mask,optimizer)
    print(" ".join(ang[25852]))
    print(" ".join(model.generate(word_map_fr["<S>"],ang_tok[25852,:],ang_mask[25852,:])))
    print(" ".join(ang[34368]))
    print(" ".join(model.generate(word_map_fr["<S>"],ang_tok[34368,:],ang_mask[34368,:])))


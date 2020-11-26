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

ang = [['<S>'] + [token.string.strip() for token in tokenizer(text.lower())] + ['</S>'] for text in ang][:30000]

fra = [['<S>'] + [token.string.strip() for token in tokenizer_f(text.lower())] + ['</S>'] for text in fra][:30000]

print(ang[25852])
print(fra[25852])


# In[155]:


from gensim.models import Word2Vec
import numpy as np

EMBEDDING_SIZE = 20
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

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

class S2S(tf.keras.Model):
    def __init__(self,Win,Wout,i2w,pad_in,pad_out):
      
        super(S2S, self).__init__() 
        
        self.nv_in = Win.shape[0]
        self.r = Win.shape[1]
        self.nv_out = Wout.shape[0]
        
        self.i2w = i2w
        
        self.Win = layers.Embedding(self.nv_in,self.r)
        self.Win.build((None, ))
        self.Win.set_weights([Win])
        self.Win.trainable = True

        self.Wout = layers.Embedding(self.nv_out,self.r)
        self.Wout.build((None, ))
        self.Wout.set_weights([Wout])
        self.Wout.trainable = True
        
        self.encoder = layers.GRU(self.r)
        
        self.decoder = layers.GRU(self.r, return_sequences=True, return_state=True)
        
        self.mapper = layers.Dense(self.nv_out,activation = "softmax")

        self.pos_encoding_in = positional_encoding(pad_in, self.r)
        self.pos_encoding_out = positional_encoding(pad_out, self.r)


    @tf.function
    def call(self,x,x_mask,x_out):
        x = self.Win(x)
        x_mask = tf.cast(x_mask,dtype=bool)
    
        x = self.encoder(x,mask=x_mask)   
        
        x_out = self.Wout(x_out)
        out,_ = self.decoder(x_out, initial_state=x)
        
        probs = self.mapper(out)

        return probs

    '''
    def generate(self,start_emb,x,x_mask):
        aout = []
        
        x = tf.expand_dims(x,axis=0)
        x_mask = tf.expand_dims(x_mask,axis=0)

        x_mask = tf.cast(x_mask,dtype=bool)
        
        x = self.Win(x)

        x = self.encoder(x,mask=x_mask) 
        
        x_out = tf.expand_dims(tf.expand_dims(self.Wout(start_emb),axis = 0),axis = 0)         
        _,out = self.decoder(x_out, initial_state=x) 
        probs = tf.squeeze(self.mapper(out))
        
        x_out = tf.math.argmax(probs)
        val,argval = tf.nn.top_k(probs, k=2, sorted=True, name=None)
        x_out = argval.numpy()[0]
        aout.append(self.i2w[x_out])
        
        for i in range(10):
            x_out = tf.expand_dims(tf.expand_dims(self.Wout(tf.constant(x_out)),axis = 0),axis = 0)     
            _,out = self.decoder(x_out, initial_state=out)  
            
            probs = tf.squeeze(self.mapper(out))
            val,argval = tf.nn.top_k(probs, k=2, sorted=True, name=None)
            x_out = argval.numpy()[0]
            aout.append(self.i2w[x_out])
            
        return aout
        '''


@tf.function
def compute_loss(model,loss_f,x,x_mask,x_out,y_out,y_out_mask):
    
    
    probs = model(x,x_mask,x_out)
    
    y_true= tf.boolean_mask(y_out,y_out_mask)
    y_pred = tf.boolean_mask(probs,y_out_mask) 
    
    return loss_f(y_true,y_pred)


@tf.function
def compute_apply_gradients(model, x,x_mask,x_out,y_out,y_out_mask, optimizer):
    loss_f = tf.keras.losses.CategoricalCrossentropy()

    with tf.GradientTape() as tape:
        
        loss = compute_loss(model, loss_f,x,x_mask,x_out,y_out,y_out_mask)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# In[191]:


model = S2S(word_vectors,word_vectors_fr,i2w)
print(" ".join(model.generate(word_map_fr["<S>"],ang_tok[25852,:],ang_mask[25852,:])))

# In[ ]:


from tqdm import tqdm 

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
epochs = 30
for epoch in range(1, epochs + 1):
    print(epoch,flush=True)
    for x,x_mask,x_out,y_out,y_out_mask in tqdm(train_data):
        y_out_onehot = tf.one_hot(y_out,depth = word_vectors_fr.shape[0])
        compute_apply_gradients(model,x,x_mask,x_out,y_out_onehot,y_out_mask,optimizer)
    print(" ".join(model.generate(word_map_fr["<S>"],ang_tok[25852,:],ang_mask[25852,:])))

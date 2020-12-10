import tensorflow as tf
from tensorflow.keras import layers,Model
import numpy as np


A = tf.random.uniform((3,3))

mask = tf.ones((A.shape[0],1))

print(mask)

paddings = tf.constant([[0, 0], [0,A.shape[0]-1]])

mask = tf.pad(mask, paddings, "CONSTANT")
print(mask)
'''
A = tf.slice(A,[0,1,0],[A.shape[0],A.shape[1]-1,A.shape[2]])
print(A)


print(tf.math.reduce_variance(A, axis=1))


def compute_variance(inpu):
    print(inpu,flush=True)
    tf.math.reduce_variance(inpu.to_tensor(),axis=1)
    return tf.zeros([10,10])

print(A)
mask = np.array([[0,1],[1,0],[1,1]])
print(mask)
mask = tf.convert_to_tensor(mask)
mask = tf.cast(mask,dtype=bool)
print(mask)
#y = tf.broadcast_to(mask, [3, 3])

doc_emb = tf.ragged.boolean_mask(A, mask )

tf.map_fn(compute_variance,doc_emb)

doc_listi = tf.constant(list(range(10)))
print(doc_listi)
    
document = layers.Embedding(10,2,tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),name = 'doc',trainable = True)

print(np.asarray(document(doc_listi)))
'''

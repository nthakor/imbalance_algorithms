import tensorflow as tf
import numpy as np
import pandas as pd

def corrupt(x):
    """Take an input tensor and add uniform masking.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.

    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    """
    return tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))

def _one_hot(label):
  return pd.get_dummies(label)

def _read_split(file,n,read=0):
  """

  Parameters
  ----------

  file: name of the csv file

  n: number of rows to skip
  """
  df=pd.read_csv(file,skiprows=n)
  if(read):

  Xy=df.as_matrix().astype(np.float32)
  y=Xy[:,Xy.shape[1]-1]
  X=np.delete(Xy,Xy.shape[1]-1,axis=1)
   




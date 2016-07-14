import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

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

def _read_split(file,read=0,oneHot=0):
  """

  Parameters
  ----------

  file: name of the csv file

  n: number of rows to skip

  Returns
  -------

  trX, teX, trY, teY
  """
  df=pd.read_csv(file,skiprows=[1])
  df=df.rename(columns={df.columns[len(list(df))-1]:'Class'})
  df0=df[df['Class'] == 0]
  df1=df[df['Class'] == 1]
  imb=float(len(df0))/float(len(df1))
  if(read):
    print 'FEATURES : %d ROWS: %d ' %(( len(list(df))-1 ), len(df))
    print 'Imbalance Ratio: %f' %(imb)

  Xy=df.as_matrix().astype(np.float32)
  y=Xy[:,Xy.shape[1]-1]
  if(oneHot):
    y=_one_hot(y)
  X=np.delete(Xy,Xy.shape[1]-1,axis=1)
  return train_test_split(X,y, test_size=0.33, random_state=42)




def _reverse_one_hot(y):
  n_class=y.shape[1]
  label=np.arange(n_class)
  return np.dot(y,label)

def _class_split(trX,trY,oneHot=0):
  if(oneHot):
    Y=_reverse_one_hot(trY)
  else:
    Y=trY
  X=np.column_stack((trX,Y))
  class_0=X[np.where(X[:,X.shape[1]-1]==0)[0]]
  class_1=X[np.where(X[:,X.shape[1]-1]==1)[0]]
  class_0=np.delete(class_0,-1,axis=1)
  class_1=np.delete(class_1,-1,axis=1)
  return class_0,class_1

def _f_count(Y):
  if (len(Y.shape))==2:
    Y=_reverse_one_hot(Y)
  c = np.bincount(Y.astype(np.int32))
  ii = np.nonzero(c)[0]
  return zip(ii,c[ii])

def process_cm(confusion_mat, i=1, to_print=True):
    # i means which class to choose to do one-vs-the-rest calculation
    # rows are actual obs whereas columns are predictions
    TP = confusion_mat[i,i]  # correctly labeled as i
    FP = confusion_mat[:,i].sum() - TP  # incorrectly labeled as i
    FN = confusion_mat[i,:].sum() - TP  # incorrectly labeled as non-i
    TN = confusion_mat.sum().sum() - TP - FP - FN
    if to_print:
        print('TP: {}'.format(TP))
        print('FP: {}'.format(FP))
        print('FN: {}'.format(FN))
        print('TN: {}'.format(TN))
    return TP, FP, FN, TN


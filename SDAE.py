import numpy as np
import tensorflow as tf

class sdae():
	"""Stacked denoising encoder.

	Parametrs
	----------

	Return
	------



	"""

	def __init__(self,dims,activation,epoch,noise,loss,batch_size,
		lr,optimizer,l1,l2,decay,
		optimizer='gradientdescent',
                 momentum=0.0, l1=0.0, l2=0.0,
                 decay=[0.0, 1.0]):

		self.dims=dims
		self.activation=activation
		self.epoch=epoch
		self.noise=noise
		self.loss=loss
		self.batch_size=batch_size
		self.depth=len(dims)
		self.lr=lr
		self.l1 = l1
        self.l2 = l2
        self.decay = decay
        self.momentum = momentum
		self._init_optimizer(self.optimizer)
		self.weights=[]
		self.biases=[]

	 def _init_optimizer(self, optimizer):
        self.global_step_ = tf.Variable(0, trainable=False)
        if self.decay[0] > 0.0:
            learning_rate = tf.train.exponential_decay(
                learning_rate=self.eta,
                global_step=self.global_step_,
                decay_steps=self.decay[1],
                decay_rate=self.decay[0])

        else:
            learning_rate = self.eta
        if optimizer == 'gradientdescent':
            opt = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
        elif optimizer == 'momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                             momentum=self.momentum)
        elif optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == 'ftrl':
            opt = tf.train.FtrlOptimizer(
                learning_rate=learning_rate,
                l1_regularization_strength=self.l1,
                l2_regularization_strength=self.l2)
        elif optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        else:
            raise AttributeError('optimizer must be "gradientdescent",'
                                 ' "momentum", "adam", "ftrl", or "adagrad"')
        return opt

    def add_noise(self, x):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), round(frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass


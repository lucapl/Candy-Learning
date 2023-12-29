import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, RepeatVector, TimeDistributed, SimpleRNN, Masking
from tensorflow.keras.layers import Reshape, GlobalMaxPool1D, Lambda, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import Sequence, plot_model, pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

def create_encoder(timesteps,features,labels,lr=0.001,device = 'CPU'):

	pus = tf.config.list_logical_devices(device)
	strategy = tf.distribute.MirroredStrategy(pus)
	model = None

	with strategy.scope():
		adam = Adam(learning_rate=lr)
		model = Sequential()

		model.add(Masking(mask_value=-2,input_shape=(timesteps,features)))
		model.add(GRU(32, input_shape=(None, features),return_sequences=True))
		model.add(GRU(8, input_shape=(None, features)))
		model.add(Dense(128,activation="relu"))
		model.add(Dropout(0.4))
		model.add(BatchNormalization(axis=1))
		model.add(Dense(32,activation="relu"))
		model.add(Dropout(0.4))
		model.add(BatchNormalization(axis=1))
		model.add(Dense(labels,activation="sigmoid"))
		
		model.build(input_shape=(None,features))
		model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
	
	return model
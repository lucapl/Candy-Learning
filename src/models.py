from keras.layers import GRU, Dense, Dropout, BatchNormalization, Masking, TimeDistributed
from keras.models import Sequential

from .layers.Bottleneck import Bottleneck


def create_encoder(time_steps: int, features: int, labels: int) -> Sequential:
	"""
	Returns a GRU encoder model.

	:param time_steps: Number of time steps in the input.
	:type time_steps: int
	:param features: Number of features in the input.
	:type features: int
	:param labels: Number of labels in the output.
	:type labels: int

	:returns: A GRU encoder model.
	:rtype: keras.models.Sequential
	"""
	model = Sequential()

	model.add(Masking(mask_value=-2, input_shape=(time_steps, features)))
	model.add(GRU(64, input_shape=(None, features), return_sequences=True))
	model.add(GRU(32, input_shape=(None, features)))
	model.add(Dense(128, activation="relu"))
	model.add(Dropout(0.4))
	model.add(BatchNormalization(axis=1))
	model.add(Dense(32, activation="relu"))
	model.add(Dropout(0.4))
	model.add(BatchNormalization(axis=1))
	model.add(Dense(labels, activation="sigmoid"))
		
	return model


def create_autoencoder(time_steps: int, features: int, labels: int) -> Sequential:
	"""
	Returns a GRU autoencoder model.
	
	:param time_steps: Number of time steps in the input.
	:type time_steps: int
	:param features: Number of features in the input.
	:type features: int
	:param labels: Number of labels in the output.
	:type labels: int

	:returns: A GRU autoencoder model.
	:rtype: keras.models.Sequential
	"""
	model = Sequential()
	
	model.add(Masking(mask_value=-2, input_shape=(time_steps, features)))
	model.add(GRU(64, input_shape=(None, features), return_sequences=True))
	model.add(GRU(32, return_sequences=True))
	model.add(Bottleneck(GRU(labels, activation="sigmoid"), time_steps))
	model.add(GRU(32, return_sequences=True))
	model.add(GRU(64, return_sequences=True))
	model.add(TimeDistributed(Dense(features)))

	return model

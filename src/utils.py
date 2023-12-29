import matplotlib.pyplot as plt

import tensorflow as tf

def visualize(history: tf.keras.callbacks.History):
	"""plots the history of neural networks training"""
	plt.plot(history.history['binary_accuracy'])
	plt.plot(history.history['val_binary_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()

def plot_sensors(xl,yl,first=5):
	''' Plot the graph of the sensors '''
	for i in range(first):
		plt.plot(xl[i])
		plt.legend([f'sensors {i}' for i in range(3)])
		print(yl[i])
		plt.show()

def compare_sensors(x_pred,x_true,cl,mask,first=5):
	''' Compare the output of the autoencoder with the input '''
	for i in range(first):
		plt.plot(x_pred[i][mask[i]])
		plt.legend([f'pred {i}' for i in range(3)])
		plt.plot(x_true[i][mask[i]])
		plt.legend([f'true {i}' for i in range(3)])
		print(cl[i])
		plt.show()
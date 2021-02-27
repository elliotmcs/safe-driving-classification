import tensorflow as tf
from tensorflow import keras

import numpy as np
from sklearn import preprocessing
from sklearn import model_selection as md
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

import os, random
from tensorflow.keras.regularizers import l2
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from numpy import expand_dims
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import argparse

width = 384
height = 216
batch_size = 64
epochs = 8

class PlotLosses(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.i = 0
		self.x = []
		self.accuracy = []
		self.val_accuracy = []
		self.logs = []
	def on_epoch_end(self, epoch, logs={}):
		self.logs.append(logs)
		self.x.append(self.i)
		self.accuracy.append(logs.get('accuracy'))
		self.val_accuracy.append(logs.get('val_accuracy'))
		self.i += 1
	def on_train_end(self, logs={}):
		self.fig = plt.figure()
		plt.plot(self.x, self.accuracy, label="train")
		plt.plot(self.x, self.val_accuracy, label="validation")
		plt.legend()
		plt.show()

def load_images(args):
	images_path = args.images
	classes = os.listdir(images_path)
	if len(classes) < 2:
		print("E: too few classes in images directory, must be 2 (safe, unsafe).")
		return None
	elif len(classes) > 2:
		print("E: too many classes in images directory, must be 2 (safe, unsafe).")
		return None
	image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
	    preprocessing_function=lambda x:x/255.,
	    validation_split=0.2)
	train_set = image_gen.flow_from_directory(images_path, target_size=(height, width), batch_size=batch_size, subset='training', class_mode='binary')
	val_set = image_gen.flow_from_directory(images_path, target_size=(height, width), batch_size=batch_size, subset='validation', class_mode='binary')
	return train_set, val_set

def build_model():
	main_input = Input(shape=(height, width,3),dtype='float32', name='main_input')
	c1= Conv2D(8, kernel_size=(2,2), activation='relu',padding='same')(main_input)
	c1 = MaxPooling2D(pool_size=(2, 2))(c1)

	c1= Conv2D(16, kernel_size=(2,2), activation='relu',padding='same')(c1)
	c1 = MaxPooling2D(pool_size=(2, 2))(c1)

	c1= Conv2D(32, kernel_size=(2,2), activation='relu',padding='same')(c1)
	c1 = MaxPooling2D(pool_size=(2, 2))(c1)

	c1= Conv2D(64, kernel_size=(2,2), activation='relu',padding='same')(c1)
	c1 = MaxPooling2D(pool_size=(2, 2))(c1)

	c1= Conv2D(128, kernel_size=(2,2), activation='relu',padding='same')(c1)
	c1 = MaxPooling2D(pool_size=(2, 2))(c1)

	flat = Flatten(name='flat')(c1)

	x1 = Dense(64, activation='relu')(flat)
	x1 = Dropout(0.35)(x1)

	x1 = Dense(32, activation='relu')(x1)
	x1 = Dropout(0.35)(x1)

	main_output = Dense(1, activation='sigmoid', name='main_output')(x1)

	model = Model(inputs=main_input,outputs=main_output)

	model.compile(loss=['binary_crossentropy'], optimizer='Adam', metrics=['accuracy'])

	model.summary()
	return model



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
			horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')



def train(args):
	checkpoint_path = args.checkpoint
	ckpt_callback = ModelCheckpoint(filepath=checkpoint_path,
		verbose=1,
		save_weights_only=True,
		monitor='val_accuracy',
		mode='max',
		save_best_only=True)

	es = EarlyStopping(monitor='val_accuracy', patience=4)
	plot_losses = PlotLosses()

	train_set, val_set = load_images(args)

	model = build_model()
	try:
		model.load_weights(checkpoint_path)
	except tf.python.framework.errors_impl.NotFoundError:
		pass
	try:
		model.fit(train_set, validation_data=val_set, shuffle=True, epochs=epochs, callbacks=[plot_losses, es, ckpt_callback])
	except KeyboardInterrupt:
		pass
	plot_losses.on_train_end()

	inp_va = np.empty(shape=(1, height, width, 3))
	tar_va = np.empty(shape=(1,))

	for _ in range(5):
		sample = next(val_set)
		inp_va = np.concatenate([inp_va, sample[0]], axis=0)
		tar_va = np.concatenate([tar_va, sample[1]], axis=0)

	Per_va = model.predict(inp_va)
	Per_class = np.zeros(len(tar_va))

	nv = np.where(Per_va>.5)[0]
	Per_class[nv] = 1

	tar_cm = tar_va
	Per_cm = Per_class

	cnf_matrix = confusion_matrix(tar_cm, Per_cm)
	np.set_printoptions(precision=2)

	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=['Safe', 'Not_Safe'], title='Confusion matrix, without normalization')

	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=['Safe', 'Not_Safe'], normalize=True, title='Normailized confusion matrix')

	plt.show()


def main():
	parser = argparse.ArgumentParser(description='Train image classifier')
	parser.add_argument('images', type=str, help='directory of input images')
	parser.add_argument('checkpoint', type=str, help='path to checkpoint file')

	args = parser.parse_args()
	train(args)

if __name__ == '__main__':
	main()
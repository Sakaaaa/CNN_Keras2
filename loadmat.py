import sys
import numpy as np                  # for algebraic operations, matrices
import h5py
import math
import os
import scipy.io as sio              # I/O
from keras.utils.np_utils import to_categorical




def fLoadData(conten):
	# prepared in matlab
	print 'Loading data'
	for sVarname in ['X_train', 'X_test', 'y_train', 'y_test']:
		if sVarname in conten:
			exec (sVarname + '=conten[sVarname]')
		else:
			exec (sVarname + '= None')

	if (not math.isnan(X_train.all())):
		pIdx = np.random.permutation(np.arange(len(X_train)))
		X_train = X_train[pIdx]
		y_train = y_train[pIdx]
	print 'X_train not empty'
	y_train = to_categorical(y_train,11)
	y_test = to_categorical(y_test,11)
	return X_train, y_train, X_test, y_test


def fRemove_entries(entries, the_dict):
	for key in entries:
		if key in the_dict:
			del the_dict[key]


def fLoadMat(sInPath):
	"""Data"""
	if os.path.isfile(sInPath):
		try:
			conten = sio.loadmat(sInPath)
		except:
			f = h5py.File(sInPath, 'r')
			conten = {}
			conten['X_test'] = np.transpose(np.array(f['X_test']), (3, 2, 0, 1))
			conten['X_train'] = np.transpose(np.array(f['X_train']), (3, 2, 0, 1))
			conten['y_train'] = np.transpose(np.array(f['y_train']))
			conten['y_test'] = np.transpose(np.array(f['y_test']))
			conten['patchSize'] = np.transpose(np.array(f['patchSize']).astype(int))
	else:
		sys.exit('Input file is not existing')
	X_train, y_train, X_test, y_test = fLoadData(conten)  # output order needed for hyperas

	fRemove_entries(('X_train', 'X_test', 'y_train', 'y_test'), conten)
	dData = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'model_name': []}
	dOut = dData.copy()
	dOut.update(conten)
	return dOut  # output dictionary (similar to conten, but with reshaped X_train, ...)


def fLoadDataForOptim(sInPath):
	if os.path.isfile(sInPath):
		conten = sio.loadmat(sInPath)
	X_train, y_train, X_test, y_test = fLoadData(conten)  # output order needed for hyperas
	return X_train, y_train, X_test, y_test, conten["patchSize"]

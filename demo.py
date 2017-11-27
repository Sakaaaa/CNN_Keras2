import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import datasets
import scipy.io as sio
import keras
import os.path,sys
from keras.models import Sequential, Model,model_from_json,load_model
from keras.layers import Dense,Conv2D,MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2  # , activity_l2
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Activation, Flatten  # , Layer  Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D as pool2
import argparse
from loadmat import fLoadData,fLoadDataForOptim,fLoadMat
import h5py
import matplotlib.pyplot as plt
import itertools
import decimal
from keras.layers.merge import concatenate
from keras.layers import Input


def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		cm = np.round(cm,decimals=3)
		print("Normalized confusion matrix")
		dim = cm.shape[0]
		print np.identity(dim)
		print np.diag(np.identity(dim)-cm)
		BER = np.sum(np.diag(np.identity(dim) - cm), axis=0) / dim
		print BER

	else:
		print('Confusion matrix, without normalization')
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


def createModel(patchSize):
	seed = 1
	np.random.seed(seed)
	input = Input(shape=(1, patchSize[0, 0], patchSize[0, 1]))
	out1 = Conv2D(filters=32,
	              kernel_size=(3, 3),
	              kernel_initializer='he_normal',
	              weights=None,
	              padding='valid',
	              strides=(1, 1),
	              kernel_regularizer=l2(1e-6),
	              activation='relu')(input)

	out2 = Conv2D(filters=64,
	              kernel_size=(3, 3),
	              kernel_initializer='he_normal',
	              weights=None,
	              padding='valid',
	              strides=(1, 1),
	              kernel_regularizer=l2(1e-6),
	              activation='relu')(out1)
	out2 = pool2(pool_size=(2, 2), data_format='channels_first')(out2)

	out3 = Conv2D(filters=128,  # learning rate: 0.1 -> 76%
	              kernel_size=(3, 3),
	              kernel_initializer='he_normal',
	              weights=None,
	              padding='valid',
	              strides=(1, 1),
	              kernel_regularizer=l2(1e-6),
	              activation='relu')(out2)

	out4 = Conv2D(filters=128,  # learning rate: 0.1 -> 76%
	              kernel_size=(3, 3),
	              kernel_initializer='he_normal',
	              weights=None,
	              padding='valid',
	              strides=(1, 1),
	              kernel_regularizer=l2(1e-6),
	              activation='relu')(out3)
	out4 = pool2(pool_size=(2, 2), data_format='channels_first')(out4)

	out5_1 = Conv2D(filters=32,
	                kernel_size=(1, 1),
	                kernel_initializer='he_normal',
	                weights=None,
	                padding='same',
	                strides=(1, 1),
	                kernel_regularizer=l2(1e-6),
	                activation='relu')(out4)

	out5_2 = Conv2D(filters=32,  # learning rate: 0.1 -> 76%
	                kernel_size=(1, 1),
	                kernel_initializer='he_normal',
	                weights=None,
	                padding='same',
	                strides=(1, 1),
	                kernel_regularizer=l2(1e-6),
	                activation='relu')(out4)
	out5_2 = Conv2D(filters=128,  # learning rate: 0.1 -> 76%
	                kernel_size=(3, 3),
	                kernel_initializer='he_normal',
	                weights=None,
	                padding='same',
	                strides=(1, 1),
	                kernel_regularizer=l2(1e-6),
	                activation='relu')(out5_2)

	out5_3 = Conv2D(filters=32,  # learning rate: 0.1 -> 76%
	                kernel_size=(1, 1),
	                kernel_initializer='he_normal',
	                weights=None,
	                padding='same',
	                strides=(1, 1),
	                kernel_regularizer=l2(1e-6),
	                activation='relu')(out4)
	out5_3 = Conv2D(filters=128,  # learning rate: 0.1 -> 76%
	                kernel_size=(5, 5),
	                kernel_initializer='he_normal',
	                weights=None,
	                padding='same',
	                strides=(1, 1),
	                kernel_regularizer=l2(1e-6),
	                activation='relu')(out5_3)

	out5_4 = pool2(pool_size=(2, 2), data_format='channels_first')(out4)
	out5_4 = Conv2D(filters=128,  # learning rate: 0.1 -> 76%
	                kernel_size=(1, 1),
	                kernel_initializer='he_normal',
	                weights=None,
	                padding='same',
	                strides=(1, 1),
	                kernel_regularizer=l2(1e-6),
	                activation='relu')(out5_4)

	out5 = concatenate(inputs=[out5_1, out5_2, out5_3], axis=1)

	out6 = Conv2D(filters=256,  # learning rate: 0.1 -> 76%
	              kernel_size=(3, 3),
	              kernel_initializer='he_normal',
	              weights=None,
	              padding='valid',
	              strides=(1, 1),
	              kernel_regularizer=l2(1e-6),
	              activation='relu')(out5)

	out7 = Conv2D(filters=256,  # learning rate: 0.1 -> 76%
	              kernel_size=(3, 3),
	              kernel_initializer='he_normal',
	              weights=None,
	              padding='valid',
	              strides=(1, 1),
	              kernel_regularizer=l2(1e-6),
	              activation='relu')(out6)
	out7 = pool2(pool_size=(2, 2), data_format='channels_first')(out7)

	out8 = Flatten()(out7)

	out9 = Dense(units=11,
	             kernel_initializer='normal',
	             kernel_regularizer='l2',
	             activation='softmax')(out8)

	cnn = Model(inputs=input, outputs=out9)
	return cnn

def fTrain(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSize=None, learningRate=None, iEpochs=None):
	# parse inputs
	batchSize = 64 if batchSize is None else batchSize
	learningRate = 0.01 if learningRate is None else learningRate
	iEpochs = 300 if iEpochs is None else iEpochs

	print 'Training CNN'
	print 'with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize)

	# save names
	_, sPath = os.path.splitdrive(sOutPath)
	sPath, sFilename = os.path.split(sPath)
	sFilename, sExt = os.path.splitext(sFilename)
	model_name = sPath + '/' + sFilename + str(patchSize[0, 0]) + str(patchSize[0, 1]) + '_lr_' + str(
		learningRate) + '_bs_' + str(batchSize)
	weight_name = model_name + '_weights.h5'
	model_json = model_name + '_json'
	model_all = model_name + '_model.h5'
	model_mat = model_name + '.mat'

	if (os.path.isfile(model_mat)):  # no training if output file exists
		return

	# create model
	cnn = createModel(patchSize)

	# opti = SGD(lr=learningRate, momentum=1e-8, decay=0.1, nesterov=True);#Adag(lr=0.01, epsilon=1e-06)
	opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]

	cnn.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])

	result = cnn.fit(X_train,
					 y_train,
					 validation_data=[X_test, y_test],
					 epochs=iEpochs,
					 batch_size=batchSize,
					 callbacks=callbacks,
					 verbose=1)

	score_test, acc_test = cnn.evaluate(X_test, y_test, batch_size=batchSize )

	prob_test = cnn.predict(X_test, batchSize, 0)
	y_pred = np.argmax(prob_test, axis=1)
	y_test = np.argmax(y_test, axis=1)
	confusion_mat=confusion_matrix(y_test,y_pred)

	# save model
	json_string = cnn.to_json()
	open(model_json, 'w').write(json_string)
	# wei = cnn.get_weights()
	cnn.save_weights(weight_name, overwrite=True)
	# cnn.save(model_all) # keras > v0.7

	# matlab
	acc = result.history['acc']
	loss = result.history['loss']
	val_acc = result.history['val_acc']
	val_loss = result.history['val_loss']

	print 'Saving results: ' + model_name
	sio.savemat(model_name, {'model_settings': model_json,
							 'model': model_all,
							 'weights': weight_name,
							 'acc': acc,
							 'loss': loss,
							 'val_acc': val_acc,
							 'val_loss': val_loss,
							 'score_test': score_test,
							 'acc_test': acc_test,
							 'prob_test': prob_test,
							 'confusion_mat':confusion_mat})

# input parsing
parser = argparse.ArgumentParser(description='''CNN feature learning''', epilog='''(c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de''')

parser.add_argument('-i', '--inPath', nargs=1, type=str, help='input path to *.mat of stored patches', default='/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/in.mat')

parser.add_argument('-o', '--outPath', nargs=1, type=str, help='output path to the file used for storage (subfiles _model, _weights, ... are automatically generated)', default='/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/out')

parser.add_argument('-r', '--result', nargs=1, type=str, help='output file', default='/home/d1214/no_backup/d1214/NOT_SET/artifact_type/120120/5Conv/')

parser.add_argument('-m', '--model', nargs=1, type=str, choices=['artifact_type','img_type'], help='select CNN model', default='artifacts')

parser.add_argument('-t', '--train', dest='train', action='store_true', help='if set -> training | if not set -> prediction')

parser.add_argument('-p', '--paraOptim', dest='paraOptim', type=str, choices=['grid', 'hyperas', 'none'], help='parameter optimization via grid search, hyper optimization or no optimization', default='none')

args = parser.parse_args()

if os.path.isfile(args.outPath[0]):
	print('Warning! Output file is already existing and will be overwritten')

# load input data
dData = fLoadMat(args.inPath[0])
dData['model_name'] = [args.outPath[0] + '/resgooglenet864180180_lr_0.0001_bs_32bestweights.56-0.07', args.outPath[0] + '/resgooglenet864180180_lr_0.0001_bs_32']
#model_json = dData['model_name'][1] + '_json'
#weight_name = dData['model_name'][0] + '.hdf5'
# save path for keras model
if 'outPath' in dData:
	sOutPath = dData['outPath']
else:
	sOutPath = args.outPath[0]

sOutFile=args.result[0]

if os.path.isfile(sOutFile):
	try:
		conten = sio.loadmat(sOutFile)
	except:
		f = h5py.File(sOutFile, 'r')
		conten = {}
		conten['prob_test'] = np.transpose(np.array(f['prob_test']))
else:
	sys.exit('Output file is not existing')

outData = {'y_test': dData['y_test'], 'prob_test': conten['prob_pre']}
y_pred = np.argmax(outData['prob_test'], axis=1)
y_test = np.argmax(outData['y_test'], axis=1)
confusion_mat = confusion_matrix(y_test, y_pred)
#model_weights=load_model(weight_name)
#model = model_from_json(model_json)
class_names=['Head_t1','Head_t1m','Abd_t1','Abd_t1m','Abd_t2','Abd_t2m','Abd_s','Liver_t1','Liver_t1m','Liver_t2','Liver_t2s']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(confusion_mat, classes=class_names,
					  title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(confusion_mat, classes=class_names, normalize=True,
					  title='confusion matrix')
plt.savefig("resdennet1180180.ps")

x = [0,1,2,3,4,5,6,7,8,9,10]
head_old_y = [0.7535,0.853,0.9076,0.9068,0.9332,0.9221,0.7458,0.8081,0.9502,0.7727,0.9208]
head_old_avg =[0.8613,0.8613,0.8613,0.8613,0.8613,0.8613,0.8613,0.8613,0.8613,0.8613,0.8613]
abd_old_y = [0.7145,0.7803,0.7066,0.7068,0.7289,0.6498,0.624,0.721,0.75,0.606,0.6921]
abd_old_avg = [0.6982,0.6982,0.6982,0.6982,0.6982,0.6982,0.6982,0.6982,0.6982,0.6982,0.6982,]

head_resdense2_4040 = [0.8286,0.8843,0.9105,0.895,0.9604,0.9207,0.8462,0.8495,0.9576,0.8859,0.9272]
head_resdense2_avg = [0.8969,0.8969,0.8969,0.8969,0.8969,0.8969,0.8969,0.8969,0.8969,0.8969,0.8969]
abd_resdense2_4040 = [0.3833,0.7651,0.6656,0.8015,0.8026,0.787,0.7057,0.5945,0.8139,0.5051,0.8521]
abd_resdense2_avg = [0.6979,0.6979,0.6979,0.6979,0.6979,0.6979,0.6979,0.6979,0.6979,0.6979,0.6979]


resdense2_y4040 = [0.5931,0.7428,0.7270,0.7882,0.8026,0.7543,0.7171,0.6886,0.8270,0.5854,0.7880]
resdense2_avgy4040 = [0.7286,0.7286,0.7286,0.7286,0.7286,0.7286,0.7286,0.7286,0.7286,0.7286,0.7286]

res6c2pv1_y4040 = [0.5825,0.7318,0.7256,0.7617,0.7849,0.7604,0.7129,0.682,0.8249,0.5828,0.7783]
res6c2pv1_avgy4040 = [0.7207,0.7207,0.7207,0.7207,0.7207,0.7207,0.7207,0.7207,0.7207,0.7207,0.7207]

resgooglenet864_y180180 = [0.6446,0.9168,0.8768,0.9192,0.9375,0.8619,0.7588,0.8133,0.9561,0.7422,0.9346]
resgooglenet864_avgy180180 =[0.8511,0.8511,0.8511,0.8511,0.8511,0.8511,0.8511,0.8511,0.8511,0.8511,0.8511]

resdensenet1_y180180 = [0.6165,0.9134,0.8663,0.9391,0.9063,0.8607,0.7354,0.8252,0.9457,0.7583,0.9135]
resdensenet1_avgy180180 = [0.8436,0.8436,0.8436,0.8436,0.8436,0.8436,0.8436,0.8436,0.8436,0.8436,0.8436,]

cm_resdense180180 = [1,0.9972,0.9904,0.9540,0.9176,0.9828,0.9502,0.9822,0.9898,0.9917,0.9904]
cm_resdense180180_avg = [0.9769,0.9769,0.9769,0.9769,0.9769,0.9769,0.9769,0.9769,0.9769,0.9769,0.9769]

cm_resgooglenet180180 = [1, 0.9942,0.9730,0.9921,0.9792,0.9883,0.9655,0.9952,0.9899,0.9974,0.9861]
cm_resgooglenet180180_avg = [0.9874,0.9874,0.9874,0.9874,0.9874,0.9874,0.9874,0.9874,0.9874,0.9874,0.9874,]


fig, ax = plt.subplots()
index = np.arange(11)
bar_width = 0.3

opacity = 0.4
rects1 = plt.bar(index, cm_resgooglenet180180, bar_width,alpha=opacity, color='b',label='Inception-Resnet')
rects2 = plt.bar(index + bar_width, cm_resdense180180, bar_width,alpha=opacity,color='r',label='Dense ResNet')
line1 = plt.plot(index,cm_resgooglenet180180_avg,'b:')
line2 = plt.plot(index,cm_resdense180180_avg,'r:')

plt.xlabel('Categories')
plt.ylabel('Validation accuracy')
plt.title('Comparison on each category')
plt.legend
#plt.xticks(index + bar_width/2, ('Head', 'Head_t1m', 'Abd_t1', 'Abd_t1m', 'Abd_t2','Abd_t2m','Abd_t2s','Liver_t1','Liver_t1m','Liver_t2','Liver_t2s'),rotation=45)
plt.xticks(index + bar_width/2, ('0', '1', '2', '3', '4','5','6','7','8','9','10'),)
plt.ylim(0,1)
plt.legend()

#plt.tight_layout()
plt.show()




plt.figure()
plt.plot(x, resgooglenet864_y180180, "g-",label="inceptionresnet")
plt.plot(x, resdensenet1_y180180, "r-",label="denseresnet")
plt.ylim(0,1)
plt.xlim(0,10)
plt.grid(True)
plt.legend
plt.xlabel("patients")
plt.ylabel("balanced accuracy")
plt.title("Cross-validation via patients")
plt.show()
plt.savefig("180180cv.jpg")





outData = {'y_test': dData['y_test'], 'prob_test': conten['prob_pre']}
y_pred = np.argmax(outData['prob_test'], axis=1)
y_test = np.argmax(outData['y_test'], axis=1)
confusion_mat = confusion_matrix(y_test, y_pred)
#model_weights=load_model(weight_name)
#model = model_from_json(model_json)
class_names=['Head','Head_m','Abd_t1','Abd_t1m','Abd_t2','Abd_t2m','Abd_s','Liver_t1','Liver_m','Liver_t2','Liver_s']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(confusion_mat, classes=class_names,
					  title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(confusion_mat, classes=class_names, normalize=True,
					  title='Normalized confusion matrix')

INPUT_FOLDER='/home/d1214/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
lstFilesDCM = []
def load_scan2(path):
	lstFilesDCM = []
	for pat_dirName, pat_subdir in os.walk(path):
		if "dicom_sorted" in pat_subdir:
			for dirName, subdirList, fileList in os.walk(path,pat_dirName):
				for filename in fileList:
					if ".ima" in filename.lower():
						lstFilesDCM.append(os.path.join(pat_dirName,dirName,filename))
	return lstFilesDCM

#first_patient = load_scan2(INPUT_FOLDER)
#dicom.read_file(lstFilesDCM[0])


fTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'],
					32, 0.00005, 54)

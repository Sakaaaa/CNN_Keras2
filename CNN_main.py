"""Import"""

import sys
import numpy as np                  # for algebraic operations, matrices
import h5py
import math
#import scipy as sp                  # numerical things, optimization, integrals
import scipy.io as sio              # I/O
import os.path                      # operating system
#import theano.tensor as T           # define, optimze, evaluate multidim arrays
import keras.utils                     # CNN
#import matplotlib.pyplot as plt     # for plotting
import argparse
import theano

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from loadmat import fLoadData,fLoadDataForOptim,fLoadMat


# input parsing
parser = argparse.ArgumentParser(description='''CNN feature learning''', epilog='''(c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de''')

parser.add_argument('-i', '--inPath', nargs=1, type=str, help='input path to *.mat of stored patches', default='/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/in.mat')

parser.add_argument('-o', '--outPath', nargs=1, type=str, help='output path to the file used for storage (subfiles _model, _weights, ... are automatically generated)', default='/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/out')

parser.add_argument('-m', '--model', nargs=1, type=str, choices=['artifact_type','img_type'], help='select CNN model', default='artifacts')

parser.add_argument('-t', '--train', dest='train', action='store_true', help='if set -> training | if not set -> prediction')

parser.add_argument('-p', '--paraOptim', dest='paraOptim', type=str, choices=['grid', 'hyperas', 'none', 'pretrain','dense'], help='parameter optimization via grid search, hyper optimization or no optimization', default='none')

parser.add_argument('-f', '--file', nargs=1, type=str, help='select a trained CNN model with weights')

args = parser.parse_args()

if os.path.isfile(args.outPath[0]):
    print('Warning! Output file is already existing and will be overwritten')

# load input data
dData = fLoadMat(args.inPath[0])
# save path for keras model
if 'outPath' in dData:
    sOutPath = dData['outPath']
else:
    sOutPath = args.outPath[0]


# dynamic loading of corresponding model
cnnModel = __import__(args.model[0], globals(), locals(), ['createModel', 'fTrain', 'fPredict', 'fFinetune', 'fDensenet'], -1) # dynamic module loading with specified functions and with relative implict importing (level=-1) -> only in Python2
print (theano.__version__)

# train (w/ or w/o optimization) and predicting
if args.train: # training
    if args.paraOptim == 'hyperas': # hyperas parameter optimization
        best_run, best_model = optim.minimize(model=cnnModel.fHyperasTrain,
                                              data=fLoadDataForOptim(args.inPath[0]),
                                              algo=tpe.suggest,
                                              max_evals=5,
                                              trials=Trials())
        X_train, y_train, X_test, y_test, patchSize = fLoadDataForOptim(args.inPath[0])
        score_test, acc_test = best_model.evaluate(X_test, y_test)
        prob_test = best_model.predict(X_test, best_run['batch_size'], 0)

    elif args.paraOptim == 'grid':  # grid search
        cnnModel.fGridTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'], [64], [0.0001], 80)

    elif args.paraOptim == 'pretrain':  #
        cnnModel.fFinetune(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'], [16], [0.0001], 100)
    elif args.paraOptim == 'dense':  #
        cnnModel.fDensenet(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'], [16,64], [0.005], 120)

    else:  # no optimization
        cnnModel.fTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'], 128, 0.01, 200)

else:  # predicting
    # dData['model_name'] = [sOutPath + '/resgooglenet864180180_lr_0.0001_bs_32bestweights.56-0.07', sOutPath + '/resgooglenet864180180_lr_0.0001_bs_32']
    dData['model_name'] = [args.file[0]]
    cnnModel.fPredict(dData['X_test'], dData['y_test'], dData['model_name'], sOutPath, dData['patchSize'], 64)


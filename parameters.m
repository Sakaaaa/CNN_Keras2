% initialize
% scaling range
iScaleRange = [0 1];
% patches
patchSize = [40 40]; % x,y
patchOverlap = 0.5; % 50
% splitting strategy
% 'normal': random percentage splitting
% 'crossvalidation_patient': cross validation on patient (leave one patient out)
% 'crossvalidation_data': cross validation on data
sSplitting = 'crossvalidation_patient';
% number of folds (only for crossvalidation_data mode)
nFolds = 11;
% splitting perrcentage in training and test set
dSplitval = 0.1;
% optimization type in keras: 'grid', 'hyperas', 'none'
sOpti = 'grid';
% optimized parameters
sOptiPara.batchSize = 128;
sOptiPara.lr = [0.0001,0.00005]; % -> hardcoded in *.py -> adapt
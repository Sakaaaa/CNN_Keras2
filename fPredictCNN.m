function fPredictCNN( sParafile, sDICOMPath, iGPU, sPathOutIn, sMode, sWeights)
%% test/predict an image
% input
% sParafile     parameter file
% sDICOMPath    to be predicted image
% sMode        used CNN model architecture
% iGPU          used GPU

% (c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de, 2017

% get current path
sCurrPath = fileparts(mfilename('fullpath'));
if(isempty(sCurrPath))
    sCurrPath = '/net/linse8-sn/home/m123/Matlab/ArtefactClassification';
end
addpath(genpath(sCurrPath));

% load parameter set
[sPathPara,sFilenamePara] = fileparts(sParafile);
if(~isempty(sPathPara)), cd(sPathPara); end;
eval([sFilenamePara,';']);
if(~isempty(sPathPara)), cd(sCurrPath); end;

% define optimal models
imgPred= fReadDICOM(sDICOMPath);
imgPred = scaleImg(imgPred,iScaleRange);
dPatches = fPatch(imgPred, patchSize, patchOverlap);
X_test =  permute(dPatches,[3 4 1 2]);
y_test = ones(size(X_test,1),1);
X_train = zeros(size(X_test));
y_train = zeros(size(y_test));


y_test = fLabel(sDICOMPath,y_test,sMode);

[tmp,filename]=fileparts(sDICOMPath);
[~,patname]=fileparts(fileparts(tmp));
if(~exist(sPathOutIn,'dir'))
   mkdir(sPathOutIn);
end
save([sPathOutIn,filesep,num2str(patchSize(1)),num2str(patchSize(2)),patname,filename,'.mat'], 'X_test', 'sMode', 'y_test', 'X_train', 'y_train','patchSize');

% call python
%fSetGPU( iGPU );
system(sprintf('python2 CNN_main.py -i %s -o %s -m %s -f %s', [sPathOutIn,filesep,num2str(patchSize(1)),num2str(patchSize(2)),patname,filename,'.mat'], sPathOutIn, sMode, sWeights));

end


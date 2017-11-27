function fTrain(sParafile, sMode, lPats, iGPU, sPath)
% inputs
% sParafile     parameter file
% sMode         CNN model for bi-classification or multi-classification
% lPats         logical array for patients used to run cross-validation -> parallization
% iGPU          used GPU

% (c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de, 2017

%% train network
% Path of the database

sCurrPath = fileparts(mfilename('fullpath'));
if(isempty(sCurrPath))
    sCurrPath = '/net/linse8-sn/home/m123/Matlab/ArtefactClassification';
end
addpath(genpath(sCurrPath));
addpath(genpath([sCurrPath,filesep,'io']));
% parse available patients
% sPats = {'01_ab', '04_dc', '05_fg', '06_hr', '07_hs', '08_jw', '09_ma', '10_mar','12_ms', '13_sg', '15_yb'};
sPats = dir(sPath); 
lMask = cell2mat({sPats(:).isdir}); if(any(~lMask)), sPats(~lMask) = []; end
sPats = sPats(3:end-2);%leave two (16_mj,17_ks) as test data

if(nargin < 5 || ~exist('sPath','var'))
    DataPath='/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol';%Path of the database
end

if(nargin < 4 || ~exist('iGPU','var'))
    iGPU = 2;
end

if(nargin < 3 || ~exist('lPats','var'))
    lPats = true(length(sPats));
end

if(nargin < 2 || ~exist('sMode', 'var'))
    % cnn mode for: 'artifact_type' |'motion_head' | 'motion_abd' | 'shim' | 'noise'
    sMode = 'motion_head';
end

if(nargin < 1 || ~exist('sParafile', 'var'))
    sParafile = 'parameters.m';
end

% load parameter set
[sPathPara,sFilenamePara] = fileparts(sParafile);
if(~isempty(sPathPara)), cd(sPathPara); end;
eval([sFilenamePara,';']);
if(~isempty(sPathPara)), cd(sCurrPath); end;

[sData,sPathOut] = fGetModelInfo(sMode,patchSize);

%labeling for data
if strcmp(sMode,'body')
    sDataAll = cat(2,sData.Head,sData.Becken,sData.Liver);
end

if strncmp(sMode,'motion',6)
    if(ischar(sData.Ref) && ischar(sData.Art))
        sData.Ref = {sData.Ref;0};
        sData.Art = {sData.Art;1};
    end
sDataAll = cat(2,sData.Ref,sData.Art);
end

if strcmp(sMode, 'artifact_type')
	sData.Head = {sData.Head;0};
	sData.Head_m = {sData.Head_m;1};
	sData.Becken_t1 = {sData.Becken_t1;2};
    sData.Becken_t1m = {sData.Becken_t1m;3};
    sData.Becken_t2 = {sData.Becken_t2;4};
    sData.Becken_t2m = {sData.Becken_t2m;5};
	sData.Becken_s = {sData.Becken_s;6};
	sData.Liver_t1 = {sData.Liver_t1;7};
	sData.Liver_m = {sData.Liver_m;8};
	sData.Liver_t2 = {sData.Liver_t2;9};
	sData.Liver_s = {sData.Liver_s;10};
    sDataAll = cat(2,sData.Head,sData.Head_m,sData.Becken_t1,sData.Becken_t1m,sData.Becken_t2,sData.Becken_t2m,sData.Becken_s,sData.Liver_t1,sData.Liver_m,sData.Liver_t2,sData.Liver_s);
end


%% data patching
allPatches = [];
allY = [];
iPats = [];
testPatches = [];
testY = [];
testiPats = [];

for iPat = 1:length(sPats) 
    fprintf('Pat %d/%d', iPat, length(sPats));
    if(strcmp(sPats(iPat).name,'03_cb')) % !!!!!!!!!!!!!! >> Data missing
        continue;
    end
    if((~isempty(regexp(sPats(iPat).name,'old', 'once')))&&strcmp(sMode,'artifact_type')) 
%some old data used as test data, TODO:more new data can also be included as test data
        for iJ=[1,2,8,9]
            fprintf('.');
            sDataIn = sDataAll{1,iJ};
            dImg = fReadDICOM([sPath,filesep,sPats(iPat).name,filesep,sDataIn]);
            
            % scaling
            dImg = scaleImg(dImg, iScaleRange);
            dimension = size(dImg);
            
            % patching
            dPatches = fPatch(dImg, patchSize, patchOverlap);
            y = sDataAll{2,iJ} .* ones(size(dPatches,3),1);
            
            testPatches = cat(3,testPatches,dPatches);
            testY = cat(1,testY,y);
            testiPats = cat(1,testiPats,iPat * ones(size(dPatches,3),1));
        end
    else
        for iJ=1:size(sDataAll,2) % collect training data and validation data
            fprintf('.');
            sDataIn = sDataAll{1,iJ};
            dImg = fReadDICOM([sPath,filesep,sPats(iPat).name,filesep,sDataIn]);
            [nX,nY,nZ] = size(dImg);
            
            % scaling
            dImg = scaleImg(dImg, iScaleRange);
            %         dImg = ((dImg - min(dImg(:))) * (range(2)-range(1)))./(max(dImg(:)) - min(dImg(:)));     
            dimension = size(dImg);
            
            % patching
            dPatches = fPatch(dImg, patchSize, patchOverlap);
            
            y = sDataAll{2,iJ} .* ones(size(dPatches,3),1);
            
            allPatches = cat(3,allPatches,dPatches);
            allY = cat(1,allY,y);
            iPats = cat(1,iPats,iPat * ones(size(dPatches,3),1));
        end
    fprintf('\n');
    end
end

dimensions = [dimension(3), 1, dimension(1),dimension(2)];
%% store test data 
X_test=testPatches;
y_test=testY;
X_test = permute(X_test,[3 4 1 2]);
X_train = zeros(size(X_test));
y_train = zeros(size(y_test));
model_name=sMode;
% save for python
if(~exist(sPathOut,'dir'))
    mkdir(sPathOut);
end
sPathMat = [sPathOut,filesep,'testdata_', num2str(patchSize(1)), num2str(patchSize(2)),'.mat'];
if(~exist(sPathMat,'file'))
    save(sPathMat,'X_train','y_train','model_name','X_test','y_test', 'dimensions', 'patchSize', 'patchOverlap', '-v7.3');
end

%% split training and validation set
if(strcmp(sSplitting,'normal'))
    nPatches = size(allPatches,3);
    dVal = floor(dSplitval* nPatches);

    rand_num = randperm(nPatches,dVal);
    X_test = allPatches(:,:,rand_num);
    y_test = allY(rand_num);

    X_train = allPatches;
    X_train(:,:,rand_num)=[]; 
    y_train = allY;
    y_train(rand_num)=[];

    % arrange data for usage in python and keras
    X_train = permute(X_train,[3 4 1 2]);
    X_test = permute(X_test,[3 4 1 2]);
    
    % save for python
    if(~exist(sPathOut,'dir'))
       mkdir(sPathOut);
    end
    sPathMat = [sPathOut,filesep,'normal_', num2str(patchSize(1)), num2str(patchSize(2)),'.mat'];
    if(~exist(sPathMat,'file'))
    save(sPathMat, 'X_train', 'X_test', 'y_train', 'y_test', 'dimensions', 'patchSize', 'patchOverlap', '-v7.3');
    end
    % call python
    %fSetGPU( iGPU );
    system(sprintf('python2 CNN_main.py -i %s -o %s -m %s -t -p %s', sPathMat, [sPathOut,filesep,'out_normal'], sMode, sOpti));
elseif(strcmp(sSplitting,'crossvalidation_data'))
    if(exist([sPathOut,filesep, num2str(patchSize(1)), num2str(patchSize(2)), filesep, 'data',filesep, 'iFolds.mat'], 'file'))
        load([sPathOut,filesep, num2str(patchSize(1)), num2str(patchSize(2)), filesep, 'data',filesep, 'iFolds.mat']);
    else
        iInd = crossvalind('Kfold', size(allPatches,3), nFolds);
        save([sPathOut,filesep, num2str(patchSize(1)), num2str(patchSize(2)), filesep, 'data',filesep, 'iFolds.mat'], 'iInd');
    end
    for iFold = 1:nFolds
        if(~lPats(iFold)), continue; end;
        fprintf('Fold %d\n', iFold);
        lMask = iInd == iFold;
        
        X_test = allPatches(:,:,lMask);
        y_test = allY(lMask);
        
        X_train = allPatches(:,:,~lMask);
        y_train = allY(~lMask);
        
        X_train = permute(X_train,[3 4 1 2]);
        X_test = permute(X_test,[3 4 1 2]);
        
        % save for python
        sPathOutCurr = [sPathOut,filesep, num2str(patchSize(1)), num2str(patchSize(2)), filesep, 'data',filesep, num2str(iFold,'%02d')];
        if(~exist(sPathOutCurr,'dir'))
            mkdir(sPathOutCurr);
        end
        sFiles = dir(sPathOutCurr); if(length(sFiles(3:end)) == prod(structfun(@(x) length(x), sOptiPara))*3 + 1), continue; end % learning rates, batch sizes, 3 output files, 1 input file

        sPathMat = [sPathOutCurr,filesep,'crossVal_data',num2str(iFold,'%02d'),'_', num2str(patchSize(1)), num2str(patchSize(2)),'.mat'];
        save(sPathMat, 'X_train', 'X_test', 'y_train', 'y_test', 'dimensions', 'patchSize', 'patchOverlap', '-v7.3');
        
        % call python
        %fSetGPU( iGPU );
        system(sprintf('python2 CNN_main.py -i %s -o %s -m %s -t -p %s', sPathMat, [sPathOutCurr,filesep,'outcrossVal_data',num2str(iFold,'%02d')], sMode, sOpti));
    end
    
elseif(strcmp(sSplitting,'crossvalidation_patient'))
    for iPat = 1:length(sPats)
        if(~lPats(iPat)), continue; end;
        fprintf('Patient %d\n', iPat);
        lMask = iPats == iPat;
        
        X_test = allPatches(:,:,lMask);
        y_test = allY(lMask);
        
        X_train = allPatches(:,:,~lMask);
        y_train = allY(~lMask);
        
        X_train = permute(X_train,[3 4 1 2]);
        X_test = permute(X_test,[3 4 1 2]);
        
        % save for python
        sPathOutCurr = [sPathOut,filesep, 'CrossValidation', filesep, 'patient', filesep, num2str(iPat,'%02d')];
        if(~exist(sPathOutCurr,'dir'))
            mkdir(sPathOutCurr);
        end
        sFiles = dir(sPathOutCurr); if(length(sFiles(3:end)) == prod(structfun(@(x) length(x), sOptiPara))*3 + 1), continue; end % learning rates, batch sizes, 3 output files, 1 input file

        sPathMat = [sPathOutCurr,filesep,'crossVal',num2str(iPat,'%02d'),'_', num2str(patchSize(1)), num2str(patchSize(2)),'.mat'];
        if(~exist(sPathMat,'file'))
        save(sPathMat, 'X_train', 'X_test', 'y_train', 'y_test', 'dimensions', 'patchSize', 'patchOverlap', '-v7.3');
        end
        % call python
        %fSetGPU( iGPU );
        system(sprintf('python2 CNN_main.py -i %s -o %s -m %s -t -p %s', sPathMat, [sPathOutCurr,filesep,'out_normal',num2str(iPat,'%02d')], sMode, sOpti));
    end
end


end

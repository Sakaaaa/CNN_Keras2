function [ hfig, dImg, dProbOverlay ] = fVisualizeOverlay( sParafile, sMode, sDICOMPath, dProbPatch, iClass, sPathOutIn, dAlpha )
% visualize overlay

if(nargin < 8)
    dAlpha = 0.4;
end

if(ispc)
    sPath = 'W:\ImageSimilarity\Databases\MRPhysics\newProtocol';    
else
    sPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol';
end

currpath = fileparts(mfilename('fullpath'));
addpath(genpath([currpath,filesep,'utils',filesep,'export_fig']));
addpath(genpath([currpath,filesep,'utils',filesep,'imoverlay']));

sPats = dir(sPath); 
lMask = cell2mat({sPats(:).isdir}); if(any(~lMask)), sPats(~lMask) = []; end
sPats = sPats(3:end);

% load parameter set
[sPathPara,sFilenamePara] = fileparts(sParafile);
if(~isempty(sPathPara)), cd(sPathPara); end;
eval([sFilenamePara,';']);
if(~isempty(sPathPara)), cd(sCurrPath); end;

if(exist('sPathOutIn','var'))
    sPathOut = sPathOutIn;
end

%% overlay
% [sData,sPathOut] = fGetModelInfo(sMode, patchSize);
% switch(sType)
%     case 'Head'
%         sDataIn = sData.Head;
%     case 'Head_m'
%         sDataIn = sData.Head_m;
%     case 'Becken_t1'
%         sDataIn = sData.Becken_t1;
%     case 'Becken_t1m'
%         sDataIn = sData.Becken_t1m;
%     case 'Becken_t2'
%         sDataIn = sData.Becken_t2;
%     case 'Becken_t2m'
%         sDataIn = sData.Becken_t2m;
%     case 'Becken_s'
%         sDataIn = sData.Becken_s;
%     case 'Liver_t1'
%         sDataIn = sData.Liver_t1;
%     case 'Liver_m'
%         sDataIn = sData.Liver_m;
%     case 'Liver_t2'
%         sDataIn = sData.Liver_t2;
%     case 'Liver_s'
%         sDataIn = sData.Liver_s;
%end
fprintf('Loading: %s\n', sDICOMPath);
dImg = fReadDICOM(sDICOMPath);
iDimImg = size(dImg);

% scaling
dImg = scaleImg(dImg, iScaleRange);
%         dImg = ((dImg - min(dImg(:))) * (range(2)-range(1)))./(max(dImg(:)) - min(dImg(:)));

% patching
fprintf('Unpatching...\n');
[dPatchImg,iPatchSizeImg] = fPatch(dImg, patchSize, patchOverlap);

dProbOverlay = fUnpatch( dProbPatch, patchSize, patchOverlap, iPatchSizeImg, iDimImg, iClass);

hfig = [];
%hfig = fPatchOverlay(dImg,dProbOverlay, [0 0.5; 0 1], dAlpha, [sPathOut,filesep,'Overlay',filesep,num2str(iPat)]);
hfig = fPatchOverlay(dImg,dProbOverlay, [1 12; 0 1], dAlpha);


end


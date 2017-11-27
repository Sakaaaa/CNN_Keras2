%% train network
iGPU = 0;
DataPath='/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol';%Path of the database
lPats = true(1,15); lPats(3)=false; %patient 3 is empty!
sParafile = 'parameters.m'; sMode = 'artifact_type';
if strcmp(sMode,'artifact_type') %multiclassification does not use the old data (patient 2,11,14)
    lPats(2)=false;  lPats(11)=false; lPats(14) = false;
end
fTrain(sParafile, sMode, lPats, iGPU, DataPath);

%% predict
sParafile = 'parameters.m'; sMode = 'artifact_type'; sPathout='NOT_SET/artifact_type/4040/Pred';
sDICOMPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/05_fg/dicom_sorted/t1_tse_tra_Kopf_Motion_0003';
%sDICOMPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/05_fg/dicom_sorted/t1_tse_tra_Kopf_0002';
%sDICOMPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/05_fg/dicom_sorted/t1_tse_tra_fs_Becken_Motion_0010';
%sDICOMPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/05_fg/dicom_sorted/t1_tse_tra_fs_Becken_0008';
%sDICOMPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/05_fg/dicom_sorted/t2_tse_tra_fs_Becken_0009';
%sDICOMPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/05_fg/dicom_sorted/t2_tse_tra_fs_Becken_Motion_0011';
%sDICOMPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/05_fg/dicom_sorted/t2_tse_tra_fs_Becken_Shim_xz_0012';
%sDICOMPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/05_fg/dicom_sorted/t1_tse_tra_fs_mbh_Leber_0004';
%sDICOMPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/05_fg/dicom_sorted/t2_tse_tra_fs_navi_Leber_0006';
%sDICOMPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/05_fg/dicom_sorted/t1_tse_tra_fs_mbh_Leber_Motion_0005';
%sDICOMPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/05_fg/dicom_sorted/t2_tse_tra_fs_navi_Leber_Shim_xz_0007';

%sWeights = '/home/s1241/no_backup/s1241/NOT_SET/artifact_type/180180/CrossValidation/patient/05/resgooglenet86405180180_lr_0.0001_bs_16bestweights.hdf5';
sWeights = '/home/s1241/no_backup/s1241/NOT_SET/artifact_type/4040/CrossValidation/patient/05/resdensenet2054040_lr_0.0001_bs_64bestweights.hdf5';
fPredictCNN( sParafile, sDICOMPath, iGPU, sPathout, sMode, sWeights);

%% visualize overlay
%sType = 'Head_m'; % 'Head'|'Head_m'|'Becken_t1'|'Becken_t1m'|'Becken_t2'|'Becken_t2m'|'Becken_s'|'Liver_t1'|'Liver_m'|'Liver_t2'|'Liver_s'
%sPath = '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Headcross/4040';
load([sPathout,filesep,'Pred_result.mat']);
iClass=11;
dProbPatch = prob_pre;
[ hfig, dImg, dProbOverlay ] = fVisualizeOverlay( sParafile, sMode, sDICOMPath, dProbPatch, iClass, sPathout);


%% visualize significant points
%iPats = [2,6];
%sParafile = 'parameters.m'; sMode = 'motion_head';
%[dDeepVis, dSubsets] = fVisualizePoint( iPats, sMode, sParafile );

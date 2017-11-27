function [sData,sPathOut,sOptiModel] = fGetModelInfo(sMode,patchSize)
% get necessary image database information and optimal pre-trained model
% input
% sMode       classification_mode

% (c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de, 2017

% get current path
sCurrPath = fileparts(mfilename('fullpath'));

switch sMode
    case 'artifact_type'
        sData.Head = ['dicom_sorted',filesep,'t1_tse_tra_Kopf_0002'];
        sData.Head_m = ['dicom_sorted',filesep,'t1_tse_tra_Kopf_Motion_0003'];
        sData.Becken_t1 = ['dicom_sorted',filesep,'t1_tse_tra_fs_Becken_0008'];
        sData.Becken_t1m = ['dicom_sorted',filesep,'t1_tse_tra_fs_Becken_Motion_0010'];
        sData.Becken_t2 = ['dicom_sorted',filesep,'t2_tse_tra_fs_Becken_0009'];
        sData.Becken_t2m = ['dicom_sorted',filesep,'t2_tse_tra_fs_Becken_Motion_0011'];
        sData.Becken_s = ['dicom_sorted',filesep,'t2_tse_tra_fs_Becken_Shim_xz_0012'];
        sData.Liver_t1 = ['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_0004'];
        sData.Liver_m = ['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_Motion_0005'];
        sData.Liver_t2 = ['dicom_sorted',filesep,'t2_tse_tra_fs_navi_Leber_0006'];
        sData.Liver_s = ['dicom_sorted',filesep,'t2_tse_tra_fs_navi_Leber_Shim_xz_0007'];
        if(ispc)
            sPathOut = ['F:\Courses\INFOTECH\MasterThesis\LIUKE\NOTSET\artifact_type',filesep,num2str(patchSize(1)), num2str(patchSize(2))];
        else
            sPathOut = ['NOT_SET/artifact_type',filesep,num2str(patchSize(1)), num2str(patchSize(2))];
        end
        sOptiModel = {};
        
	case 'image_type'
        sData.Ref = {['dicom_sorted',filesep,'t1_tse_tra_Kopf_0002'],['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_0004']; ...
					 0												,0};
        sData.Art = {['dicom_sorted',filesep,'t1_tse_tra_Kopf_Motion_0003'],['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_Motion_0005']; ...
					 1													  , 1};
        if(ispc)
            sPathOut = ['F:\Courses\INFOTECH\MasterThesis\LIUKE\NOTSET\image_type',filesep,num2str(patchSize(1)), num2str(patchSize(2))];
        else
            sPathOut = ['NOT_SET/artifact_type',filesep,num2str(patchSize(1)), num2str(patchSize(2))];
        end
        sOptiModel = {};
        
    case 'body'
        sData.Head = {['dicom_sorted',filesep,'t1_tse_tra_Kopf_0002'],['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_0004']; ...
                        0												,0};
        sData.Becken = {['dicom_sorted',filesep,'t1_tse_tra_fs_Becken_0008'],['dicom_sorted',filesep,'t1_tse_tra_fs_Becken_Motion_0010'],['dicom_sorted',filesep,'t2_tse_tra_fs_Becken_0009'],['dicom_sorted',filesep,'t2_tse_tra_fs_Becken_Motion_0011'],['dicom_sorted',filesep,'t2_tse_tra_fs_Becken_Shim_xz_0012']; ...
                        1													,1                                                          ,1                                                   ,1                                                          ,1};
        sData.Liver = {['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_Motion_0005'],['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_0004'],['dicom_sorted',filesep,'t2_tse_tra_fs_navi_Leber_0006'],['dicom_sorted',filesep,'t2_tse_tra_fs_navi_Leber_Shim_xz_0007']; ...
                        2                                                            ,2                                                     ,2                                                       ,2};
        if(ispc)
            sPathOut = ['F:\Courses\INFOTECH\MasterThesis\LIUKE\NOTSET\body',filesep,num2str(patchSize(1)), num2str(patchSize(2))];
        else
            sPathOut = ['NOT_SET/body',filesep,num2str(patchSize(1)), num2str(patchSize(2))];
        end
        sOptiModel = {};
        
    case 'motion_head'
        sData.Ref = ['dicom_sorted',filesep,'t1_tse_tra_Kopf_0002'];
        sData.Art = ['dicom_sorted',filesep,'t1_tse_tra_Kopf_Motion_0003'];
        if(ispc)
            sPathOut = 'W:\ImageSimilarity\Databases\MRPhysics\CNN\Headcross';
        else
            sPathOut = ['NOT_SET/motion_head',filesep,num2str(patchSize(1)), num2str(patchSize(2))];
        end
        sOptiModel = {[sCurrPath,filesep,'bestModels',filesep,'head_3030_lr_0.0001_bs_64']; ...
            [sCurrPath,filesep,'bestModels',filesep,'head_4040_lr_0.0001_bs_64']; ...
            [sCurrPath,filesep,'bestModels',filesep,'head_6060_lr_0.0001_bs_64'];};
        
    case 'motion_abd'
        sData.Ref = ['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_0004'];
        sData.Art = ['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_Motion_0005'];
        if(ispc)
            sPathOut = 'W:\ImageSimilarity\Databases\MRPhysics\CNN\Abdcross';
        else
            sPathOut = ['NOT_SET/motion_abd',filesep,num2str(patchSize(1)), num2str(patchSize(2))];
        end
        sOptiModel = {[sCurrPath,filesep,'bestModels',filesep,'abdomen_3030_lr_0.0001_bs_64']; ...
                      [sCurrPath,filesep,'bestModels',filesep,'abdomen_4040_lr_0.0001_bs_64']; ...
                      [sCurrPath,filesep,'bestModels',filesep,'abdomen_6060_lr_0.0001_bs_64'];};
					  
	case 'motion_all'
        sData.Ref = {['dicom_sorted',filesep,'t1_tse_tra_Kopf_0002'],['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_0004']; ...
					 0												,0};
        sData.Art = {['dicom_sorted',filesep,'t1_tse_tra_Kopf_Motion_0003'],['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_Motion_0005']; ...
					 1													  , 1};
        if(ispc)
            sPathOut = 'W:\ImageSimilarity\Databases\MRPhysics\CNN\Allcross';
        else
            sPathOut = ['NOT_SET/motion_all',filesep,num2str(patchSize(1)), num2str(patchSize(2))];
        end
        sOptiModel = {};
        
    case 'shim'
       sData.Ref = ['dicom_sorted',filesep,'t2_tse_tra_fs_Becken_0009'];
       sData.Art = ['dicom_sorted',filesep,'t2_tse_tra_fs_Becken_Shim_xz_0012'];
       if(ispc)
           sPathOut = 'W:\ImageSimilarity\Databases\MRPhysics\CNN\Shimcross';
       else
            sPathOut = ['NOT_SET/shim',filesep,num2str(patchSize(1)), num2str(patchSize(2))];
       end
       sOptiModel = [sCurrPath,filesep,'bestModels',filesep,'MISSING'];
        
    case 'noise'
        
end

end
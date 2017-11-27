function y=fLabel(sDICOMPath,y,sModel)



switch sModel
    case 'artifact_type'
        if(~isempty(regexp(sDICOMPath,'Kopf_Motion', 'once')))
            y = y.*1;
        elseif(~isempty(regexp(sDICOMPath,'Kopf', 'once')))
            y = y.*0;
        end
        
        if(~isempty(regexp(sDICOMPath,'Becken_Shim', 'once')))
            y = y.*6;
        elseif(~isempty(regexp(sDICOMPath,'Becken_Motion_0010', 'once')))
            y = y.*3;
        elseif(~isempty(regexp(sDICOMPath,'Becken_Motion_0011', 'once')))
            y = y.*5;
        elseif(~isempty(regexp(sDICOMPath,'Becken_0008', 'once')))
            y = y.*2;
        elseif(~isempty(regexp(sDICOMPath,'Becken_0009', 'once')))
            y = y.*4;
        end
        
        if(~isempty(regexp(sDICOMPath,'Leber_Shim', 'once')))
            y = y.*10;
        elseif(~isempty(regexp(sDICOMPath,'Leber_Motion_0005', 'once')))
            y = y.*8;
        elseif(~isempty(regexp(sDICOMPath,'Leber_0004', 'once')))
            y = y.*7;   
        elseif(~isempty(regexp(sDICOMPath,'Leber_0006', 'once')))
            y = y.*9;
        end
        
	case 'image_type'
        
    case 'body'
        
    case 'motion_head'
        
    case 'motion_abd'

	case 'motion_all'
        
    case 'shim'
        
    case 'noise'
        
end

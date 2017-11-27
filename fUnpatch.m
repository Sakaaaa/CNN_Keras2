function dImg = fUnpatch( dPatch, patchSize, patchOverlap, iZpaddedSize, iActualSize, iClass )
% unpatch to image
% dPatch:   either nPatches x iPatchSize(1) x iPatchSize(2)
%           or     nPatches x iClass

% (c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de, 2017

if(nargin < 6), iClass = 1; end
if(nargin < 5), iActualSize = iZpaddedSize; end

% check input
if(ndims(dPatch) > 3)
    error('Invalid Patch size input');
end
if(size(dPatch,1) == patchSize(1) && size(dPatch,2) == patchSize(2)) % patches in first two dimensions
    dPatch = permute(dPatch,[3 1 2]);    
elseif(size(dPatch,1) == patchSize(2) && size(dPatch,2) == patchSize(1)) % patches in first two dimensions
    dPatch = permute(dPatch,[3 2 1]);
end

% init for resulting image
rows_recon = iZpaddedSize(1); cols_recon = iZpaddedSize(2); layers_recon = iZpaddedSize(3); 
dImg=zeros(rows_recon,cols_recon,layers_recon);
dImgLayer=zeros(rows_recon,cols_recon);

% dPatch_rou = round(dPatch);
if(size(dPatch,2) == patchSize(1) && size(dPatch,3) == patchSize(2)) % from image patches
    lMode = true;
else % from estimated class labels
    lMode = false;
    [~,dMaxProbIndex] = max(dPatch,[],2);
    ambiguous_label=iClass+1;
end
    
count_row=1;
count_col=1;
count_layer=1;
lFilled=false(rows_recon,cols_recon); % overlay just possible inside slice
for iIdx=1:size(dPatch,1)
    if(lMode)
        tmp = squeeze(dPatch(iIdx,:,:));
    else
        tmp = ones(patchSize(1),patchSize(2))*dMaxProbIndex(iIdx);
    end
    lMask = false(rows_recon,cols_recon); lMask(max([1,count_row]):min([count_row+patchSize(1)-1,rows_recon]),max([1,count_col]):min([count_col+patchSize(2)-1,cols_recon])) = true;
    lMean = lMask & lFilled;    %overlapping area between patches
    iX = max([1,count_row]):min([count_row+patchSize(1)-1,rows_recon]);
    iY = max([1,count_col]):min([count_col+patchSize(2)-1,cols_recon]);
    dImgLayer(iX,iY) = dImgLayer(iX,iY) + tmp(1:length(iX),1:length(iY));
    dImgLayer(lMean) = 0.5.*dImgLayer(lMean);
    if(~lMode && any(any(lMean))) % if in estimated label mode and has overlapping area
        labels=unique(dImgLayer(lMask));
        num_labels=size(labels(labels~=(ambiguous_label+unique(tmp))/2),1);
        if(num_labels>=2)    %if overlapping area between patches has two different labels
            dImgLayer(lMean) = ambiguous_label.*dImgLayer(lMean)./dImgLayer(lMean); %label that overlappingarea 12
        else
            dImgLayer(lMask) = tmp;
        end
    end
    lFilled = lFilled | lMask;
    if(all(lFilled))
        table=tabulate(dImgLayer(dImgLayer(lFilled)~=ambiguous_label));
        [~,idx]=max(table(:,2));
        most_label=table(idx);
        %dImgLayer(find(dImgLayer==ambiguous_label))=most_label;
    end
%    if(all(lFilled))
%        border=true(size(dImgLayer));
%        border(1+patchSize(2)*patchOverlap:end-patchSize(2)*patchOverlap,1+patchSize(1)*patchOverlap:end-patchSize(1)*patchOverlap)=0;
%        corner=false(size(dImgLayer));
%        corner(1:patchSize(2)*patchOverlap,1:patchSize(1)*patchOverlap)=1;
%        corner(end-patchSize(2)*patchOverlap+1:end,1:patchSize(1)*patchOverlap)=1;
%        corner(1:patchSize(2)*patchOverlap,end-patchSize(1)*patchOverlap+1:end)=1;
%        corner(end-patchSize(2)*patchOverlap+1:end,end-patchSize(1)*patchOverlap+1:end)=1;
%        unchanged=dImgLayer;
%        unchanged_corner=dImgLayer(corner);
%        dImgLayer(border)=unchanged(border)./2;
%        dImgLayer(~border)=unchanged(~border)./4;
%        dImgLayer(find(dImgLayer~=groundtruth))=ambiguous_label;
%        dImgLayer(corner)=unchanged_corner;
%    end
    count_col = count_col + patchOverlap .* patchSize(2);
    if(count_col + patchSize(2)-1 > cols_recon)
        count_col = 1;
        count_row = count_row + patchOverlap .* patchSize(1);
    end
    if(count_row + patchSize(1)-1 > rows_recon) % next layer
        dImg(:,:,count_layer) = dImgLayer;
        count_col = 1;
        count_row = 1;
        count_layer = count_layer + 1;
        lFilled=false(rows_recon,cols_recon);
        dImgLayer=zeros(rows_recon,cols_recon);
    end
end


if(any(iZpaddedSize ~= iActualSize))
    dImg = crop(dImg, iActualSize);
end

end


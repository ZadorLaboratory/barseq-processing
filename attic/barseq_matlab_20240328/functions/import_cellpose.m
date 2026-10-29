
function import_cellpose(dilation_radius)
if ~exist('dilation_radius','var')
    dilation_radius = 5; %how many pixels to expand on each cell
end
folders=get_folders();
fprintf('Importing Cellpose segmentation results ...\n')

imcell={};%dilated
imcell_original={};%undilated
cellpos={};
celllist={};
parfor i=1:length(folders)
    cd(folders{i});
    cd aligned
    S=load('cellmask.mat');
    maski=S.maski;
    maski_dilated=maski;
    if dilation_radius>0
        for n=1:dilation_radius
            maski1=imdilate(maski_dilated,ones(3));
            maski_dilated(maski_dilated==0)=maski1(maski_dilated==0);
        end
    end
    %imcell{i}=maski;
    imcell{i}=maski_dilated;
    imcell_original{i}=maski;
    celllist{i}=unique(imcell{i});
    celllist{i}(celllist{i}==0)=[];
    cellpos{i}=zeros(length(celllist{i}),2);
    for n=1:length(celllist{i})
        [y,x]=ind2sub(size(imcell{i}),find(imcell{i}==celllist{i}(n)));
        cellpos{i}(n,:)=[median(x),median(y)];
    end
    cd ../..
end


save('allsegmentation.mat','imcell_original','imcell','cellpos','celllist','-v7.3');
fprintf('Finished saving segmentations to allsegmentation.mat.\n')
end

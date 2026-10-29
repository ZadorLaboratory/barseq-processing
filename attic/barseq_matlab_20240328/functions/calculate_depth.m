function calculate_depth(use_mock)

if ~exist('use_mock','var')
    use_mock=0;
end


[folders,pos]=get_folders();
if use_mock==0
    %% mark cortex contour
    %mark the contours of the cortex manually
    cd 10x
    allfiles10x=dir('*.tif');
    allfiles10x=sort_nat({allfiles10x.name});
    contours=mmsetsurface(allfiles10x,4);
    cd ..
    save('contours.mat','contours');
else
    contours=repmat({[]},numel(folders),1);
end

% calculate depth of rolonies and cells (leftover0
% we don't use depths and angle anymore, but keeping them here for
% compatibility with old data format

pixelsize=6.5/10; %10x on the kinetix
depths={};angle={};

load('lroi10x.mat','lroi10x','lroihyb10x','cellpos10x');



%uniqpos=sort_nat(unique(pos));
for i=1:length(folders)
    %[~,posidx]=ismember(pos{i},uniqpos);
    cd(folders{i});
    cd aligned %change folder to save output .mat file from mmcalculatedepthcells to the right place
    [depths{i},angle{i}]=mmcalculatedepthrols(contours{i},lroi10x{i},pixelsize);
    [depthshyb{i},anglehyb{i}]=mmcalculatedepthrols(contours{i},lroihyb10x{i},pixelsize);
    [celldepths{i},cellangle{i}]=mmcalculatedepthrols(contours{i},cellpos10x{i},pixelsize);
    cd ../..
end
save('depths.mat','depths','angle','depthshyb','anglehyb','celldepths','cellangle','-v7.3');
end
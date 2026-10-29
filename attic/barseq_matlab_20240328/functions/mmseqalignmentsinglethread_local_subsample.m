function [chprofile,tforms,warnmsg]=mmseqalignmentsinglethread_local_subsample(filename,chprofile, bgnradius,shiftmat,subsample_rate)
%fix bleedthrough in all tif images in the current folder and align them.
%This version works for the xlight seq-o-matic. This version requires bleedthrough profile 6/10/2019. This version uses a
%single core. This version preserves non-seq frames and allows the use of
%dic image to pre-align. This version adds the option of shifting channels
%to correct for dichroic thickness. shiftmat is n x 2 for n channels. Each
%row is [Tx Ty].
%this version uses my_xcorr2 on small crops instead of imregtform
%this version cuts small crops based on original file size, so it should
%work on arbitrary image size.

%check if channels need to be shifted
if ~exist('shiftmat','var')
    shiftmat=zeros(4,2);
end

if ~exist('subsample_rate','var')
    subsample_rate=1;
end

lastwarn('');

%Find all files
files=dir(fullfile([filename,'*.tif']));
files=sort_nat({files.name});
L=numel(files);
fixedfilelist=cell(L);
mkdir original;

% Fixes bleedthrough for all files, and store output filename list
for k=1:L
    currentfile=files{k};
    [~,fixedfilelist{k}]=fixbleed(currentfile,chprofile,bgnradius,shiftmat);
    movefile(files{k},['original/',files{k}]);
end

% Registration
copyfile(fixedfilelist{1},['aligned',fixedfilelist{1}]);
tforms{1}=affine2d(eye(3));
if L>=2
    for k=2:L
        currentfile=fixedfilelist{k};
        tforms{k}=alignseq(currentfile,fixedfilelist{1},subsample_rate);
    end
else
    tforms={};
end

save(['tforms_',filename,'.tif'],'tforms');
% Cleanup files
movefile('aligned*.tif','aligned/');
delete('fixed*tif');

warnmsg=lastwarn;
end


% This subfunction fixes bleed through between channels in illumina seq tiff file.
function [chprofile,fixedimage]=fixbleed(filename,chprofile,radius,shiftmat)
%read file and median filter
info=imfinfo(filename);
im=[];im1=[];
for i=1:4
    im(:,:,i)=imread(filename,i);
end
if length(info)>4
    for i=5:length(info)
        im1(:,:,i-4)=imread(filename,i);
    end
end
im0=im;
%shift chanels, this now works for both the original translation-only
%format, and new affine transformation
if iscell(shiftmat)
    for m=1:numel(shiftmat)
        %im(:,:,m)=imtranslate(im(:,:,m),shiftmat(m,:));
        im(:,:,m)=imwarp(im(:,:,m),shiftmat{m},'OutputView',imref2d(size(im(:,:,1))));
    end
else
    for m=1:size(shiftmat,1)
        im(:,:,m)=imtranslate(im(:,:,m),shiftmat(m,:));
        %im(:,:,i)=imwarp(im(:,:,i),shiftmat{i},'OutputView',imref2d(size(im(:,:,1))));
    end
end

%bgn subtraction
%radius=20;
if radius>0
    im=im-imopen(im,strel('ball', radius, radius));
end

% Assign bleed through coefficients and bgn values
im=reshape(uint16(double(reshape(im,[],4))/chprofile),size(im,1),size(im,2),4);%subtract camera baseline and correct for bleeding

fixedimage=strcat('fixed',filename);
outputfile=fixedimage;
imwrite(uint16(im(:,:,1)),outputfile);
for i=2:4
    imwrite(uint16(im(:,:,i)), outputfile, 'WriteMode','append');
end
if length(info)>4
    for i=5:length(info)
        imwrite(uint16(im1(:,:,i-4)),outputfile,'WriteMode','append');
    end
end
end

function tform=alignseq(imagename,templatename, subsample_rate)
%read images and calculate sum of seq channels. 
info=imfinfo(imagename);
im=zeros(info(1).Height,info(1).Width,length(info));
for n=1:length(info)
    %im(:,:,n)=imwarp(imread(imagename,n),tform,'OutputView',Rfixed);
    im(:,:,n)=imread(imagename,n);
   
end
alignedim=im;

imagesum=(double(sum(im(:,:,1:4),3))./double(max(max(sum(im(:,:,1:4),3)))));
template1=imread(templatename,1);
template2=imread(templatename,2);
template3=imread(templatename,3);
template4=imread(templatename,4);
templatesum=(double(template1+template2+template3+template4)./max(max(double(template1+template2+template3+template4))));

%alignment using ECC
% par.transform = 'translation';
% par.levels = 5;
% par.iterations = 100;
% ransacWarp=iat_ecc(imagesum(200:end-200,200:end-200),templatesum(200:end-200,200:end-200),par);
% [M,N]=size(template4);
% for n=1:length(info)
%     [alignedim(:,:,n),~]=iat_inverse_warping(im(:,:,n),ransacWarp,par.transform,1:N, 1:M);
% end
% tform=[];

%align using imregtform
% [optimizer,metric] = imregconfig('multimodal');
% optimizer.InitialRadius = optimizer.InitialRadius/100; %trial 1 conditions
% optimizer.GrowthFactor=1.01; %trial 1 conditions
% 
% optimizer.Epsilon=optimizer.Epsilon/10;
% optimizer.MaximumIterations=optimizer.MaximumIterations*10;

% tform = imregtform(imagesum(200:end-200,200:end-200), templatesum(200:end-200,200:end-200), 'translation', optimizer, metric,'PyramidLevels',5);
% Rfixed=imref2d(size(templatesum));
% for n=1:length(info)
%     alignedim(:,:,n)=imwarp(im(:,:,n),tform,'OutputView',Rfixed);
% end

%align using normxcorr2
resize_factor=2;
block_size=128;
block_num_1=floor(size(templatesum,1)/block_size);
block_num_2=floor(size(templatesum,2)/block_size);

if block_num_1*block_size~=size(templatesum,1)
    templatesum=templatesum(1:block_num_1*block_size,:);
    imagesum=imagesum(1:block_num_1*block_size,:);
end
if block_num_2*block_size~=size(templatesum,2)
    templatesum=templatesum(:,1:block_num_2*block_size);
    imagesum=imagesum(:,1:block_num_2*block_size);
end

im1=uint8(imresize(templatesum,resize_factor)*255);
im2=uint8(imresize(imagesum,resize_factor)*255);
im1split=mat2cell(im1,128*resize_factor*ones(1,block_num_1),128*resize_factor*ones(1,block_num_2));
im2split=mat2cell(im2,128*resize_factor*ones(1,block_num_1),128*resize_factor*ones(1,block_num_2));
% sort little tiles by total intensity, take the top fractions based on
% subsample_rate. Might be thrown off by very bright stuffs?
im1split_sum=cellfun(@(x) sum(x,'all'),im1split);
[~,idx]=sort(im1split_sum(:),'descend');

c=zeros(size(im1split{1},1)*2-1,size(im1split{1},2)*2-1);
for i=1:round(numel(im1split)/subsample_rate)
    if max(im1split{idx(i)},[],'all')>0
        c=c+my_xcorr2(im1split{idx(i)},im2split{idx(i)});
    end
end
%c=sum(c,3);
%
[ypeak,xpeak] = find(c==max(c(:)));
yoffSet = (ypeak-size(im1split{1},1))/resize_factor;
xoffSet = (xpeak-size(im1split{1},2))/resize_factor;
[~,I]=min(abs(xoffSet)+abs(yoffSet));

alignedim=imtranslate(im,[xoffSet(I),yoffSet(I),0],'bilinear');
tform=affine2d([1,0,0;0,1,0;xoffSet(I) yoffSet(I),1]);

%write aligned image to file.
alignedfile=strcat('aligned',imagename);
imwrite(uint16(alignedim(:,:,1)),alignedfile);
for i=2:size(im,3)
    imwrite(uint16(alignedim(:,:,i)),alignedfile, 'WriteMode','append');
end
end


function c=my_xcorr2(a,b)
    c=conv2(a,rot90(b,2));
end





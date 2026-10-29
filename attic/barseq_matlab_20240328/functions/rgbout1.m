function rgbout1(string,strength,m,b,shiftmat,chprofile)
% write 4-channel images into 8-bit rgb images for display. The four
% channels are assigned as Cyan, Yellow, Magenta, and White. m indicates
% median filter size (0 = no median filtering). b indicates ball radius for
% top hat filtering (0 = no filtering).This version adds the option of shifting channels
%to correct for dichroic thickness. shiftmat is 4 x 2 for 4 channels. Each
%row is [Tx Ty].
if ~exist('m','var')
    m=0;
end
if ~exist('b','var')
    b=0;
end
if ~exist('shiftmat','var')
    shiftmat=zeros(4,2);
end
if isempty(shiftmat)
    shiftmat=zeros(4,2);
end
    

if m<=3&&m>0
    m=3;
end

if ~exist('chprofile','var')
    chprofile=[];
end



fileselection=strcat(string,'*.tif');
files=dir(fullfile(fileselection));
files={files.name};
L=size(files);

for k=1:L(2)
    currentfile=files{k};
    imageC=imtranslate(imread(currentfile,1),shiftmat(1,:));
    imageY=imtranslate(imread(currentfile,2),shiftmat(2,:));
    imageM=imtranslate(imread(currentfile,3),shiftmat(3,:));
    imageW=imtranslate(imread(currentfile,4),shiftmat(4,:));
    if m>0
        imageC=medfilt2(imageC,[m m]);
        imageY=medfilt2(imageY,[m m]);
        imageM=medfilt2(imageM,[m m]);
        imageW=medfilt2(imageW,[m m]);
    end
    if b>0
        se=offsetstrel('ball',b,b);
        imageC=imtophat(imageC,se);
        imageY=imtophat(imageY,se);
        imageM=imtophat(imageM,se);
        imageW=imtophat(imageW,se);
    end
    
    if ~isempty(chprofile)
        im=imageC;im(:,:,2)=imageY;im(:,:,3)=imageM;im(:,:,4)=imageW;
        im=reshape(uint16(double(reshape(im,[],4))/chprofile),size(im,1),size(im,2),4);%subtract camera baseline and correct for bleeding
        imageC=im(:,:,1);
        imageY=im(:,:,2);
        imageM=im(:,:,3);
        imageW=im(:,:,4);
    end
%     imageM=imageW;
%     imageW(:)=0;
%     imageC=imageC/2;
    [p,n]=size(imageC);
    imageRGB(1:p,1:n,1:3)=0;
    imageRGB(1:p,1:n,1)=double(imageY+imageM+imageW)*strength;
    imageRGB(1:p,1:n,2)=double(imageY+imageC+imageW)*strength;
    imageRGB(1:p,1:n,3)=double(imageC+imageM+imageW)*strength;
    
    imageRGB=uint8(imageRGB);
    filename=strcat('RGB',currentfile);
    imwrite(imageRGB, filename);
end
end



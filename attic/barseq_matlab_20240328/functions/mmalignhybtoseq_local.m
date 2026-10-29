function [templatesum,alignedim]=mmalignhybtoseq_local(hybname,seqname,ch,chprofile,radius,nuclearch,cycle,chradius,shiftmat)
% align a single channel from hyb files to the sum of four sequencing
% channels from the first seq file. ch indicates the channel in hyb name to
% use for registration. chprofile is bleedthrough profile. radius is the
% radius for background subtraction. cycle is the sequencing cycle to align to. 
%This version also allows registration of the nuclear stain image after probes. 

%check if there is a nuclear channel.
if ~exist('nuclearch','var')
    nuclearch=0;
end

if ~exist('cycle','var')
    cycle=1;
end

if ~exist('chradius','var')
    chradius=30;
end

if ~exist('shiftmat','var')
    shiftmat=zeros(4,2);
end

% check if hybfiles exist. If not, go into original folder and check again.
hybfiles=dir([hybname,'*.tif']);
if ~isempty(hybfiles)
    hybfiles=sort_nat({hybfiles.name});
    o=0;
else
    cd original
    hybfiles=dir([hybname,'*.tif']);
    hybfiles=sort_nat({hybfiles.name});
    cd ..
    o=1;
end

% check if seqfiles exist. If not, go into original folder and check again.
seqfiles=dir(['*',seqname,'*.tif']);
if ~isempty(seqfiles)
    seqfiles=sort_nat({seqfiles.name});
else
    cd original
    seqfiles=dir(['*',seqname,'*.tif']);
    seqfiles=sort_nat({seqfiles.name});
    seqfiles=cellfun(@(x) ['original/',x],seqfiles,'UniformOutput',0);
    cd ..
end

template1=imread(seqfiles{cycle},1);
template2=imread(seqfiles{cycle},2);
template3=imread(seqfiles{cycle},3);
template4=imread(seqfiles{cycle},4);
templatesum=double(template1+template2+template3+template4)./max(max(double(template1+template2+template3+template4)));

%align hyb files ch channel to the first seqfiles.
for i=1:length(hybfiles)
    fprintf(hybfiles{i})
    if o==0
        info=imfinfo(hybfiles{i});
        im=zeros(info(1).Height,info(1).Width,length(info));
        for n=1:length(info)
            %im(:,:,n)=medfilt2(imread(hybfiles{i},n));
            im(:,:,n)=imread(hybfiles{i},n);
        end
    else
        info=imfinfo(['original/',hybfiles{i}]);
        im=zeros(info(1).Height,info(1).Width,length(info));
        for n=1:length(info)
            %im(:,:,n)=medfilt2(imread(['original/',hybfiles{i}],n));
            im(:,:,n)=imread(['original/',hybfiles{i}],n);
        end
    end

    %shift channels
    if iscell(shiftmat)
        for m=1:size(shiftmat,1)
            %im(:,:,m)=imtranslate(im(:,:,m),shiftmat(m,:));
            im(:,:,m)=imwarp(im(:,:,m),shiftmat{m},'OutputView',imref2d(size(im(:,:,1))));
        end
    else
        for m=1:size(shiftmat,1)
            im(:,:,m)=imtranslate(im(:,:,m),shiftmat(m,:));
            %im(:,:,i)=imwarp(im(:,:,i),shiftmat{i},'OutputView',imref2d(size(im(:,:,1))));
        end
    end

    % for m=1:size(shiftmat,1)
    %     im(:,:,m)=imtranslate(im(:,:,m),shiftmat(m,:));
    % end

    %subtract background
    ball=strel('ball', radius, radius);
    im(:,:,1:4)=im(:,:,1:4)-imopen(im(:,:,1:4),ball);
    
    %if there is a nuclear channel, also background subtract the nuclear
    %image
    if nuclearch>0
        im(:,:,nuclearch)=im(:,:,nuclearch)-imopen(im(:,:,nuclearch),ball);
    end
    
    %correct for bleeding
    im(:,:,1:4)=reshape(uint16(double(reshape(im(:,:,1:4),[],4))/chprofile),size(im,1),size(im,2),4);
    %subtract background of the rolony channel using smaller chradius
    im(:,:,ch(i))=im(:,:,ch(i))-imopen(im(:,:,ch(i)),strel('ball',chradius,chradius)); %Aixin changed here
    
    
%     %alignment using ECC
%     par.transform = 'euclidean';
%     par.levels = 4;
%     par.iterations = 100;
%     ransacWarp=iat_ecc(im(50:end-50,50:end-50,ch),templatesum(50:end-50,50:end-50),par);
%     %ransacWarp=iat_ecc(im(:,1300:end,ch),templatesum(:,1300:end),par);
%     
%     [M,N]=size(template4);
%     
%     %transform images
%     for n=1:length(info)
%         [alignedim(:,:,n),~]=iat_inverse_warping(im(:,:,n),ransacWarp,par.transform,1:N, 1:M);
%     end
    
    
    
     %align using imregtform or imregcorr
     [optimizer,metric] = imregconfig('multimodal');
     optimizer.InitialRadius = optimizer.InitialRadius/5;
     optimizer.MaximumIterations=optimizer.MaximumIterations*5;
     tform = imregtform(im(100:end-100,100:end-100,ch(i)), templatesum(100:end-100,100:end-100), 'translation', optimizer, metric);
     % tform=imregcorr(imagesum(100:end-100,100:end-100), templatesum(100:end-100,100:end-100),'rigid','Window',0);
     Rfixed=imref2d(size(templatesum));
     for n=1:length(info)
         alignedim(:,:,n)=imwarp(im(:,:,n),tform,'OutputView',Rfixed);
     end

    %align using xcorr2
%    resize_factor=1;%doesn't need to be pixel-perfect, hence no sub-pixel shifts
%    im1=uint8(imresize(templatesum,resize_factor)*255);
%    im2=uint8(imresize(im(:,:,ch)/max(max(im(:,:,ch))),resize_factor)*255);
%    im1split=mat2cell(im1,128*resize_factor*ones(1,25),128*resize_factor*ones(1,25));
%    im2split=mat2cell(im2,128*resize_factor*ones(1,25),128*resize_factor*ones(1,25));
%    c=zeros(size(im1split{1},1)*2-1,size(im1split{1},2)*2-1);
%    for n=1:numel(im1split)
%        if max(im1split{n},[],'all')>0
%            c=c+my_xcorr2(im1split{n},im2split{n});
%        end
%    end
    %c=sum(c,3);
    %
%    [ypeak,xpeak] = find(c==max(c(:)));
%    yoffSet = (ypeak-size(im1split{1},1))/resize_factor;
%    xoffSet = (xpeak-size(im1split{1},2))/resize_factor;
%    [~,I]=min(abs(xoffSet)+abs(yoffSet));

%    alignedim=imtranslate(im,[xoffSet(I),yoffSet(I),0],'bilinear');
        
    
    
    alignedfile=strcat('aligned',hybfiles{i});
    imwrite(uint16(alignedim(:,:,1)),alignedfile);
    for n=2:size(im,3)
        imwrite(uint16(alignedim(:,:,n)),alignedfile, 'WriteMode','append');
    end
    if o~=1
        movefile(hybfiles{i},['original/',hybfiles{i}]);
    end
end
end


function c=my_xcorr2(a,b)
    c=conv2(a,rot90(b,2));
end
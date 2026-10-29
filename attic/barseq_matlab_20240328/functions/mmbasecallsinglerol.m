function [seq,seqC,score,sig,lroi1]=mmbasecallsinglerol(rolthresh1,gaussrad,relaxed)
%basecalling barcode rolonies


%% parse inputs

if ~exist('rolthresh1','var')
    rolthresh1=[10 10 10 10];
end

if ~exist('gaussrad','var')
    gaussrad=0;
elseif isempty(gaussrad)
    gaussrad=0;
end

if ~exist('relaxed','var')
    relaxed=0;
end

%% read 1st cycle and find rolonies
seqfiles=dir(fullfile('*BC*seq*.tif'));
[seqfiles,~]=sort_nat({seqfiles.name});
 lim=imread(seqfiles{1},1);
 for n=2:4
     lim(:,:,n)=imread(seqfiles{1},n);
 end

if gaussrad~=0
    lim=imgaussfilt(lim,gaussrad);
end


im=lim;

%%
%find rolonies using a prominence threshold.
r={};rsub={};
if relaxed==0
    parfor n=1:4
        a=im(:,:,n);
        CC = bwconncomp(imregionalmax(imreconstruct(max(a-rolthresh1(n),0),a)));
        r{n}=zeros(length(CC.PixelIdxList),1);
        for i=1:length(CC.PixelIdxList)
            [~,I]=max(a(CC.PixelIdxList{i}));
            r{n}(i)=CC.PixelIdxList{i}(I); %linear indexed peak positions
        end
        [y,x]=ind2sub(size(im(:,:,n)),r{n});
        rsub{n}=[x,y];%rsub is in x, y, consistent with lroi1
    end
else
    parfor n=1:4
        a=lim(:,:,n);
        CC = imregionalmax(imreconstruct(max(a-hybthresh1(n),0),a),4)&imregionalmax(a,4); % find peaks, filter by reconstruted peaks, using connectivity=4
        [y,x]=find(CC);
        r{n}=find(CC);
        rsub{n}=[x,y];%rsub is in x, y, consistent with lroi1
    end
end

%%
%combine rolonies
lpeaks=zeros(size(lim,1),size(lim,2));
%lpeaks(sub2ind([size(lim,1),size(lim,2)],currrol1(:,2)+1,currrol1(:,1)+1))=1;
for n=1:4
    lpeaks(r{n})=1;
end


lpeaks=lpeaks&~imdilate(lpeaks,triu(ones(3))-diag([ones(1,2),zeros(1,1)]));
[lidxy,lidxx]=find(lpeaks);
lroi1=[lidxx,lidxy];

%figure out which channel these rolonies are from
lpeaks1=zeros(size(lim,1),size(lim,2));
for n=1:4
    lpeaks1(r{n})=n;
end

lpeaks1=lpeaks1.*(lpeaks>0);
rsub1={};       
for n=1:4
    [y,x]=ind2sub(size(lim(:,:,n)),find(lpeaks1(:)==n));
    rsub1{n}=[x,y];
end




%% readout signals after gaussian convolution
sig=ones(size(lroi1,1),length(seqfiles),4);

for m=1:length(seqfiles) %for each cycle, read images and convolute by 7x7
    for n=1:4
        if gaussrad==0
            lim(:,:,n)=imread(seqfiles{m},n);
        else
            lim(:,:,n)=imgaussfilt(imread(seqfiles{m},n),gaussrad);
        end
    end
    for n=1:size(lroi1,1) %for each rolony, readout signal
        sig(n,m,:)=lim(lroi1(n,2),lroi1(n,1),:);
    end
end


%%
%calculate score
score=max(sig,[],3)./sqrt(sum(sig.^2,3));
score(isnan(score))=0.5;


% basecall
[~,seq]=max(sig,[],3);

seqC=char(seq);
seqC(seq==1)='G';
seqC(seq==2)='T';
seqC(seq==3)='A';
seqC(seq==4)='C';


save('basecalls-bc.mat','seq','seqC','score','sig','lroi1');

end
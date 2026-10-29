function [cellid1,seq1,seqC1,sig1,score1]=mmbasecallcells(filename,segim1,nucprofile,c)
%using segmented images, call barcodes from cells. If a nucprofile is
%provided, then reduce nuclear background. nucprofile should be a 1x4
%vector that gives the relative strengths of nuc background signals in the
%four channels.
%runs within each aligned folder.
if ~exist('nucprofile','var')
    nucprofile=[];
end

if ~exist('c','var')
    c=0.8;
end


cellid1=unique(segim1(:));
cellid1=cellid1(cellid1~=0);
    
if ~isempty(cellid1)
    files=dir(['*',filename,'*.tif']);
    files=sort_nat({files.name});
    info=imfinfo(files{1});
    
    im=zeros(info(1).Height,info(1).Width,4);
    for i=1:length(files)
        for n=1:4
            im(:,:,n,i)=imread(files{i},n);
        end
    end
    im=reshape(im,size(im,1)*size(im,2),4,[]);
    
    segim1=reshape(segim1,[],1);
    
    %read signal
    
    sig1=zeros(size(cellid1,1),size(im,2),size(im,3));
    for i=1:size(sig1,1)
        sig1(i,:,:)=mean(im(segim1==cellid1(i),:,:));
    end
    
    %if nucprofile is provided, 
    if ~isempty(nucprofile)
        [minsig,I]=min(sig1,[],2);
        nucsig=repmat(minsig./reshape(nucprofile(I),[],1,size(minsig,3)),1,4,1).*repmat(nucprofile,size(minsig,1),1,size(minsig,3));
        sig1=max(sig1-nucsig*c,0);
    end
    
    %call sequences
    [score1,seq1]=max(sig1,[],2);
    seq1=reshape(seq1,size(seq1,1),[]);
    seqC1=char(seq1);
    seqC1(seq1==1)='G';
    seqC1(seq1==2)='T';
    seqC1(seq1==3)='A';
    seqC1(seq1==4)='C';
    
    score1=score1./sqrt(sum(sig1.^2,2));
    score1=reshape(score1,size(score1,1),[]);
    score1(isnan(score1))=0.5;
else
    sig1=[];
    seq1=[];
    seqC1=char([]);
    cellid1=[];
    score1=[];
end
end



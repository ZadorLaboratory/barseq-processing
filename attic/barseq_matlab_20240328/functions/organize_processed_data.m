
function fname=organize_processed_data(startingsliceidx,fname)
if ~exist('fname','var')
    fname=['alldata',char(datetime('today','Format','yyyyMMdd')),'.mat'];
end
[folders,pos]=get_folders();
%
% concatenate lroi10x and id22
%this needs to be more efficient and easier to read.
if ~exist('startingsliceidx','var')
    startingsliceidx=1;%what is the slice number for the first slice. Useful for datasets combining multiple sequencing runs
end
startingfovidx=1;
tilesize=[3200 3200];

idall=[];
idall2=[];
lroi10xall=[];
lroi40xall=[];
depthsall=[];
angleall=[];
lroihyb10xall=[];
lroihyb40xall=[];
idhyball=[];
depthshyball=[];
anglehyball=[];
cellidall=[];
cellidhyball=[];
celllistall=[];
cellpos10xall=[];
cellpos40xall=[];
celldepthsall=[];
cellangleall=[];
fov=[];
hybfov=[];
cellfov=[];
uniqpos=sort_nat(unique(pos));


load('basecalls.mat','id2','lroi');
load('lroi10x.mat','lroi10x','lroihyb10x','cellpos10x');
load('cellid.mat','cellid','cellidhyb');
load('depths.mat','depths','angle','depthshyb','anglehyb','celldepths','cellangle');
load('genehyb.mat','lroihyb','idhyb')
load('allsegmentation.mat','celllist','cellpos')
load('codebook.mat','codebook');
load('../codebookhyb.mat','codebookhyb');
sliceidall=[];
sliceidhyball=[];
cellsliceidall=[];
for i=1:numel(folders)
%which slice
    [~,posidx]=ismember(pos{i},uniqpos);
%     sequencing rolonies
    fov=[fov;ones(numel(cellid{i}),1)*i];
    idall2=[idall2;id2{i}];
    lroi10xall=[lroi10xall;lroi10x{i}];
    lroi40xall=[lroi40xall;lroi{i}];
    depthsall=[depthsall;depths{i}];
    angleall=[angleall;angle{i}];
    cellidall=[cellidall;double(cellid{i})+(startingfovidx+i-1)*10000];
    sliceidall=[sliceidall;double(ones(size(lroi{i},1),1))*(posidx+startingsliceidx-1)];
%     sequential rolonies
    idhyball=[idhyball;idhyb{i}];
    hybfov=[hybfov;ones(numel(cellidhyb{i}),1)*i];
    lroihyb10xall=[lroihyb10xall;lroihyb10x{i}];
    lroihyb40xall=[lroihyb40xall;lroihyb{i}];
    depthshyball=[depthshyball;depthshyb{i}];
    anglehyball=[anglehyball;anglehyb{i}];
    cellidhyball=[cellidhyball;double(cellidhyb{i})+(startingfovidx+i-1)*10000];
    sliceidhyball=[sliceidhyball;double(ones(size(lroihyb{i},1),1))*(posidx+startingsliceidx-1)];
%   
%     cells
    celllistall=[celllistall;double(celllist{i})+(startingfovidx+i-1)*10000];
    cellpos10xall=[cellpos10xall;cellpos10x{i}];
    cellpos40xall=[cellpos40xall;cellpos{i}];
    cellfov=[cellfov;ones(numel(celllist{i}),1)*i];
    
    celldepthsall=[celldepthsall;celldepths{i}];
    cellangleall=[cellangleall;cellangle{i}];
    cellsliceidall=[cellsliceidall;double(ones(numel(celllist{i}),1))*(posidx+startingsliceidx-1)];
    
end

% combine two codebooks and filter out fake rolonies
idhyball1=idhyball+size(codebook,1);
%cid=[idall2(:,1);idhyball1];
cid=[idall2;idhyball1];
cfov=[fov;hybfov];
croi=[lroi10xall;lroihyb10xall];
croi40x=[lroi40xall;lroihyb40xall];
cdepths=[depthsall;depthshyball];
cangles=[angleall;anglehyball];
ccodes=[codebook;codebookhyb];
ccellid=[cellidall;cellidhyball];
csliceidall=[sliceidall;sliceidhyball];
ccelldepths=celldepthsall;
ccellangle=cellangleall;
ccellsliceid=cellsliceidall;

ccelllist=celllistall;
ccellpos=cellpos10xall;

% filter out uncalled rolonies
croi(cid==0,:)=[];
cfov(cid==0,:)=[];
croi40x(cid==0,:)=[];
cdepths(cid==0,:)=[];
cangles(cid==0,:)=[];
ccellid(cid==0,:)=[];
csliceidall(cid==0,:)=[];
cid(cid==0,:)=[];
%

%filter out rolonies that were not called or rolonies at the edge 7% of
%images
edge_filter_size=0.07;%fraction to filter around the edge
I=find(~(cid==0|croi40x(:,1)<tilesize(1)*edge_filter_size|croi40x(:,1)>tilesize(1)*(1-edge_filter_size)| ...
    croi40x(:,2)<tilesize(2)*edge_filter_size|croi40x(:,2)>tilesize(2)*(1-edge_filter_size)));
croi=croi(I,:);
cfov=cfov(I,:);
cdepths=cdepths(I,:);
cangles=cangles(I,:);
ccellid=ccellid(I,:);
csliceidall=csliceidall(I,:);
cid=cid(I,:);
croi40x=croi40x(I,:);


% make expression matrix
expmat=sparse(size(ccelllist,1),size(ccodes,1));
cid1=cid(ismember(ccellid,ccelllist));
ccellid1=ccellid(ismember(ccellid,ccelllist));
[~,ccellid1]=ismember(ccellid1,ccelllist);
%
tic
for i=1:size(ccodes,1)

    expmat(:,i)=accumarray(ccellid1,1*(cid1==i),[size(expmat,1),1]);
end
toc




% save data
save processeddata -v7.3

% organize variable names
%fix depths


rolonies=struct;
rolonies.id=cid;
rolonies.pos=croi;
rolonies.pos40x=croi40x;
rolonies.cellid=ccellid;
rolonies.angle=cangles;
rolonies.depth=cdepths;
rolonies.genes=ccodes;
rolonies.slice=csliceidall;
rolonies.fov=cfov;
rolonies.fov_names=folders;

neurons.expmat=expmat;
neurons.id=ccelllist;
neurons.pos=ccellpos;
neurons.depth=ccelldepths;
neurons.angle=ccellangle;
neurons.slice=ccellsliceid;
neurons.pos40x=cellpos40xall;
neurons.genes=ccodes;
neurons.fov=cellfov;
neurons.fov_names=folders;

save(fname,'rolonies','neurons','-v7.3');
end

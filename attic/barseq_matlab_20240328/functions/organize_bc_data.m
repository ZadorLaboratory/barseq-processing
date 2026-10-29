
function organize_bc_data(count_thresh,err_corr_thresh,data_fname,mismatch_thresh)
[folders,pos]=get_folders();
dom_bc=cell(numel(folders),1);
all_bc=cell(numel(folders),1);
all_bc_count=cell(numel(folders),1);
dom_bc_count=cell(numel(folders),1);

load('allsegmentation.mat','celllist');
load('bccellid.mat','bccellid');
load('bclroi10x.mat','bclroi10x');
load('bc.mat','bcseq','bcscore','bcint','bcsig','bclroi')


parfor i=1:length(folders)
    
    dom_bc{i}={};
    dom_bc_count{i}=zeros(numel(celllist{i}),1);
    all_bc{i}={};
    all_bc_count{i}={};

    for n=1:numel(celllist{i})

        if sum(bccellid{i}==celllist{i}(n))>count_thresh % if more than threshold number of bc, then count bc molecules
            
            bc=bcseq{i}(bccellid{i}==celllist{i}(n),:); %all barcode sequences in that cell.
            %collapse simialr barcodes.

            [uniqbc,~,ic]=unique(bc,'rows');
            c1=histcounts(ic,[0;unique(ic)]+0.5);
            [sortedc1,I]=sort(c1,'descend');
            uniqbc=uniqbc(I,:); %sorted by abundunce within the cell

            bclist=uniqbc(1,:);bccount=sortedc1(1);
            
            if size(uniqbc,1)>0
                for m=2:size(uniqbc,1)
                    [min_d,I]=min(pdist2(bclist,uniqbc(m,:),'hamming')*size(uniqbc,2));
                    if min_d<=err_corr_thresh
                        bccount(I)=bccount(I)+sortedc1(m);
                    else
                        bclist(end+1,:)=uniqbc(m,:);
                        bccount(end+1)=sortedc1(m);
                    end
                end
            end
            
            [~,I]=max(bccount);
            bclist=int8(bclist);

            if bccount(I)>count_thresh
                dom_bc{i}{n}=bclist(I,:);
                dom_bc_count{i}(n)=bccount(I);
                all_bc{i}{n}=bclist;
                all_bc_count{i}{n}=bccount;
            else
                dom_bc{i}{n}=[];
                dom_bc_count{i}(n)=0;
                all_bc{i}{n}=[];
                all_bc_count{i}{n}=0;
            end
        else
                dom_bc{i}{n}=[];
                dom_bc_count{i}(n)=0;
                all_bc{i}{n}=[];
                all_bc_count{i}{n}=0;
        end

    end

end
save('cellbc.mat','dom_bc','all_bc');

% combine barcode information with neurons
load(data_fname,'rolonies');
load('filt_neurons.mat','filt_neurons');

filt_neurons.dom_bc=cell(numel(filt_neurons.id),1);
filt_neurons.dom_bc_count=zeros(numel(filt_neurons.id),1);
filt_neurons.all_bc=cell(numel(filt_neurons.id),1);
filt_neurons.all_bc_count=cell(numel(filt_neurons.id),1);

for i=1:numel(dom_bc)
    [~,I]=ismember(double(celllist{i})+i*10000,filt_neurons.id);
    filt_neurons.dom_bc(I(I>0))=dom_bc{i}(I>0);
    filt_neurons.dom_bc_count(I(I>0))=dom_bc_count{i}(I>0);
    filt_neurons.all_bc(I(I>0))=all_bc{i}(I>0);
    filt_neurons.all_bc_count(I(I>0))=all_bc_count{i}(I>0);
end


% bc
bc=struct;
bc.pos40x=cell2mat(bclroi');
bc.pos=cell2mat(bclroi10x');
bc.slice=[];
bc.seq=cell2mat(bcseq');
bc.qual=cell2mat(bcscore');
bc.int=cell2mat(bcint');
bc.sig=cell2mat(bcsig');
bc.matched_cell=[];
%

startingsliceidx=min(filt_neurons.slice);

uniqpos=sort_nat(unique(pos));
[~,posidx]=ismember(pos,uniqpos);
bcslice={};
bcfov={};
parfor i=1:numel(bclroi)
    bcslice{i}=double(ones(size(bclroi{i},1),1))*(posidx(i)+startingsliceidx-1);
    bcfov{i}=double(ones(size(bclroi{i},1),1))*(i);
end
bc.slice=cell2mat(bcslice');
bc.fov=cell2mat(bcfov');
bc.fov_names=folders;



% match bc to cells by calculating projection on the ref barcode vector
% and/or mismatches. qual/int thresh are optional


%filt= min(bc.qual)>qual_thresh & ...
%    min(bc.int)>int_thresh;
% 
%all cellular barcodes
all_cell_bc=cell2mat(filt_neurons.dom_bc(~cellfun(@isempty,filt_neurons.dom_bc)));
all_cell_bc_count=filt_neurons.dom_bc_count(~cellfun(@isempty,filt_neurons.dom_bc));
all_cell_bc_cellid=filt_neurons.id(~cellfun(@isempty,filt_neurons.dom_bc));


% sort by counts, then throw away cells with repeated barcodes
[~,I]=sort(all_cell_bc_count,'descend');
all_cell_bc=all_cell_bc(I,:);
all_cell_bc_cellid=all_cell_bc_cellid(I);
%
[all_cell_bc,ia,~]=unique(all_cell_bc,'rows','first');
all_cell_bc_cellid=all_cell_bc_cellid(ia);
%label cells with repeat barcodes
d1=squareform(pdist(all_cell_bc,'hamming'));
d1=(d1+eye(size(d1)))*size(all_cell_bc,2);
repeat_bc=min(d1,[],2)<=mismatch_thresh;
%

% match bc to cells
batch_size=100000;
batch_num=floor(size(bc.seq,1)/batch_size);
batch_left=1+(0:batch_num)*batch_size;
batch_right=[(1:batch_num)*batch_size, size(bc.seq,1)];
idx=zeros(size(bc.seq,1),1);
for n=1:numel(batch_left)
    d=pdist2(bc.seq(batch_left(n):batch_right(n),:),all_cell_bc,'hamming')*size(bc.seq,2);
    [dmin,I]=min(d,[],2);
    idx(batch_left(n):batch_right(n))=(dmin<=mismatch_thresh).*(sum(d==dmin,2)==1).*I;
end
bc.matched_cell=zeros(size(bc.pos40x,1),1);
bc.matched_cell(idx>0)=all_cell_bc_cellid(idx(idx>0));
%
filt_neurons.repeat_bc=repeat_bc;

save([data_fname(1:end-4),'-bc.mat'],'rolonies','filt_neurons','bc','-v7.3');
save('filt_neurons-bc.mat','filt_neurons');

end

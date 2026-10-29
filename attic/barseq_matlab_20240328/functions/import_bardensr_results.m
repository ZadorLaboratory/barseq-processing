function import_bardensr_results(batches, codebook_name)
    if ~exist('codebook_name','var')
        codebook_name='codebook.mat';
    end
    [folders,~]=get_folders();
    load(codebook_name,'codebook');
    id2={};
    lroi={};
    %for i=55
    for i=1:numel(folders)    
        cd(folders{i})
        cd aligned
        try
            m=dlmread('bardensrresult.csv',',',1,0);
            lroi{i}=m(:,[3 2])+1;
            id2{i}=m(:,4)+1;
        catch
            warning('%s has no rolonies\n',folders{i});
            lroi{i}=[];
            id2{i}=[];
        end
    
        
        cd ../..
    end
    save('basecalls.mat','lroi','id2','-v7.3')
    % manually check false positive rate, reiterate basecalling if necessary (keep fpr <5%)
    unusedidx=find(contains(codebook(:,1),'unused'));
    %sum(ismember(cell2mat(id2'),unusedidx))
    %numel(cell2mat(id2'))
    fpr=sum(ismember(cell2mat(id2'),unusedidx))/numel(cell2mat(id2'))/numel(unusedidx)*sum(~contains(codebook(:,1),'unused'));
    fprintf('Finished importing bardensr results.\n Overall FPR is %.3u\n',fpr);
    
    if exist('batches','var')
        uniq_batches=unique(batches);
        num_batches=numel(uniq_batches);
        for i=1:num_batches
            fpr=sum(ismember(cell2mat(id2(batches==uniq_batches(i))'),unusedidx))/numel(cell2mat(id2(batches==uniq_batches(i))'))/numel(unusedidx)*sum(~contains(codebook(:,1),'unused'));
            fprintf('Batch %u, FPR is %.3u\n',uniq_batches(i),fpr);
        end
    end
    
    
            



end
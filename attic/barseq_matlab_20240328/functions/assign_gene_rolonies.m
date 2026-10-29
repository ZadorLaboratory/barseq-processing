function assign_gene_rolonies()
    load('basecalls.mat','lroi');
    load('allsegmentation.mat','imcell');
    load('genehyb.mat','lroihyb');
    
    folders=get_folders();
    
    cellid={};cellidhyb={};
    parfor i=1:length(folders)
        if ~isempty(lroi{i})
            cellid{i}=mmassignrol2cell(lroi{i},imcell{i});
        else
            cellid{i}=[];
        end
        if ~isempty(lroihyb{i})
            cellidhyb{i}=mmassignrol2cell(lroihyb{i},imcell{i});
        else
            cellidhyb{i}=[];
        end
    end
    %
    save('cellid.mat','cellid','cellidhyb','-v7.3');
end
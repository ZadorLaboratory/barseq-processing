function assign_bc2cell()
    load('bc.mat','bclroi');
    load('allsegmentation.mat','imcell');
    % assign rolonies to cells
    folders=get_folders();
    bccellid={};
    parfor i=1:length(folders)
        if ~isempty(bclroi{i})
            bccellid{i}=mmassignrol2cell(bclroi{i},imcell{i}); %rolonies outside of cells are 0
        else
            bccellid{i}=[];
        end
    end
    %
    save('bccellid.mat','bccellid','-v7.3');
    fprintf('Assigned barcodes to cells.\n')
end

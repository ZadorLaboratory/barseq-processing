function rolonies_to_10x()
    load('basecalls.mat','lroi');
    load('genehyb.mat','lroihyb');
    load('allsegmentation.mat','cellpos');
    load('40xto10x.mat','tform40xto10x');
    
    % transform rolony and cell coordinates to 10x
    lroi10x=lroi;
    lroihyb10x=lroihyb;
    cellpos10x=cellpos;
    for i=1:length(lroi)
        if ~isempty(lroi{i})
            lroi10x{i}=transformPointsForward(tform40xto10x{i},lroi{i});
        end
        if ~isempty(lroihyb{i})
            lroihyb10x{i}=transformPointsForward(tform40xto10x{i},lroihyb{i});
        end
        if ~isempty(cellpos{i})
            cellpos10x{i}=transformPointsForward(tform40xto10x{i},cellpos{i});
        end
    end
    save('lroi10x.mat','lroi10x','lroihyb10x','cellpos10x','-v7.3');
end
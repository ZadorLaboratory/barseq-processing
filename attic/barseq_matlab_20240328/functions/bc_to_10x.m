function bc_to_10x()
    % transform rolony coordinates to 10x
    load('bc.mat','bclroi');
    load('40xto10x.mat','tform40xto10x');
    bclroi10x=bclroi;
    for i=1:length(bclroi)
        if ~isempty(bclroi{i})
            bclroi10x{i}=transformPointsForward(tform40xto10x{i},bclroi{i});
        end
    
    end
    save('bclroi10x.mat','bclroi10x','-v7.3');
    fprintf('Transformed barcodes to stitched 10x coordinates.\n')
end
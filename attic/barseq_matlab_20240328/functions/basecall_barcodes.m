function [bcseq, bcseqC, bcscore, bcsig, bclroi,bcint, bcqual]=basecall_barcodes(rolthresh, gaussrad,relaxed)
if ~exist('relaxed','var')
    relaxed=0;
end
folders=get_folders();
parfor i=1:numel(folders)
    cd(folders{i})
    cd aligned
    [bcseq{i},bcseqC{i},bcscore{i},bcsig{i},bclroi{i}]=mmbasecallsinglerol(rolthresh,gaussrad,relaxed);
    bcint{i}=max(bcsig{i},[],3);
    bcqual{i}=bcint{i}./sqrt(sum(bcsig{i}.^2,3));
    cd ../..
end

%
save('bc.mat','bcsig','bcseq','bcqual','bcint','bcseqC','bcscore','bclroi','-v7.3');
fprintf('Basecalled barcodes.\n')
end

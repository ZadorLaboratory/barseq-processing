function organize_hyb_files_multi(hybfolder)
%this can't run before geneseq
% move files
    folders=get_folders();
    if ~isempty(folders)
        %
        cd ..
    end

    if ~exist('hybfolder','var')
        hybfolder=dir('hyb*');
        hybfolder={hybfolder([hybfolder.isdir]).name};
        cd processed
    end
    

    for n=1:numel(hybfolder)
        cd(['../',hybfolder{n}]);
        if ismember(hybfolder(n),{'hyb01'})
            for i=1:length(folders)
                movefile([folders{i},'.tif'],fullfile('..','processed',folders{i},[hybfolder{n},'.tif']));
                %copyfile([folders{i},'.tif'],fullfile('..','processed',folders{i},'nuclear.tif'));
            end
        else

            % copy sequential round images to the correct folders
            for i=1:length(folders)
                movefile([folders{i},'.tif'],fullfile('..','processed',folders{i},[hybfolder{n},'.tif']));
            end
        end

    end
    cd ../processed
end
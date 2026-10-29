function cleanup_files(zip_aligned)
%zip aligned folder. zip original folder. delete original files pre n2v.
%Option to skip aligned files
if ~exist('zip_aligned','var')
    zip_aligned=1;
end

folders=get_folders();
parfor i=1:numel(folders)
    if exist(fullfile(folders{i},'original'),'dir')
        zip(fullfile(folders{i},'original.zip'),fullfile(folders{i},'original'));
        rmdir(fullfile(folders{i},'original'),'s');
    else
        fprintf('Folder %s already compressed, skip.\n',fullfile(folders{i},'original'));
    end

    if zip_aligned
        if exist(fullfile(folders{i},'aligned'),'dir')
            zip(fullfile(folders{i},'aligned.zip'),fullfile(folders{i},'aligned'));
            rmdir(fullfile(folders{i},'aligned'),'s');
        else
            fprintf('Folder %s already compressed, skip.\n',fullfile(folders{i},'aligned'));
        end
    else
        fprintf('Skpping aligned files.\n')
    end
end

% cleanup original files
folders=get_folders();
for i=1:numel(folders)
    cd(folders{i});
    delete *.tif
    cd ..
end

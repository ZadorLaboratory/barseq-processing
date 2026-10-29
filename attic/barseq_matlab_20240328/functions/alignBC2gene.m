function alignBC2gene(BC_refch,gene_refch,BC_name,gene_name)
% align BC to genes

folders=get_folders();
mkdir('BCtogenealignmentcomparison');
tic
parfor i=1:length(folders)
    cd(folders{i});
    % if genes have been registered, move the original gene files out.
    if isfolder('original')
        cd original
        if ~isempty(dir(['*',gene_name,'*.tif']))
            movefile(['*',gene_name,'*.tif'], '../');
        end
        regfiles=dir('*reg*.tif');
        if ~isempty(regfiles)
            delete *reg*.tif
        end
        cd ..
    end
    %if BC has been registered before, delete the registered files.
    regfiles=dir('*reg*.tif')
    if ~isempty(regfiles)
        delete *reg*.tif
    end
    
    mmalignBCtogene(BC_refch,gene_refch,BC_name,gene_name);
    movefile('comp.tif',fullfile('..','BCtogenealignmentcomparison',[folders{i},'comp.tif']));
    
    %if genes have been registered, move the original gene files back.
    if isfolder('original')
        movefile(['*',gene_name,'*.tif'], 'original/')
    end
   
    cd ..
end
toc
end
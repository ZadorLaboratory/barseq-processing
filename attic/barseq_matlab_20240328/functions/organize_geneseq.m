function organize_geneseq()
    % Sort max proj files into sequences.
    fprintf('Putting geneseq files in the correct folders ...')
    genefolders=dir('*geneseq*');
    genefolders(~[genefolders.isdir])=[];
    genefolders=sort_nat({genefolders.name});
    
    mkdir processed
    
    for i=1
        fprintf(genefolders{i})
        cd([genefolders{i}]);
        files=dir('MAX*.tif');
        files=sort_nat({files.name});
        parfor (n=1:length(files),4)
            filename=textscan(files{n},'%s','Delimiter','.');%remove .tif
            filename=filename{1}{1};
            mkdir(['../processed/',filename]);
            movefile(files{n},['../processed/',filename,'/','geneseq01.tif']);
        end
        cd ..
    end
    
    for i=2:length(genefolders)
        fprintf(genefolders{i})
        cd([genefolders{i}]);
        files=dir('MAX*.tif');
        files=sort_nat({files.name});
        parfor (n=1:length(files),4)
            filename=textscan(files{n},'%s','Delimiter','.');%remove .tif
            filename=filename{1}{1};
            movefile(files{n},['../processed/',filename,'/','geneseq',num2str(i,'%.2u'),'.tif']);
        end
        cd ..
    end
    fprintf('Done.')
end
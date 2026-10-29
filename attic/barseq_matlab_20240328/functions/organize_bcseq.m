function organize_bcseq()
% Sort max proj files into sequences.
fprintf('Putting bcseq files in the correct folders ...')


if ~exist('processed','file')
    mkdir processed;
    isfirstcycle=1;
else
    isfirstcycle=0;
end

%bc seq
bcfolders=dir('*bcseq*');
bcfolders(~[bcfolders.isdir])=[];
bcfolders=sort_nat({bcfolders.name});

if isfirstcycle==1
    for i=1
        fprintf(bcfolders{i})
        cd([bcfolders{i}]);
        files=dir('MAX*.tif');
        files=sort_nat({files.name});
        parfor (n=1:length(files),4)
            filename=textscan(files{n},'%s','Delimiter','.');%remove .tif
            filename=filename{1}{1};
            mkdir(['../processed/',filename]);
            movefile(files{n},['../processed/',filename,'/','bcseq01.tif']);
        end
        cd ..
    end

    for i=2:length(bcfolders)
        fprintf(bcfolders{i})
        cd([bcfolders{i}]);
        files=dir('MAX*.tif');
        files=sort_nat({files.name});
        parfor (n=1:length(files),4)
            filename=textscan(files{n},'%s','Delimiter','.');%remove .tif
            filename=filename{1}{1};
            movefile(files{n},['../processed/',filename,'/','bcseq',num2str(i,'%.2u'),'.tif']);
        end
        cd ..
    end
else
    for i=1:length(bcfolders)
        cd([bcfolders{i}]);
        files=dir('MAX*.tif');
        files=sort_nat({files.name});
        for n=1:length(files)
            filename=textscan(files{n},'%s','Delimiter','.');%remove .tif
            filename=filename{1}{1};
            movefile(files{n},['../processed/',filename,'/','bcseq',num2str(i,'%.2u'),'.tif']);
        end
        cd ..
    end
end
fprintf('Done\n');
end
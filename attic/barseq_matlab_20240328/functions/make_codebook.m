function make_codebook(codebookname,cycle_num)
% try to find out how many cycles were sequenced. Truncate codebook as needed.
if ~exist('cycle_num','var')
    folders=get_folders();
    orig_dir=cd(folders{1});
    f=dir('*geneseq*.tif');
    if ~isempty(f)
        f={f.name};
        cycle_num=max(sum(contains(f,'n2v')),sum(~contains(f,'n2v')));
    elseif isfolder('aligned')
        cd('aligned')
        f=dir('*geneseq*.tif');
        cycle_num=numel(f);
    else
        cycle_num=0;
    end
    cd(orig_dir);
end



load(codebookname,'codebook')
codebook1=char(codebook(:,2));
if cycle_num~=0
    codebook1=codebook1(:,1:cycle_num);
end
codebookbin=double(size(codebook1));
codebookbin(codebook1=='G')=8;
codebookbin(codebook1=='T')=4;
codebookbin(codebook1=='A')=2;
codebookbin(codebook1=='C')=1;
codebookbin=reshape(codebookbin,size(codebook1,1),[]);
codebookbin=codebookbin*(2.^((4*size(codebook1,2)-4):-4:0))';
codebookbin=uint8(dec2bin(codebookbin))-48;
codebookbin=[codebook(:,1),mat2cell(codebookbin,ones(size(codebookbin,1),1))];
save('codebook.mat','codebook','codebookbin');
save('../codebook.mat','codebook','codebookbin');
%save codebook for bardensr
codebookbin1=reshape(cell2mat(codebookbin(:,2)),size(codebookbin,1),4,[]);
save('../codebookforbardensr.mat','codebookbin1');
end
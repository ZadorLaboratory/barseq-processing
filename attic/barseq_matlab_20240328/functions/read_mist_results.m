function read_mist_results(fname, rescale_factor)

if ~exist('fname','var')
    fname='n2vgeneseq01';
end
if ~exist('rescale_factor','var')
    rescale_factor=0.5;
end


%%
[folders,pos,xxx,yyy]=get_folders;
uniqpos=unique(pos);
T_mat=cell(numel(uniqpos),1);
T_fnames=T_mat;
for n=1:numel(uniqpos)
    tform_fname=fullfile('stitch',fname,'stitched',['MAX_',uniqpos{n},'_global-positions-0.txt']);
    in_pos=ismember(pos,uniqpos(n));
    T=readtable(tform_fname, ...
    'ReadVariableNames',0, ...
    'delimiter',{' ',':',';',',','(',')'}, ...
    'ConsecutiveDelimitersRule','join');

    T_mat{n}=[T.Var6,T.Var7];
    T_fnames{n}=cellfun(@(x)x(1:end-4),T.Var2,'UniformOutput',false);

end
T_mat=cell2mat(T_mat);
T_fnames=vertcat(T_fnames{:});


%sort T_fnames
[~,I]=ismember(folders',T_fnames);
tforms=T_mat(I,:)*rescale_factor;

for n=1:numel(I)
        T=[rescale_factor,0,0;0,rescale_factor,0;tforms(n,:),1];
    tforms_converted{n}=affine2d(T);
end
tform40xto10x=tforms_converted;


save('40xto10x.mat','tform40xto10x');
fprintf('All done. Saved transformations to 40xto10x.mat.\n')
end

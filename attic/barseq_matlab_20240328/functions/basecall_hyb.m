
function basecall_hyb(hybthresh,hybbgn,codebookhyb,no_deconv,filter_overlap,relaxed)
[folders,~]=get_folders();

if ~exist('no_deconv','var') % deconvolution during peak finding, default is with deconv.
    no_deconv=0;
end

if ~exist('filter_overlap','var') %filter overlapping rolonies in the first two channels? defaul is no
    filter_overlap=0;
end

if ~exist('codebookhyb','var')  % set a default codebook if not specified.
    load('../codebookhyb.mat','codebookhyb');
end

if ~exist('relaxed','var')  % testing relaxed peak calling
    relaxed=0;
end


load('psf.mat','psf')
% read sequential signals.
idhyb={};
lroihyb={};
sighyb={};
chprofile=eye(4);
parfor(i=1:numel(folders),20)
%for i=1:numel(folders)
    cd(folders{i});
    cd aligned

    if ischar(codebookhyb{1,2})
        [idhyb{i},lroihyb{i},sighyb{i}]=mmbasecallhyb_multi(codebookhyb,hybthresh,hybbgn,psf,chprofile,no_deconv,relaxed);
    else
        [idhyb{i},lroihyb{i},sighyb{i}]=mmbasecallhyb(codebookhyb,hybthresh,hybbgn,psf,chprofile,no_deconv,relaxed);
    end
    
    cd ../..
end




%% need to add code to check and filter overlapping rolonies when not correcting for shift.
if filter_overlap ~= 0
    %filter overlapping neurons here
end



save('genehyb.mat','idhyb','lroihyb','sighyb','-v7.3');
end
function [cellid,seq,seqC,sig,score]=basecall_somas(thread_num)
%using segmented images, call barcodes from cells based on INTENSITY. If a nucprofile is
%provided, then reduce nuclear background. nucprofile should be a 1x4
%vector that gives the relative strengths of nuc background signals in the
%four channels.
if ~exist('thread_num','var')
    thread_num=18;
end

folders=get_folders();
cellid={}; seq={};seqC={};sig={};score={};
parfor(i=1:numel(folders),thread_num)
    %load cell mask
    original_dir=cd(fullfile(folders{i},'aligned'));
    S=load('cellmask.mat');
    maski=S.maski;
    [cellid1,seq1,seqC1,sig1,score1]=mmbasecallcells('bcseq',maski); % this is based on mean signal in segmented somas
    m=matfile('bcsoma.mat','Writable',true)
    m.cellid1=cellid1;
    m.seq1=seq1;
    m.sig1=sig1;
    m.seqC1=seqC1;
    m.score1=score1;

    cellid{i}=cellid1;
    seq{i}=seq1;
    seqC{i}=seqC1;
    sig{i}=sig1;
    score{i}=score1;
    cd(original_dir)
end

save('all_bccells_intensity.mat','cellid','seq','seqC','sig','score');




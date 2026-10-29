
function add_somabc(fname)
% load basecalled soma data and filt_neurons-bc, add soma bc data.
% this should run after adding non-soma bc data, but also has the option of
% running without callin bc rolonies.
if ~exist('fname','var')
    fname='filt_neurons-bc.mat';
end
try
    S=load(fname);
    filt_neurons=S.filt_neurons;
catch ME
    fprintf('Can\''t load filt_neurons, abort.\n' );
    rethrow(ME);
end
try
    T=load('all_bccells_intensity.mat');
    seq=T.seq';
    seqC=T.seqC';
    sig=T.sig';
    cellid=(T.cellid');
    score=T.score';
catch ME
    fprintf('Can\''t load soma_bc information, abort.\n' );
    rethrow(ME);
end



for n=1:numel(cellid)
    cellid{n}=double(cellid{n})+n*10000;%restore id in filt_neurons
end
cellid_ex=cell2mat(cellid(~cellfun(@isempty,cellid)));
seq_ex=cell2mat(seq(~cellfun(@isempty,cellid)));
sig_ex=cell2mat(sig(~cellfun(@isempty,cellid)));
score_ex=cell2mat(score(~cellfun(@isempty,cellid)));

[~,I]=ismember(filt_neurons.id,cellid_ex);
if sum(I==0)>0
    error('Some filt_neuron ids don\''t match soma cellid, check for errors.\n');
else
    filt_neurons.soma_bc=seq_ex(I,:);
    filt_neurons.soma_bc_sig=sig_ex(I,:,:);
    filt_neurons.soma_bc_score=score_ex(I,:);
end

save('filt_neurons-bc-somas.mat','filt_neurons','-v7.3');
end

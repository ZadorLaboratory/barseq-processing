
function filter_somabc(fname,varargin)
% determine which cells are barcoded



if ~exist('fname','var')
    fname='filt_neurons-bc-somas.mat';
end
% parse thresholds
if ~isempty(varargin)
    varargin=reshape(varargin,2,[]);
    for i=1:size(varargin,2)
        switch lower(varargin{1,i})
            case 'complexity'
                complexity_thresh=varargin{2,i};
            case 'score'
                score_thresh=varargin{2,i};
            case 'signal'
                sig_thresh=varargin{2,i};
        end
    end
end


% default thresholds
if ~exist('complexity_thresh','var')
    complexity_thresh=-0.9;
end
if ~exist('score_thresh','var')
    score_thresh=0.85;
end
if ~exist('sig_thresh','var')
    sig_thresh=200;
end

%load dataset
S=load(fname,'filt_neurons');
filt_neurons=S.filt_neurons;

%calculate complexity of barcodes
bc_complexity=zeros(numel(filt_neurons.id),1);
for i=1:numel(filt_neurons.id)
    bc_complexity(i)=log10(prod(lingseqcomplexity(char(48+filt_neurons.soma_bc(i,:)))));
end



pass_complexity=bc_complexity>=complexity_thresh;
high_score=median(filt_neurons.soma_bc_score,2)>=score_thresh; %median score
high_sig=median(max(filt_neurons.soma_bc_sig,[],2),3)>=sig_thresh; %median signal of max channel

filt_neurons.is_barcoded=pass_complexity&high_score&high_sig;
save('filt_neurons-bc-somas.mat','filt_neurons','-v7.3');

end


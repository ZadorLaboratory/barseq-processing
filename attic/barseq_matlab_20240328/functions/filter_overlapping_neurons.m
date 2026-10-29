function filt_neurons=filter_overlapping_neurons(fname_neurons,boxsize,pixelsize)
load(fname_neurons,'neurons');


if ~exist('boxsize','var')
    boxsize=5;
end

if ~exist('pixelsize','var')
    pixelsize=6.5/10;
end

%
slice=unique(neurons.slice)';

removecells_all=zeros(numel(neurons.id),1);
%
for i=slice
    %
    inslice=neurons.slice==i;
    sliceneuron_fovnum=floor(neurons.id(inslice,:)/10000);
    sliceneuron_pos=neurons.pos(inslice,:);
    sliceneuron_expmat=neurons.expmat(inslice,:);

    % "sort and sweep' to identify overlapping cells
    % make every cell into a 10um square box. This is smaller than a lot of
    % cells, but sincewe are looking for complete overlaps, smaller shoudl
    % be fine.

    sliceneuron_x_l=sliceneuron_pos(:,1)*pixelsize-boxsize/2;
    sliceneuron_x_u=sliceneuron_pos(:,1)*pixelsize+boxsize/2;
    sliceneuron_y_l=sliceneuron_pos(:,2)*pixelsize-boxsize/2;
    sliceneuron_y_u=sliceneuron_pos(:,2)*pixelsize+boxsize/2;
    %sort intervals
    [interval_list_x,AABB_x]=sort([sliceneuron_x_l;sliceneuron_x_u]);
    [~,AABB_x]=sort(AABB_x);
    AABB_x=reshape(AABB_x,[],2);
    [interval_list_y,AABB_y]=sort([sliceneuron_y_l;sliceneuron_y_u]);
    [~,AABB_y]=sort(AABB_y);
    AABB_y=reshape(AABB_y,[],2);
    %for each upper bound, find everything smaller than it in the interval
    %list
    %
    c={};
    c1={};
    %tic
    for n=1:size(AABB_x,1)
        %c{n}=find(AABB_x(n,2)>AABB_x(:,1)&AABB_x(n,2)<AABB_x(:,2)& ...
        %    AABB_y(n,2)>AABB_y(:,1)&AABB_y(n,2)<AABB_y(:,2)& ...
        %    sliceneuron_fovnum~=sliceneuron_fovnum(n)); % cells that are potentially overlapping and from different FOVs
        c1{n}=find(AABB_x(n,2)>AABB_x(:,1)&AABB_x(n,2)<AABB_x(:,2)& ...
            AABB_y(n,2)>AABB_y(:,1)&AABB_y(n,2)<AABB_y(:,2)); %just overlapping
        if ~isempty(c1{n})
            c{n}=c1{n}(sliceneuron_fovnum(c1{n})~=sliceneuron_fovnum(n));
        else
            c{n}=c1{n};
        end
    end
    %numel(cell2mat(c'))
    %numel(cell2mat(c1'))
    fprintf('Slice %u: %u out of %u filtered cells are from different fovs\n',i,numel(cell2mat(c')),numel(cell2mat(c1')));

    %toc



    % Ideally we want to evaluate  cell overlap pixel by pixel, but this may  be unnecesasry.
    % Go through the potentially overlapping list, remove cells with lower read counts.
    %tic

    d=find(~cellfun(@isempty,c));
    remove_neuron=zeros(numel(c),1);
    for m=d
        if remove_neuron(m)==0
            idx=[m;c{m}];
            if sum(idx>numel(remove_neuron))>0
                m
            end
            [~,keep_idx]=max(sum(sliceneuron_expmat(idx,:),2));
            idx(keep_idx)=[];
            remove_neuron(idx)=1;
        end
    end

    %toc
    removecells_all(inslice)=remove_neuron;
 
end

% remove the labeled cells
filt_neurons=neurons;
filt_neurons.expmat=filt_neurons.expmat(removecells_all==0,:);
filt_neurons.id=filt_neurons.id(removecells_all==0,:);
filt_neurons.pos=filt_neurons.pos(removecells_all==0,:);
filt_neurons.depth=filt_neurons.depth(removecells_all==0,:);
filt_neurons.angle=filt_neurons.angle(removecells_all==0,:);
filt_neurons.slice=filt_neurons.slice(removecells_all==0,:);
filt_neurons.pos40x=filt_neurons.pos40x(removecells_all==0,:);
filt_neurons.fov=filt_neurons.fov(removecells_all==0,:);

%only for barcoded data
if isfield(filt_neurons,'dom_bc')
    filt_neurons.dom_bc=filt_neurons.dom_bc(removecells_all==0,:);
    filt_neurons.barcoded=cellfun(@(x) ~isempty(x),filt_neurons.dom_bc);

end
if isfield(filt_neurons,'dom_bc_count')
    filt_neurons.dom_bc_count=filt_neurons.dom_bc_count(removecells_all==0,:);
end
if isfield(filt_neurons,'all_bc')
    filt_neurons.all_bc=filt_neurons.all_bc(removecells_all==0,:);
end
if isfield(filt_neurons,'all_bc_count')
    filt_neurons.all_bc_count=filt_neurons.all_bc_count(removecells_all==0,:);
end

save('filt_neurons.mat','filt_neurons','removecells_all');
end

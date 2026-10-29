% Note: run 01 after 00 finishes. We separate the two to provide a break so
% that the operator can visually examine registration accuracy and
% stitching accuracy. This was essential in the early days, but with
% standardized and automated runs, this should not be needed. Consider
% making it possible to call 01 directly at the end of 00 without a pause.


%% Section 1: Configurations for each experiment. 
% Reads the configuration mat file generated in 00 

f=dir('*input.mat');
f={f.name};
load(f{1});

% Do you want to zip files after this analysis is complete? It will save space 
% on your analysis and storage computers. But set this to 0 if you might want 
% to do more image processing on this dataset or look at raw/aligned images. 

zip_files = 1; % we usually set this to 1 unless foreseeing troubleshooting/optimization after processing a dataset. 
% Create a log book for each analysis run

t=datetime('today','Format','yyyyMMdd');
diary([char(t),'_a01.log']);

% Check that FIJI is installed with the proper plugins

try 
    Miji(false)
    MIJ.exit
    fprintf('MIJ installed correctly.\n')
catch ME
    javaaddpath('C:\barseq_env\FIJI_jar\mij.jar')
    try 
        Miji(false)
        MIJ.exit
    catch ME2
        error('MIJ was not set up and not found in path. Were all env files set up properly?\n')
    end
    warning('MIJ was not set up. Successfully added MIJ to path.\n')
end
 
% Create parallel processing pool (for processing multiple images simultaneously). 

clc
numcores=feature('numcores');

p=gcp('nocreate');
if isempty(p)
    parpool("local",round(numcores*1.5));
elseif p.NumWorkers<round(numcores*1.5)
    parpool("local",round(numcores*1.5));
end
%% Run bardensr (python package) on gene sequencing cycles. This basecalls each 
% rolony and assigns them a code (ACTG). Then checks against the provided codebook 
% and matches each rolony to a gene
% 
% Use_predefined_thresh is defined in the initial parameters, only non-zero 
% if you are running more than 1 batch
% 
% import_bardensr_results saves a file basecalls.mat with all bardensr results

cd processed
clc
load('40xto10x.mat','tform40xto10x');
batches=ones(1,numel(tform40xto10x))*input.batch_num; 
save('batches.mat','batches');

bardensr_cmdout=run_bardensr(input.use_predefined_thresh);


import_bardensr_results(batches, fullfile('..',input.codebook_name));
%% Basecall hybridization cycle(s). This code also outputs an image from one 
% FOV for you to visually check (default FOV #35, if there are less than 35 FOVs, 
% picks the middle FOV). 
% 
% 
% 
% Basecalls the hybridization channels based on your provided hybridization 
% codebook: matches spots in each channel directly to genes in codebook. 


load(fullfile('..',input.codebookhyb_name),'codebookhyb');
relaxed=1;
no_deconv=1;
filter_overlap=0;
basecall_hyb(input.hybthresh,input.hybbgn,codebookhyb,no_deconv,filter_overlap,relaxed);


folders = get_folders();
if numel(folders)>=35
FOV=35;
else 
FOV=round(numel(folders)/2);
end

check_hyb_call(FOV,codebookhyb)
exportgraphics(gcf,fullfile('..',['FOV',num2str(FOV),'_hybcall.jpg']));
%% Segment cells using cellpose (python package). Current cellpose uses the "all 
% genes" channel from hybridization cycle as the cytoplamsic signal and DAPI as 
% the nuclear signal. Saves a file with cell masks for each FOV. 
% 
% import_cellpose takes all cell mask lists and saves one file allsegementation.mat 
% with a list of all segmented cells, cell masks and positions. 

% run cellpose in python, dilate each cell by 3 pixels
cellpose_cmdout=run_cellpose();
dilation_radius=3;
import_cellpose(dilation_radius);

% Note: cellpose parameters may need to be tuned for different
% species/regions/ages. Consider making it possible to pass parameters from
% config.json into cellpose. 

%% Assign rolonies to cells, combine bardensr and cellpose datasets to assign 
% each rolony to a cell if it falls within a cell mask. 
% 
% transfer all coordinates to 10x (full slice), based on stitching from a00
% 
% organize data into output file "alldata_yyyymmdd.mat", containing data from 
% all the rolonies individually and from neuron-based data. 

assign_gene_rolonies();

rolonies_to_10x();

use_mock=1;
calculate_depth(use_mock);% we always use mock these days and no longer use depth and angle in filt_neurons. Leave this out in the final implementation.

organize_processed_data(input.startingsliceidx);
%% Find overlapping cells and remove them (overlaps between FOVs). Outputs "filt_neurons.mat" 
% which is used in most of our future analysis. This saves a file in cell x gene 
% and other metadata format. Set box size bigger or larger if you want to be more 
% or less stringent for detecting overlapping cells. 

data_fname=dir('alldata*.mat');
data_fname=sort_nat({data_fname.name});
data_fname=data_fname{end};

boxsize=5; % box size for identifying overlaps
pixelsize=6.5/10; % 10x pixel size
filter_overlapping_neurons(data_fname,boxsize,pixelsize);

%Caveat: Currently we consider a neuron as a box of uniform size with
%median at the location of the soma and edges of certain length. This is a
%crude approximation and can miss many cells. Could be much better to use
%the real bounding box of neurons based on their segmentations. Also, we
%currently don't follow up the sort-n-sweep with a more precise check of
%potential overlapping neurons. This should probably be added.


%%  Basecall and assign barcodes, if barcoded. Add to filt_neurons.mat. Note quality 
% control values on filter_somabc, can change this if you want to adjust sensitivity. 

if input.is_barcoded==1
    gaussrad=0;
    basecall_barcodes(input.rolthresh, gaussrad); %change how barcodes are "called" - but not for all of them just those in cell
    % assign barcodes rolonies to neurons
    assign_bc2cell(); %%This is the code I probably have to change
    %transform barcodes to 10x coordinates
    bc_to_10x();
    % organize and error correct barcode rolonies
    data_fname=dir('alldata*.mat');
    data_fname=sort_nat({data_fname.name});
    data_fname=data_fname{end};
    mismatch_thresh=1; % allow this mismatch when matching barcodes
    organize_bc_data(input.count_thresh, ...
        input.err_corr_thresh, ...
        data_fname, ...
        mismatch_thresh);
    % basecall somas and add soma bc data
    mem=memory;
    thread_num=floor(mem.MemAvailableAllArrays/2^30/8);% need ~8GB ram per thread for basecalling somas.
    basecall_somas(thread_num);
    add_somabc();
    % QC soma barcodes. 
    filter_somabc('filt_neurons-bc-somas.mat', ...
        'complexity',-0.9, ...
        'score',0.85, ...
        'signal',150 ...
        );

    % Note: currently the complexity, score, and signal thresholds in
    % filter_somabc are hard coded. Consider moving these to config.json.
    % Currently, we rarely tune these but I can imagine that people might want to
    % change these for differernt types of experiments.
end
%% Convert RGB to .jpg, zip files for smaller storage size

% convert_RGB_jpg
if zip_files == 1
    cleanup_files
end

% Note: In this step, the 8-bit RGB tif files (which we use for manually
% inspecting the data) are converted to jpg, and the 16-bit data tif files
% are zipped. There is no reason not to write the 8-bit RGB files directly
% into jpg. Consider changing this in steps that call rgbout1 (this
% includes most registration steps).


%% Log off. a01 typically takes only a few hours to 1 day to finish running, 
% depending on number of cycles and area imaged. 

diary off
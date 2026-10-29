%% inputs needed.

% Data: max projection images from the microscope. Each file should be
% stored in a multi-plane tif file with the name: {cycle_name}/MAX_Pos{n}_{xxx}_{yyy}.tif.
% This file should contain 5 images (G, T, A,C, DIC) for geneseq01 and
% bcseq01, 4 images (G,T,A,C) for any non-first cycle geneseq and bcseq,
% and 6 images (GFP, YFP, TxRed,Cy5, DAPI, DIC) for hyb cycles.

% Codebook: The user should provide a codebook for hyb cycles and a
% codebook for geneseq cycles. In the geneseq codebook, the unused-{n}
% genes are required at the end for estimating FDR during basecalling. For
% hyb codebook, there are two formats. If an experiment contains a single
% hyb cycle (this is the majority of cases), then the hyb code is a number
% (1, 2, or 4) which correspond to the channel that the gene should appear
% in (1=GFP, 2=YFP, 4=Cy5). If an experiment contains multiple hyb cycles,
% then the code should be a string of numbers. The length of the string
% should equal the number of cycles, and the numbers should be all 0 except
% in a single digit. The number again correspond to the channel and the
% position of the number indicates which hyb cycle that gene is detected
% in. For example, a code '0010' indicates a gene found in the GFP channel
% in hyb03, and a code '4000' indicates a gene found in the Cy5 channel in
% hyb01.

% Config.json: This provides settings that users can potentially change in
% the processing.
% dataset_id: unique identifier for the dataset.
% batch_num: if an experiment is part of a larger dataset (e.g. multiple
% sequencing batches that cover a whole brain), this is used to distinguish
% different experiments within a large dataset
% startingsliceidx: the starting slice number. Usually 1 if a single batch,
% but for multiple batches, this should change to reflect the sequence of
% slices across batches.
% use_predefined_thresh: bardensr has the option of using a predefined
% threshold across multiple batches in the same large dataset. This maybe
% useful to reduce batch effect, but we usually set this to 0.
% scope_name: used to uniquely identify which channel profiles to load for
% each microscope. The channel profiles should be defined by each lab
% running their own experiments, but the profiles shouldn't need to be
% easily changed by an operator. i.e. the operator should specify scopes,
% whereas the profile that the scope correspond to should be set by the lab
% and usually stay unchanged.
% codebook_name: codebook filename for geneseq cycles
% codebookhyb_name: codebook filename for hyb cycle(s)
% hybthresh: prominence threshold for detecting dots in hyb cycles.
% hybbgn: intensity threshold for detecting dots in hyb cycles.
% hyb_reg_ch: the channel in hyb cycle that's used to register to geneseq
% cycles. We usually use channel 3 = Txred to do this. In most experiments,
% we use a Txred probe that detects all geneseq rolonies during hyb01. This
% is then registered to the sum of GTAC in geneseq01.
% is_barcoded: whether to process bcseq files
% rolthresh: prominence threshold for finding barcode rolonies
% count_thresh: We assign individual barcode rolonies to cells. If a cell
% contains >=count_thresh rolonies of a particular barcode, then we
% consider this cell as labeled by that barcode.
% err_corr_thresh: If a cell contains multiple barcodes, we collapse
% barcodes with <=err_corr_thresh to the most abundant barcode.

% Planned future changes:
% currently all of our datasets contain geneseq, hyb, and optionally bcseq.
% We are working on making this more flexible, so that you can run multiple
% runs of geneseq targetting different sets of genes. In that case, the
% other batches of geneseq may be called some arbitrary *seq name (e.g.,
% cdhseq), and they should be processed in the same way as geneseq (n2v,
% correction/registration, bardensr, assign to cells).
% We are also planning to run experiments with multiple bcseq batches
% (e.g., sindbis barcodes first, then rabies barcodes).
% If starting from scratch, I would consider asking the users to indicate
% all seq names in a sample, and indicate whether each should be treated as
% geneseq or bcseq. For geneseq, the user should provide a corresponding
% and separate codebook for each.

%% MATLAB setup
% Before you start, instructions for new computers setting up MATLAB properly 
% for BARseq processing
% 
% 1) Change the number of available cores for parallel processing. 
% 
% Click on preferences:
% 
% 
% 
% Click on Parallel Computing Toolbox, then Cluster Profile Manager
% 
% Make sure that the option for "shut down and delete parallel pool after it 
% is idle for:" is UNCHECKED. You want your parallel pool to remain on indefinitley. 
% 
% 
% 
% Click on the pencil for "edit" in toolbar and change "Number of workers to 
% start on your local machine" to 1.5x the number of cores your computer has. 
% Look this up in computer properties. Click done and save. 
% 
% 
% 
% 
% 
% 2) Change Java Heap Memory. 
% 
% Click on preferences:
% 
% 
% 
% Go to General -> Java heap memory
% 
% 
% 
% Set Java heap size to Maximum. Click "Apply". Close preferences 
% 
% 
% 
%% Section 1: Configurations for each experiment. 

% all configurations are now saved in a json file. This also saves a .mat
% file with all the settings in place
input=parse_barseq_config('config.json');

% Create a log book for each analysis run

t=datetime('today','Format','yyyyMMdd');
diary([char(t),'_a00.log']);
% Check that FIJI is installed with the proper plugins

try 
    Miji(false)
    MIJ.exit
    fprintf('MIJ installed correctly.\n')
catch ME
    javaaddpath('C:\barseq_envs\FIJI_jar\mij.jar')
    try 
        Miji(false)
        MIJ.exit
    catch ME2
        error('MIJ was not set up and not found in path. Were all env files set up properly?\n')
    end
    warning('MIJ was not set up. Successfully added MIJ to path.\n')
end
% Create parallel processing pool (for processing multiple images across multiple 
% folders simultaneously)

clc
numcores=feature('numcores');

p=gcp('nocreate');
if isempty(p)
    parpool("local",round(numcores*1.5));
elseif p.NumWorkers<round(numcores*1.5)
    parpool("local",round(numcores*1.5));
end
%% Organize files into the proper folders. 
% 
% organize_geneseq moves files from each gene cycle folder ( e.g. "geneseq01") 
% to a new folder named "processed", organized by each feild of view (FOV). We 
% will do all of the rest of our analysis within this processed folder. Here each 
% folder in processed is one FOV. 
% 
% Format: MAX_Pos*slice_number*_xxx_yyy. For example, "MAX_Pos4_003_002" contains 
% images from slice #4, four tiles from the right and three tiles down from the 
% top left of the image. All FOV numbers start from 000 and proceed sequentially 
% to the right (xxx) and down (yyy).  

organize_geneseq; %move files




%% 
% 
% 
% Denoise gene sequeincing images using N2V (python package)
% 
% N2V =  Noise to Void -> github: <https://github.com/hanyoseob/pytorch-noise2void 
% https://github.com/hanyoseob/pytorch-noise2void>
% 
% n2v takes model data that has been denoised and applies it to each of your 
% experimental images. It uses previously denoised models of each BARseq imaging 
% channel. To run this code, you will need to have the python script "n2vprocessing.py" 
% in your experimental folder, and have downloaded our "n2vmodel" folder, which 
% is included in the BARseq processing scripts package. 
% 
% Outputs: n2v processing creates a new version of each geneseq cycle in each 
% FOV folder. e.g. 'geneseq01.tif" -> 'n2vgeneseq01.tif' 

cmdout=n2v_processing('n2vprocessing.py');
%% 
% 
% 
% Register and align gene sequencing images for all cycles, fix bleedthrough. 
% (This step typically takes several hours - a few days)
% 
% Inputs: fname: default names for all geneseq denoised images. This code works 
% on files containing "n2vgene" created by the previous step
% 
% ball radius: filter size for background subtraction
% 
% rgb_intensity: Sets the brightness of the RGB output images after images have 
% been registered. Needs to be >0. Default of 0.6 works in most cases. 
% 
% local_registration: By default this is set to 1. Registers images locally, 
% which is faster than the alternative of MATLAB's registration function. Use 
% local_registration = 0 if the original registration fails or you are dealing 
% with strangely tilted images. 
% 
% chprofile20x: channel profile loaded above
% 
% chshift20x: channel shift loaded above

cd processed

%register geneseq files
fname='n2vgene';
ball_radius=6;
rgb_intensity=0.6;
local_registration = 1;
subsample_rate=4;
tic
register_seq_images_subsample(fname, ...
    input.scope_settings.chprofile20x, ...
    ball_radius, ...
    input.scope_settings.chshift20x, ...
    rgb_intensity, ...
    local_registration, ...
    subsample_rate)
t=toc;
%% 
% 
% 
% Make a codebook for reading out genes later (used in a01)
% 
% Inputs: reads codebook file (list of genes and associated genetic codes)
% 
% Outputs: binary version of codebook that is genes x channels x cycles. E.g. 
% for our 4 channel imaging cycles, with 104 genes and 7 nucleotide gene codes, 
% codebookforbardensr is 104 x 4 x 7

% Make codebook for bardensr
make_codebook(fullfile('..',input.codebook_name));
%% 
% 
% 
% Repeat same steps for hybridization files. Organize files and, denoise with 
% N2V

% Process hyb images
organize_hyb_files_multi();%move files

cd ..
n2v_processing('n2vprocessing_hyb.py');%n2v hyb images
cd processed
%%
%% 
% 
% 
% chprofilehyb: channel profile for calculating bleedthrough between hybridization 
% imaging channels
% 
% chshifthyb: pixel shift between hybridization imaging channels
% 
% 
% 
% hyb_reg_ch: which channel to align to, typically the channel with signal from 
% all genes
% 
% nuclearch: DAPI channel number
% 
% radius: radius for background subtraction on all channels
% 
% reg_cycle: which cycle in geneseq should you register to you. Typically leave 
% this at 1.
% 
% chradius: radius for background subtraction on sequencing cycles
% 
% rgb_intensity:  Sets the brightness of the RGB output images after images 
% have been registered. Needs to be >0. Default of 0.6 works in most cases. 


% register images
nuclearch=5; %nuclear channel
radius=100; %radius for bgn subtraction of all channels.
reg_cycle=1; % register to 1st cycle
chradius=30; % bgn radius for seq rolony cycle
rgb_intensity=0.6;

register_hyb_images_multi(input.hyb_reg_ch, ...
    radius, ...
    nuclearch, ...
    reg_cycle, ...
    chradius, ...
    input.scope_settings.chprofilehyb, ...
    input.scope_settings.chshifthyb, ...
    rgb_intensity);
%% 
% 
%% 
% Stitch images from the first cycle. This code uses the FIJI plugin MIST. 
% 
% fname: cycle name to be used as reference for stitching. This is usualy n2vgeneseq01 
% or regn2vbcseq01 or n2vhyb01
% 
% registration channel: which channel to use for registration. Leave at 4 unless 
% you need to change for some reason. 
% 
% overlap: percent overlap between each FOV (x and y direction)
% 
% rescale_factor: how much you want to downsample the images for the full stitched 
% image. Default is 0.5

%% Stitch images from first cycle
fname='n2vgeneseq01';
reg_ch=4;
overlap=0.23;
rescale_factor=0.5; % rescaling images by this
stitch_10x_images_mist(fname,reg_ch,overlap,rescale_factor);

% note: overlap is hard coded, but this should probably be moved to the
% scope config file (not operator defined).


%% Generate images to check registration quality: 
% stitch the transformed 40x 
% image and overlay with 10x images for visual inspection. This produces 5x smaller 
% images than the original 10x to ease loading, but should still be enough since 
% the registration only need to be roughly correct.

%check_stitching('40xto10x.mat') % nolonger work/not necessary with mist
% 
intensity_scaling=3;
checkregistration('geneseq','40xto10x.mat',intensity_scaling);
checkregistration('hyb','40xto10x.mat',intensity_scaling);
%%  IF this is a barcoded experiment. Now repeat all steps on barcoded images. 
% Organize, denoise with N2V, register and check registration. 

cd ..
if input.is_barcoded

    organize_bcseq()
    cmdout=n2v_processing('n2vprocessing_bc.py');
    cd processed
    
    %register BC to geneseq
    BC_refch=5;
    gene_refch=5;
    BC_name='n2vbcseq';
    gene_name='n2vgeneseq';
    alignBC2gene(BC_refch,gene_refch,BC_name,gene_name);
    
    % Register BC 40x

    fname='regn2vbc';
    ball_radius=100;
    rgb_intensity=0.6;
    local_registration=1;
    subsample_rate=4;
    register_seq_images_subsample(fname, ...
        input.scope_settings.chprofile20x, ...
        ball_radius, ...
        input.scope_settings.chshift20x, ...
        rgb_intensity, ...
        local_registration, ...
        subsample_rate);
%
    
    %generate checkregistration files
    intensity_scaling = 3;
    checkregistration('bcseq','40xto10x.mat',intensity_scaling);
    
    cd ..
end
%% Log off! 
% This full run takes anywhere from a few hours to several days to 
% complete depending on how fast your computer is, how many cycles and how large 
% of an area imaged. 

diary off



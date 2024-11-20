
% Logging
t=datetime('today','Format','yyyyMMdd');
diary([char(t),'_a00.log']);

warning('off','MATLAB:MKDIR:DirectoryExists')
mkdir output

%% configuration file paths, only change these
if ~exist('setting_fname','var')
    setting_fname='default_settings.json';
end
config_fname='config.json';

%% Setting up
% load processing settings
settings=load_settings(setting_fname);

% Parse configuration
config=load_experiment_config(config_fname);

% Check that FIJI is installed with the proper plugins
check_fiji_install()

% set up worker pools
numcores=feature('numcores');
[~,mem]=memory;
maxcore=min(round(numcores*1.5), floor(mem.PhysicalMemory.Total/4000000000));
p=gcp('nocreate');
if isempty(p)
    parpool("local",maxcore);
elseif p.NumWorkers<maxcore
    parpool("local",maxcore);
end

%%

if config.scope_settings.automation==1
    %if automated, then make a processed folder, and move all max folders
    %inside.
    mkdir processed
    movefile MAX* processed
end

%% Register images
   % Move geneseq files if not automated
if config.scope_settings.automation==0
    organize_geneseq; %move files
    if ~isempty(config.codebookoptional_name)
        organize_geneseq(settings.optseq_cyclename);
    end
end

% Denoise geneseq %Ellie/Angela
cmdout=n2v_processing(settings.gene_n2v_scriptname);
if ~isempty(config.codebookoptional_name)
    cmdout=n2v_processing([settings.gene_n2v_scriptname,' ',settings.optseq_cyclename]);
end

%% Register geneseq
cd processed
%
tic
register_seq_images_subsample_highres(settings.gene_fname, ...
    config.scope_settings.chprofile20x, ...
    settings.gene_ball_radius, ...
    config.scope_settings.chshift20x, ...
    config.scope_settings.gene_rgb_intensity, ...
    settings.gene_local_registration, ...
    settings.gene_subsample_rate, ...
    config.scope_settings.gene_max_thresh);
t=toc;

% Make codebook for bardensr
make_codebook(fullfile('..',config.codebook_name));
if ~isempty(config.codebookoptional_name)
    make_codebook_opt(fullfile('..',config.codebookoptional_name));
end

% register optional geneseq cycles
if ~isempty(config.codebookoptional_name)
    alignBC2gene(settings.opt2opt_refch, ...
        settings.opt2gene_refch, ...
        settings.opt_name, ...
        settings.opt_gene_name ...
        );
    register_seq_images_subsample_highres(settings.optseq_fname, ...
        config.scope_settings.chprofile20x, ...
        settings.opt_ball_radius, ...
        config.scope_settings.chshift20x, ...
        config.scope_settings.opt_rgb_intensity, ...
        settings.opt_local_registration, ...
        settings.opt_subsample_rate, ...
        config.scope_settings.opt_max_thresh ...
        );
end

% Organize hyb files if not automated %Heuun/Davide
if config.scope_settings.automation==0
    organize_hyb_files_multi();%move files
end
% denoise hyb files
cd ..
n2v_processing(settings.hyb_n2v_scriptname);%n2v hyb images
cd processed

% register hyb images
register_hyb_images_multi(config.hyb_reg_ch, ...
    config.scope_settings.hyb_radius, ...
    settings.hyb_nuclearch, ...
    settings.hyb_reg_cycle, ...
    config.scope_settings.hyb_regchradius, ...
    config.scope_settings.chprofilehyb, ...
    config.scope_settings.chshifthyb, ...
    config.scope_settings.hyb_rgb_intensity);

% Stitch images from first cycle 
stitch_10x_images_mist(settings.stitch_fname, ...
    settings.stitch_refch, ...
    config.scope_settings.stitch_overlap, ...
    settings.stitch_rescale_factor, ...
    config.scope_settings.stitch_xcam, ...
    config.scope_settings.stitch_ycam);

% Generate stitched images
checkregistration('geneseq', ...
    '40xto10x.mat', ...
    config.scope_settings.registration_intensity_scaling ...
    );
checkregistration('hyb', ...
    '40xto10x.mat', ...
    config.scope_settings.registration_intensity_scaling ...
    );
if ~isempty(config.codebookoptional_name)
    checkregistration(settings.optseq_cyclename, ...
        '40xto10x.mat', ...
        config.scope_settings.registration_intensity_scaling ...
        );
end


cd ..
%%
% Process barcodes %Takuya/Zeynep
if config.is_barcoded
    % move bcseq files if not automated
    if config.scope_settings.automation==0
        organize_bcseq()
    end
    % denoise bc files
    cmdout=n2v_processing(settings.bc_n2vscriptname);
    cd processed

    %register BC to geneseq
    alignBC2gene(settings.bc_refch, ...
        settings.bc2gene_refch, ...
        settings.bc_name, ...
        settings.bc_gene_name ...
        );

    % Register BC 40x
    register_seq_images_subsample_highres(settings.bc_fname, ...
        config.scope_settings.chprofile20x, ...
        settings.bc_ball_radius, ...
        config.scope_settings.chshift20x, ...
        config.scope_settings.bc_rgb_intensity, ...
        settings.bc_local_registration, ...
        settings.bc_subsample_rate, ...
        config.scope_settings.max_thresh_bc);

    %generate checkregistration files
    checkregistration('bcseq', ...
        '40xto10x.mat', ...
        config.scope_settings.bc_registration_intensity_scaling ...
        );

    cd ..
end
%%
copyfile processed/checkregistration output/checkregistration

%% Process registered images

% basecall geneseq % Ezequiel/Chenyue
cd processed
load('40xto10x.mat','tform40xto10x');
batches=ones(1,numel(tform40xto10x))*config.batch_num;
save('batches.mat','batches');

bardensr_cmdout=run_bardensr(config.use_predefined_thresh, ...
    'geneseq', ...
    'codebook.mat', ...
    'codebookforbardensr.mat', ...
    settings.bardensr_scriptname);
if ~isempty(config.codebookoptional_name)
    bardensr_cmdout1=run_bardensr(config.use_predefined_thresh, ...
        settings.optseq_cyclename, ...
        'codebook_optseq.mat', ...
        'codebookforbardensr_optseq.mat', ...
        settings.bardensr_scriptname);
end

if ~isempty(config.codebookoptional_name)
    import_bardensr_results(batches, ...
        fullfile('..',config.codebook_name), ...
        fullfile('..',config.codebookoptional_name) ...
        );
else
    import_bardensr_results(batches, ...
        fullfile('..',config.codebook_name) ...
        );
end
%%
% basecall hyb cycles and quick check one FOV
load(fullfile('..',config.codebookhyb_name),'codebookhyb');

basecall_hyb(config.hybthresh, ...
    config.hybbgn, ...
    codebookhyb, ...
    settings.hybcall_no_deconv, ...
    settings.hybcall_filter_overlap, ...
    settings.hybcall_relaxed);
folders = get_folders();
if numel(folders)>=settings.hybcall_check_fov
    FOV=settings.hybcall_check_fov;
else
    FOV=round(numel(folders)/2);
end
check_hyb_call(FOV,codebookhyb)
exportgraphics(gcf,fullfile('..','output',['FOV',num2str(FOV),'_hybcall.jpg']));

% Segment cells %Katie/Yihan
cellpose_cmdout=run_cellpose(settings.cellpose_scriptname);
import_cellpose(settings.cellpose_dilation_radius);

% Assign rolonies to cells and to slice cooridnates
assign_gene_rolonies();
rolonies_to_10x();

use_mock=1;
calculate_depth(use_mock); % for backward compatibility, remove in the future
% Alice/Nina
data_fname = organize_processed_data(config.startingsliceidx);

% filter overlapping cells

overlap_boxsize=settings.overlap_cellsize*settings.stitch_rescale_factor; % box size for identifying overlaps
overlap_pixelsize=config.scope_settings.cam_pixel/settings.stitch_rescale_factor; % pixel size in stitched image
filter_overlapping_neurons(data_fname, ...
    overlap_boxsize, ...
    overlap_pixelsize ...
    );
%%

% update batch number and slice number into filt_neurons
filt_neuron_filename='filt_neurons.mat';
update_slice_number(filt_neuron_filename, ...
    config.batch_num, ...
    config.slice_num ...
    );

% at this point, filt_neurons is completed if it's non-barcoded

fprintf('Finished processing mRNA data.\n')

if config.is_barcoded==0
    copyfile('filt_neurons.mat',['../output/',config.dataset_id, '-filt_neurons.mat']);
    %copyfile filt_neurons.mat ../output/filt_neurons.mat
end


%% Basecall barcodes. % Samantha/Magan %Xiaotang/Dalton
if config.is_barcoded==1


    % basecall all bc rolonies
    basecall_barcodes_highres(config.rolthresh, ...
        settings.bccall_gaussrad, ...
        settings.bccall_relaxed, ...
        settings.bccall_bgnrad ...
        ); %including a smaller tophat to improve rolony calling
    % transform barcodes to 10x coordinates
    bc_to_10x();
    organize_bc_rolonies();

    % at this point, bc-rolonies.mat is completed with all bc rolony
    % information.


    % find cells that each barcode rolony is in
    assign_bc2cell(); %%assigning rolonies to individual somas based on cell mask. Useful for rabies, not so much for sindbis.
%%
    % basecall somas as a whole and add soma bc data
    %mem=memory;
    %thread_num=floor(mem.MemAvailableAllArrays/2^30/8);% need ~8GB ram per thread for basecalling somas.
    thread_num=floor(maxcore/2);%need ~8GB per thread for somas
    basecall_somas_xt(thread_num);
    %% 
    add_somabc('filt_neurons.mat');
%%    
    % filter somabc. If a model is provided, use the model, score, signal, and complexity thresholds.
    % If the model is not provided, use the default score and intensity
    % thresholds. This can be easily rerun after whole processing, because
    % all information needed here is stored in filt_neurons.
    if isfield(config,'somabc_model_name')&&~isempty(config.somabc_model_name)
        T=load(config.somabc_model_name,'CVMdl');
        filter_somabc_xt('filt_neurons-bc-somas.mat', ...
            'complexity',config.somabc_complexity_thresh, ... %works well for 15-mer
            'score',config.somabc_score_thresh, ...
            'signal',config.somabc_signal_thresh, ...
            'classifier',T.CVMdl ...
            );
    else
        filter_somabc_xt('filt_neurons-bc-somas.mat', ...
            'complexity',config.somabc_complexity_thresh, ... %works well for 15-mer
            'score',config.somabc_score_thresh, ...
            'signal',config.somabc_signal_thresh ...
            );
    end

    %% add single rolonies to cells
    add_singlebc(config.count_thresh, ...
        config.err_corr_thresh, ...
        'filt_neurons-bc-somas.mat' ...
        );

    % at this point, filt_neurons-bc-somas.mat contains completed
    % filt_neurons for barcoded cells.
    
    % Moving barcode matching to a separate script, since this should be
    % done after pooling data batches.
    % % organize and error correct barcode rolonies
    % data_fname=dir('alldata*.mat');
    % data_fname=sort_nat({data_fname.name});
    % data_fname=data_fname{end};
    % mismatch_thresh=1; % allow this mismatch when matching barcodes
    % organize_bc_data(input.count_thresh, ... % For Sindbis, set this to nan (null in the json file) so the script will skip calling single barcodes in cells. This requires that add_somabc has already run
    %     input.err_corr_thresh, ... % For Sindbis, set input.count_thresh to nan, then this doesn't matter.
    %     data_fname, ...
    %     mismatch_thresh,...
    % 	'filt_neurons-bc-somas.mat', ...
    % 	input.slice_num);
    %     % QC soma barcodes
    %
    copyfile('filt_neurons-bc-somas.mat',['../output/',config.dataset_id, '-filt_neurons-bc-somas.mat']);
    %copyfile filt_neurons-bc-somas.mat ../output/filt_neurons-bc-somas.mat
    copyfile('bc-rolonies.mat',['../output/',config.dataset_id, '-bc-rolonies.mat']);
    %copyfile bc-rolonies.mat ../output/bc-rolonies.mat

    fprintf('Finished processing barcode data.\n');
end

%% Convert RGB to .jpg, zip files for smaller storage size

if settings.zip_files == 1
    convert_RGB_jpg;
    cleanup_files;
end

% log off
diary off

cd ..
%%

function check_fiji_install()
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
end


%% Moved to scope settings. These are potentially scope-specific
% 
% input.scope_settings.gene_rgb_intensity=0.6;
% input.scope_settings.gene_max_thresh=2000;
% input.scope_settings.opt_rgb_intensity=0.6;
% input.scope_settings.opt_max_thresh=2000;
% input.scope_settings.bc_rgb_intensity=0.6;
% input.scope_settings.max_thresh_bc=1000;
% 
% input.scope_settings.hyb_radius=100; %radius for bgn subtraction of all channels.
% input.scope_settings.hyb_regchradius=30; % bgn radius for seq rolony cycle
% input.scope_settings.hyb_rgb_intensity=0.6;
% 
% input.scope_settings.stitch_overlap=0.23;
% input.scope_settings.stitch_xcam=3200;
% input.scope_settings.stitch_ycam=3200;
% input.scope_settings.cam_pixel=0.33;
% 
% input.scope_settings.registration_intensity_scaling=3;
% input.scope_settings.bc_registration_intensity_scaling=3;


%% now stored in settings.json. general settings. 
% 
% % for geneseq registration
% settings.gene_fname='n2vgene';
% settings.gene_ball_radius=6;
% settings.gene_local_registration = 1;
% settings.gene_subsample_rate=4;
% 
% % optseq reg
% settings.opt2opt_refch=5;
% settings.opt2gene_refch=5;
% settings.opt_name='n2voptseq';
% settings.opt_gene_name='n2vgeneseq';
% 
% 
% % optseq reg, use same settings. as geneseq
% settings.optseq_fname='regn2vopt';
% settings.opt_ball_radius=6;
% settings.opt_local_registration=1;
% settings.opt_subsample_rate=4;
% 
% % hyb reg
% settings.hyb_nuclearch=5; %nuclear channel
% settings.hyb_reg_cycle=1; % register to 1st cycle
% 
% 
% % stitch
% settings.stitch_fname='n2vgeneseq01';
% settings.stitch_rescale_factor=0.5; % rescaling images by this
% settings.stitch_refch=4;
% 
% % bc reg
% settings.bc_refch=5;
% settings.bc2gene_refch=5;
% settings.bc_name='n2vbcseq';
% settings.bc_gene_name='n2vgeneseq';
% 
% %bc reg
% settings.bc_fname='regn2vbc';
% settings.bc_ball_radius=100;
% settings.bc_local_registration=1;
% settings.bc_subsample_rate=4;
% 
% 
% % hyb call
% settings.hybcall_relaxed=1;
% settings.hybcall_no_deconv=1;
% settings.hybcall_filter_overlap=0;
% settings.hybcall_check_fov=35;
% 
% % cell pose
% settings.cellpose_dilation_radius=3; % dilate each cell by 3 pixesl to accomodate potential error in registration/localizing rolonies
% 
% % overlap
% settings.overlap_cellsize=10;
% 
% % bc rolony call
% settings.bccall_gaussrad=0;
% settings.bccall_bgnrad=6;
% 
% % soma bc call
% settings.somabc_complexity_thresh=-0.9;
% settings.somabc_score_thresh=0.85;
% settings.somabc_signal_thresh=150;
% 
% 
% % python scripts
% settings.optseq_cyclename='optseq';
% settings.gene_n2v_scriptname='n2vprocessing.py';
% settings.bc_n2vscriptname='n2vprocessing_bc.py';
% settings.hyb_n2v_scriptname='n2vprocessing_hyb.py';
% settings.bardensr_scriptname='bardensrbasecall.py';
% settings.cellpose_scriptname='Cellsegmentation-v065.py';
% 
% % diagnostics
% settings.zip_files = 1; % for diagnostics. If expecting fine tuning at the data processing level, set this to 0 so that files are not zipped at the end.

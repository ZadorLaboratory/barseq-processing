# denoise_n2v.py
#
# before simplification
    
if image_type in ['geneseq','bcseq']:
    model_stem = cp.get('n2v',stem_key)
    logging.debug(f'model_stem={model_stem } basedir={basedir}')
    model_1= N2V(config=None, name=model_name+'G', basedir=basedir)
    model_2= N2V(config=None, name=model_name+'T', basedir=basedir)
    model_3= N2V(config=None, name=model_name+'A', basedir=basedir)
    model_4= N2V(config=None, name=model_name+'C', basedir=basedir)
        
elif image_type == 'hyb':
    model_name = 'n2v_hyb_20230323'
    logging.debug(f'model_stem={model_name} basedir={basedir}')
    model_1= N2V(config=None, name=model_name+'GFP', basedir=basedir)
    model_2= N2V(config=None, name=model_name+'YFP', basedir=basedir)
    model_3= N2V(config=None, name=model_name+'TxRed', basedir=basedir)
    model_4= N2V(config=None, name=model_name+'Cy5', basedir=basedir)
        
else:
    logging.error(f'image_type={image_type} not recognized.')
    

    def _parse_experiment_tiles(self, ddict, cp=None):
        if cp is None:
            cp = get_default_config()
        image_regex = cp.get('barseq' , 'image_regex')
        
        fdict = {}    
        for mode in self.modes:
            # list of lists of files, by cycle, hashed by mode
            fdict[mode] = []
        for mode in self.modes:
            # geneseq
            for d in self.ddict[mode]:
                # geneseq01
                cycledir = f'{self.expdir}/{d}'
                logging.debug(f'listing cycle dir {cycledir}')
                flist = os.listdir(cycledir)
                flist.sort()
                fnlist = []
                for f in flist:
                    dp, base, ext = split_path(f)
                    m = re.search(image_regex, base)
                    if m is not None:
                        fname = f'{d}/{f}'
                        fnlist.append(fname)
                fdict[mode].append(fnlist)
        return fdict
    

def process_denoise(indir, outdir, bse, cp=None):
    '''
    @arg indir  is top-level input directory (with cycle dirs below)
    @arg outdir outdir is top-level out directory (with cycle dirs below)
    @arg bse  bse is BarseqExperiment metadata object with relative file/mode layout

    @return None

    handle de-noising of all modalities, all cycles, all images.
    work is bundled by cycle directory.
    
    general approach to sub-conda environments...
    process = subprocess.Popen(
    "conda run -n ${CONDA_ENV_NAME} python script.py".split(), , stdout=subprocess.PIPE
    )
    output, error = process.communicate()


    '''
    if cp is None:
        cp = get_default_config()
    logging.debug(f'handling cycle types={bse.modes} indir={indir} outdir={outdir}')
    
    # general parameters
    tool = cp.get('denoise','tool')
    conda_env = cp.get('denoise','conda_env')
    
    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    
    script_name = f'denoise_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )

    logging.info(f'tool={tool} conda_env={conda_env} script_path={script_path} outdir={outdir}')
    logging.debug(f'script_name={script_name} script_dir={script_dir}')

    tdict = bse.get_cycleset()  # all modes, all tiles

    # order matters.
    log_arg = ''
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    command_list = []
    
    cdirs = list(tdict.keys())
    logging.debug(f'handling all tiles in {cdirs}')
    for cdir in cdirs:
        sub_outdir = f'{outdir}/{cdir}'
        cmd = ['conda','run',
               '-n', conda_env , 
               'python', script_path,
               log_arg,  
               '--outdir', sub_outdir,                      
               ]        
        for rname in tdict[cdir]:
            infile = f'{indir}/{rname}'
            cmd.append(infile)
        command_list.append(cmd)
        logging.info(f'handled {indir}/{cdir}')
    
    logging.info(f'Creating jobset for {len(command_list)} jobs on {n_jobs} CPUs ')    
    jstack = JobStack()
    jstack.setlist(command_list)
    jset = JobSet( max_processes = n_jobs, jobstack = jstack)
    logging.debug(f'running jobs...')
    jset.runjobs()
    
    logging.info(f'done with denoising...')
    
def make_codebook_bin(pth,
                      num_channels=4,
                      codebook_name='codebookM1all.htna.mat'):
    [folders,_,_,_]=get_folders(pth)
    genecycles=len(sorted(glob.glob(os.path.join(pth,'processed',folders[0],'original',"*geneseq*"))))    
    codebook=scipy.io.loadmat(os.path.join(config_pth,codebook_name))['codebook']
    genes=np.reshape(np.array([str(x[0][0]) for x in codebook]),(np.size(codebook,0),-1))
    codebook1=np.zeros((np.size(codebook,0), genecycles), dtype=str)

    for i in range(np.size(codebook,0)):
        for j in range(genecycles):
            codebook1[i,j]=codebook[i][1][0][j]

    #
    codebook_bin=np.ones(np.shape(codebook1),dtype=np.double)
    codebook_bin=np.reshape(np.array([float(x.replace('G','8').replace('T','4').replace('A','2').replace('C','1')) for y in codebook1 for x in y]),np.shape(codebook1))

    codebook_bin=np.matmul(np.uint8(codebook_bin),2**np.transpose(np.array((np.arange(4*genecycles-4,-1,-4)))))

    codebook_bin=np.array([bin(i)[2:].zfill(genecycles*num_channels) for i in codebook_bin])
    codebook_bin=np.reshape([np.uint8(i) for j in codebook_bin for i in j],(np.size(codebook1,0),genecycles*num_channels))

    co=[[genes[i],codebook_bin[j,:]] for i in range(np.size(genes,0))]
    co=[codebook,co]  
    codebook_bin1=np.reshape(codebook_bin,(np.size(codebook_bin,0),-1,num_channels))
    dump(co, os.path.join(pth,'processed','codebook.joblib'))
    dump(codebook_bin1, os.path.join(pth,'processed','codebookforbardensr.joblib'))
    

def process_stage_tilelist_notemplate(indir, outdir, bse, stage='register', cp=None, force=False):
    '''
    process any stage that handles a list of tiles, writing each to parallel (cycle) output 
    subdirs.
    
    
    @arg indir          Top-level input directory (with cycle dirs below)
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp.
    @arg cp             ConfigParser object to refer to.    
 
    @return None

    handle all images in a related list, with output to parallel folders.   
    
    '''
    if cp is None:
        cp = get_default_config()
    logging.info(f'handling stage={stage} types={bse.modes} indir={indir} outdir={outdir}')

    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )

    current_env = os.environ['CONDA_DEFAULT_ENV']

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )

    logging.info(f'tool={tool} conda_env={conda_env} script_path={script_path} outdir={outdir}')
    logging.debug(f'script_name={script_name} script_dir={script_dir} ')

    # cycle, directory mappings
    ddict = bse.ddict
    # cycle files
    clist = bse.get_cycleset()  # all modes, all tiles
 
    # order matters.
    log_arg = ''
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    command_list = []
    
    for mode in modes:
        logging.info(f'handling mode {mode}')
        n_cmds = 0
        dirlist = bse.ddict[mode]
        tilelist = bse.get_imageset(mode)
               
        # Handle batches by tile index. Define template...
        for i, flist in enumerate( tilelist):
            logging.debug(f'handling mode={mode} tile_index={i} n_images={len(flist)} num_cycles={num_cycles}')

            template_rpath = template_list[i]
            num_files = 0
            if conda_env == current_env :
                logging.debug(f'same envs needed, run direct...')
                cmd = ['python', script_path,
                           log_arg,
                           '--config' , runconfig, 
                            ]
            else:
                logging.debug(f'different envs. user conda run...')
                cmd = ['conda','run',
                           '-n', conda_env , 
                           'python', script_path,
                           log_arg, 
                           '--config' , runconfig ,                            
                           ]            
            cmd.append('--stage')
            cmd.append(f'{stage}')
            cmd.append( '--outdir ' )
            cmd.append( f'{outdir}' )            
                       
            for j, rname in enumerate(flist):
                if j < num_cycles:
                    infile = f'{indir}/{rname}'
                    outfile = f'{outdir}/{rname}'
                    cmd.append(infile)
                    num_files += 1       
            logging.info(f'handled tileset {i}')
        logging.info(f'created {n_cmds} commands for mode={mode}')
    
    if n_cmds > 0:
        logging.info(f'Creating jobset for {len(command_list)} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
    else:
        logging.info(f'All output exits. Skipping.')
    logging.info(f'done with stage={stage}...')
    
    
def regcycle_ski_notemplate(infiles, outdir, stage=None, cp=None ):
    '''
    
    @arg infiles    tiles across cycles
    @arg outdir     TOP-LEVEL out directory
    @arg template   optional file to use as template against infiles.
    @arg cp         ConfigParser object
    @arg stage      stage label in cp


    '''  
    if cp is None:
        cp = get_default_config()    
    if stage is None:
        stage = 'regcycle'

    #template_mode = cp.get(stage, 'template_mode')
    #if template_mode == 'None':
    #    template_mode = None
    #template_source = cp.get(stage, 'template_source')
    num_cycles = int(cp.get(stage, 'num_cycles'))


    subsample_rate = int(cp.get(stage,'subsample_rate'))
    resize_factor = int(cp.get(stage,'resize_factor'))
    block_size = int(cp.get(stage,'block_size')) 
    do_coarse = get_boolean( cp.get(stage, 'do_coarse') )
    #num_initial_channels=int(cp.get(stage,'num_initial_channels'))
    #num_later_channels=int(cp.get(stage,'num_later_channels'))
    # [geneseq]
    # channels=G,T,A,C,DIC
    # [bcseq]
    # channels=G,T,A,C,DIC
    # [hyb]
    # channels=GFP,YFP,TxRed,Cy5
    
    # chmap = {}
    num_channels = int(cp.get(stage,'num_channels'))
    
    logging.info(f'outdir={outdir} stage={stage} template={template}')
    logging.debug(f'num_channels={num_channels} do_coarse={do_coarse} block_size={block_size}')
    logging.debug(f'resize_factor={resize_factor} subsample_rate={subsample_rate}')

    if template is None:
        fixed_file = infiles[0]
    else:
        fixed_file = template

    #fixed=tf.imread( fixed_file, key=range(0,num_initial_channels,1))
    #fixed=tf.imread( fixed_file )
    fixed = read_image( fixed_file )
    fixed_sum = np.double(np.sum(fixed,axis=0))
    fixed_sum = np.divide(fixed_sum,np.max(fixed_sum, axis=None))
    sz=fixed_sum.shape
    b_x=np.floor(sz[0]/block_size)
    b_y=np.floor(sz[1]/block_size)

    for i, infile in enumerate( infiles):
        outfile = outfiles[i]
        (outdir, file) = os.path.split(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            logging.debug(f'made outdir={outdir}')
        logging.info(f'Handling {infile} -> {outfile}')
        (dirpath, base, ext) = split_path(os.path.abspath(infile))        
        (prefix, subdir) = os.path.split(dirpath)

        moving = read_image( infile )
        total_channels = len(moving )
        logging.debug(f'loaded image w/ {total_channels} channels. processing {num_channels} channels.')
        
        moving_sum=np.double(np.sum(moving, axis=0))
        moving_sum=np.divide(moving_sum, np.max(moving_sum, axis=None))

        if b_x*block_size!=sz[0]:
            fixed_sum=fixed_sum[0:b_x*block_size-1,:]
            moving_sum=moving_sum[0:b_x*block_size-1,:]
    
        if b_y*block_size!=sz[1]:
            fixed_sum=fixed_sum[:,0:b_y*block_size-1]
            moving_sum=moving_sum[:,0:b_y*block_size-1]
            
        moving_rescaled=np.uint8(ski.transform.rescale(moving_sum, resize_factor)*255) 
        
        # check if this uint8 needs to be changed as per matlab standard-ng
        fixed_rescaled=np.uint8(ski.transform.rescale(fixed_sum, resize_factor)*255)
        moving_split=view_as_blocks(moving_rescaled, block_shape=( block_size*resize_factor, 
                                                                   block_size*resize_factor))
        fixed_split=view_as_blocks(fixed_rescaled, block_shape=( block_size*resize_factor, 
                                                                 block_size*resize_factor))
        fixed_split_lin=np.reshape(fixed_split,(-1, fixed_split.shape[2], fixed_split.shape[3]))       
        fixed_split_sum=[np.sum(j,axis=None) for i,j in enumerate(fixed_split_lin)]
        idx=np.argsort(fixed_split_sum)[::-1]
        fixed_split_sum=np.reshape(fixed_split_sum,(fixed_split.shape[0],fixed_split.shape[1]))
        moving_split_lin=np.reshape(moving_split,(-1,moving_split.shape[2],moving_split.shape[3]))
        xcorr2=lambda a,b: fftconvolve(a, np.rot90(b,k=2))
        c = np.zeros( (fixed_split_lin.shape[1]*2-1, fixed_split_lin.shape[2]*2-1) )
        
        for i in range( np.int32(np.round(fixed_split_lin.shape[0]/subsample_rate))): # check for int32-ng
            if np.max( fixed_split_lin[idx[i]], axis=None)>0:
                c=c+xcorr2( np.double(fixed_split_lin[idx[i]]), np.double(moving_split_lin[idx[i]]))
                
        shift_yx = np.unravel_index(np.argmax(c), c.shape)
        yoffset = -np.array([(shift_yx[0]+1-fixed_split_lin.shape[1])/resize_factor])
        xoffset = -np.array([(shift_yx[1]+1-fixed_split_lin.shape[2])/resize_factor])
        idx_minxy = np.argmin(np.abs(xoffset) + np.abs(yoffset))
        tform = ski.transform.SimilarityTransform( translation=[xoffset[idx_minxy], yoffset[idx_minxy]])
    
        logging.debug(f'transform calculated for {infile} to {fixed_file} Applying...')
        moving_aligned=np.zeros_like(moving)
        
        for i in range(moving.shape[0]):
            moving_aligned[i,:,:] = np.expand_dims( ski.transform.warp((np.squeeze(moving[i,:,:])),
                                                   tform,
                                                   preserve_range=True),
                                                   0)
        #,output_shape=(moving.shape[1],moving.shape[2])),0)# check if output size specification is necessary -ng
      
        moving_aligned=uint16m(moving_aligned)
        moving_aligned_full=moving_aligned.copy()

        logging.debug(f'done processing {base}.{ext} ')
        logging.info(f'writing to {outfile}')
        write_image(outfile, moving_aligned_full)
        logging.debug(f'done writing {outfile}')
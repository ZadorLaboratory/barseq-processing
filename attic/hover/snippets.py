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
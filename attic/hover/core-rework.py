
class BarseqExperiment():

    def old__init__(self, indir, outdir, cp=None):
        '''
        @arg indir     overall input data directory
        @arg outdir    ovarall output working directory
        @arg cp        experiment config
                
        '''
        self.inputdir = os.path.abspath( os.path.expanduser(indir))
        self.cp = cp
        if cp is None:
            self.cp = get_default_config()
            
        # geneseq, bcseq, hyb ,n order. 
        self.modes = [ x.strip() for x in self.cp.get('experiment','modes').split(',') ]
    
        # create directory dict. 
        # mapping from modality to directory names
        self.ddict = self._parse_experiment_indirs(indir, cp = self.cp)
        
        # ordered lists of files by cycle
        self.cdict = self._parse_experiment_cycles(cp=self.cp)
        # self.tdict = self._parse_experiment_tiles(self.ddict, cp=self.cp)
        
        # tiles grouped by position
        self.pdict = self._parse_experiment_images(cp=self.cp)

        # storage for stage-specific input-output maps
        # save initial?
        #   stagemaps <stagedir> 
        self.stagemaps = {}
        logging.debug('BarseqExperiment metadata object intialized. ')
    
    
    def old_parse_experiment_indirs(self, indir, cp=None):
        '''
        determine input data structure and files. 
        return dict of lists of dirs by cycle    
        '''    
        if cp is None:
            cp = get_default_config()
        
        re_list = []
        pdict = {}
        ddict = {}
        modes = [ x.strip() for x in cp.get('experiment','modes').split(',') ]
        for mode in modes:
            p = re.compile( cp.get( 'barseq',f'{mode}_regex'))
            re_list.append(p)
            pdict[p] = mode 
            ddict[mode] = []
                           
        dlist = os.listdir(indir)
        dlist.sort()
        for d in dlist:
            for p in pdict.keys():
                if p.search(d) is not None:
                    k = pdict[p]
                    ddict[k].append(d)
        return ddict

    def old_parse_experiment_cycles(self, cp=None):
        '''
        make dict of dicts of modes and cycle directory names to all tiles within. 
        
        .cdict = { <mode> ->  list of cycles -> list of relative filepaths
        
        '''
        if cp is None:
            cp = get_default_config()
        image_regex = cp.get('barseq' , 'image_regex')
                
        cdict = {}
        for mode in self.modes:
            cdict[mode] = []  # list of lists
            for d in self.ddict[mode]:
                cyclelist = []
                cycledir = f'{self.expdir}/{d}'
                logging.debug(f'listing cycle dir {cycledir}')
                flist = os.listdir(cycledir)
                flist.sort()
                fnlist = []
                for f in flist:
                    dp, base, ext = split_path(f)
                    m = re.search(image_regex, base)
                    if m is not None:
                        rfile = f'{d}/{base}.{ext}'
                        cyclelist.append(rfile)
                    else:
                        logging.warning(f'file {f} did not pass image regex.')
                cdict[mode].append(cyclelist)
        return cdict





    def parse_experiment_dirs(self, stage=None):
        '''
        determine input directory structure and files. 
        return dict of lists of dirs by cycle    
        
        
        '''           
        re_list = []
        pdict = {}
        ddict = {}
        modes = [ x.strip() for x in self.cp.get('experiment','modes').split(',') ]
        for mode in modes:
            p = re.compile( self.cp.get( 'barseq',f'{mode}_regex'))
            re_list.append(p)
            pdict[p] = mode 
            ddict[mode] = []
        
        if stage is None:
            parse_dir = self.inputdir
        else:
            stagedir = self.cp.get(stage, 'stagedir' )
            parse_dir = os.path.join( self.outputdir, stagedir )
        logging.debug(f'parse directory is {parse_dir}')
        
        dlist = os.listdir(parse_dir)
        dlist.sort()
        for d in dlist:
            for p in pdict.keys():
                if p.search(d) is not None:
                    k = pdict[p]
                    ddict[k].append(d)
        logging.debug(f'directory dict = {ddict}')
        return ddict
   
    def parse_experiment_cycles(self, stage=None):
        '''
        make dict of dicts of modes and cycle directory names to all tiles within. 
        
        .cdict = { <mode> ->  list of cycles -> list of relative filepaths
        
        '''
        if stage is None:
            parse_dir = self.inputdir
            file_regex = self.cp.get( 'barseq' , 'file_regex')
        else:
            stagedir = self.cp.get(stage, 'stagedir' )
            parse_dir = os.path.join( self.outputdir, stagedir )
            file_regex = self.cp.get( stage , 'file_regex')
        logging.debug(f'parse directory is {parse_dir}')
        
        
                
        cdict = {}
        for mode in self.modes:
            cdict[mode] = []  # list of lists
            for d in self.ddict[mode]:
                cyclelist = []
                cycledir = f'{parse_dir}/{d}'
                logging.debug(f'listing cycle dir {cycledir}')
                flist = os.listdir(cycledir)
                flist.sort()
                fnlist = []
                for f in flist:
                    dp, base, ext = split_path(f)
                    m = re.search(file_regex, base)
                    if m is not None:
                        rfile = f'{d}/{base}.{ext}'
                        cyclelist.append(rfile)
                    else:
                        logging.warning(f'file {f} did not pass image regex.')
                cdict[mode].append(cyclelist)
        return cdict

    def parse_experiment_files(self, stage=None ):
        '''
        sets of files, grouped by position 
        '''
        if stage is None:
            parse_dir = self.inputdir
            file_regex = self.cp.get( 'barseq' , 'file_regex')
        else:
            stagedir = self.cp.get(stage, 'stagedir' )
            parse_dir = os.path.join( self.outputdir, stagedir )
            file_regex = self.cp.get( stage , 'file_regex')
        logging.debug(f'parse directory is {parse_dir} file_regex={file_regex}')

        pdict = {}
        # 
        # pdict[mode] -> cyclist[0] -> posdict['1'] ->  dok_matrix
        #
        for mode in self.modes:
            pdict[mode] = []
            cycfilelist = self.cdict[mode]
            for i, cycle in enumerate( cycfilelist ):
                logging.debug(f'creating cycle dict for {mode}[{i}]')
                cycdict = {}
                for rfile in cycle:
                    posarray = None
                    afile = os.path.abspath(f'{parse_dir}/{rfile}')
                    dp, base, ext = split_path(afile)
                    logging.debug(f'dp={dp} base={base} ext={ext} for file={afile}')
                    m = re.search(file_regex, base)
                    if m is not None:
                        pos = m.group(1)
                        x = m.group(2)
                        y = m.group(3)
                        x = int(x)
                        y = int(y)
                        logging.debug(f'mode={mode} cycle={i} pos={pos} x={x} y={y} type(pos)={type(pos)}')
                        logging.debug(f'cycdict.keys() = {list( cycdict.keys() )}')
                        pos = str(pos).strip()
                        try:    
                            posarray = cycdict[pos] 
                            logging.debug(f'success. got posarray for cycle[{i}] position {pos}')
                        
                        except KeyError:
                            logging.debug(f'KeyError: creating new position dict for {pos} type(pos)={type(pos)}')
                            #cycdict[pos] = lil_matrix( (50,50), dtype='S128' )
                            #cycdict[pos] = coo_matrix( (50,50), dtype='S128' )
                            cycdict[pos] = SimpleMatrix()
                            logging.debug(f'type = {type( cycdict[pos]) }')
                              
                        fname = f'{rfile}'
                        logging.debug(f"saving posarray[{x},{y}] = '{rfile}'")                            
                        cycdict[pos][x,y] = fname 
                    else:
                        logging.warning(f'File {afile} fails regex match.')
                pdict[mode].append(cycdict)
                
            logging.debug(f'fixing sparse matrices...')
            for i, cycdict in enumerate( pdict[mode]):
                pkeys = list(cycdict.keys())
                pkeys.sort()
                for p in pkeys:
                    sm = cycdict[p]
                    logging.debug(f"fixing sarray {mode} cycle[{i}] position '{p}' type={type(sm)} ")
                    #pnew = self._fix_sparse(sarray)
                    pnew = sm.to_ndarray()
                    logging.debug(f"pnew type={type(pnew)} ")
                    cycdict[p] = pnew
        return pdict



       
    def _parse_experiment_images(self, cp=None):
        '''
        sets of files, grouped by position 
        '''
        if cp is None:
            cp = get_default_config()
        image_regex = cp.get('barseq' , 'image_regex')        
        logging.debug(f'image_regex={image_regex}')
        pdict = {}
        # 
        # pdict[mode] -> cyclist[0] -> posdict['1'] ->  dok_matrix
        #
        for mode in self.modes:
            pdict[mode] = []
            cycfilelist = self.cdict[mode]
            for i, cycle in enumerate( cycfilelist ):
                logging.debug(f'creating cycle dict for {mode}[{i}]')
                cycdict = {}
                for rfile in cycle:
                    posarray = None
                    afile = os.path.abspath(f'{self.expdir}/{rfile}')
                    dp, base, ext = split_path(afile)
                    logging.debug(f'dp={dp} base={base} ext={ext} for file={afile}')
                    m = re.search(image_regex, base)
                    if m is not None:
                        pos = m.group(1)
                        x = m.group(2)
                        y = m.group(3)
                        x = int(x)
                        y = int(y)
                        logging.debug(f'mode={mode} cycle={i} pos={pos} x={x} y={y} type(pos)={type(pos)}')
                        logging.debug(f'cycdict.keys() = {list( cycdict.keys() )}')
                        pos = str(pos).strip()
                        try:    
                            posarray = cycdict[pos] 
                            logging.debug(f'success. got posarray for cycle[{i}] position {pos}')
                        
                        except KeyError:
                            logging.debug(f'KeyError: creating new position dict for {pos} type(pos)={type(pos)}')
                            #cycdict[pos] = lil_matrix( (50,50), dtype='S128' )
                            #cycdict[pos] = coo_matrix( (50,50), dtype='S128' )
                            cycdict[pos] = SimpleMatrix()
                            logging.debug(f'type = {type( cycdict[pos]) }')
                              
                        fname = f'{rfile}'
                        logging.debug(f"saving posarray[{x},{y}] = '{rfile}'")                            
                        cycdict[pos][x,y] = fname 
                    else:
                        logging.warning(f'File {afile} fails regex match.')
                pdict[mode].append(cycdict)
                
            logging.debug(f'fixing sparse matrices...')
            for i, cycdict in enumerate( pdict[mode]):
                pkeys = list(cycdict.keys())
                pkeys.sort()
                for p in pkeys:
                    sm = cycdict[p]
                    logging.debug(f"fixing sarray {mode} cycle[{i}] position '{p}' type={type(sm)} ")
                    #pnew = self._fix_sparse(sarray)
                    pnew = sm.to_ndarray()
                    logging.debug(f"pnew type={type(pnew)} ")
                    cycdict[p] = pnew
        return pdict


    def get_imagelist(self, mode=None, chunksize=None):
        ''' 
        returns FLAT list of ALL image files across ALL cycles for mode(s)
        '''
        tlist = []
        if mode is None:
            modes = self.modes
        else:
            modes = [mode]
            
        for m in modes:
            for cyc in self.pdict[m]:
                for p in list( cyc.keys()):
                    for t in cyc[p].flatten():
                        try:
                            t = t.decode('UTF-8')
                        except:
                            t = str(t)
                        tlist.append(t)
        return tlist     
    
#def process_stage_allfiles_map(new_indir, outdir, bse, stage='denoise-geneseq', cp=cp)
def process_stage_allfiles(indir, outdir, bse, stage='background', cp=None, force=False):
    '''
    process any stage that acts on all images singly, 
    by default, batched by cycle directory as arbitrary load balancing.
     
    @arg indir    is top-level input directory (with cycle dirs below)
    @arg outdir   outdir is top-level out directory (with cycle dirs below)
    @arg stage    which pipeline stage should be executed.  
    @arg bse      bse is BarseqExperiment metadata object with relative file/mode layout

    @return None
    handle all images in all modes in parallel to outdir. 
    
    '''
    if cp is None:
        cp = get_default_config()
    logging.info(f'handling stage={stage} indir={indir} outdir={outdir}')
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )

    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)

    current_env = os.environ['CONDA_DEFAULT_ENV']

    logging.info(f'current_env = {current_env} tool={tool} conda_env={conda_env} script_path={script_path} outdir={outdir}')
    logging.debug(f'script_name={script_name} script_dir={script_dir}')

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
        
        filelist = bse.get_

        # handle batches of cycle directories...
        for i, dirname in enumerate(dirlist):
            sub_outdir = f'{outdir}/{dirname}'
            logging.debug(f'handling mode={mode} dirname={dirname} sub_outdir={sub_outdir}')
            num_files = 0
            flist = cyclist[i]

            if conda_env == current_env :
                logging.debug(f'same envs needed, run direct...')
                cmd = ['python', script_path,
                           log_arg,
                           '--config' , runconfig ,                            
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
            cmd.append( f'{sub_outdir}')                
            for rname in flist:
                infile = f'{indir}/{rname}'
                outfile = f'{outdir}/{rname}'
                if not os.path.exists(outfile):
                    cmd.append(infile)
                    num_files += 1
                else:
                    logging.debug(f'outfile {outfile} exists. omitting file.')
            if num_files > 0:
                command_list.append(cmd)
                n_cmds += 1
            else:
                logging.debug(f'all outfiles exist. omitting command.')
            
            logging.info(f'handled {indir}/{dirname}')
        logging.info(f'created {n_cmds} commands for mode={mode}')
    if n_cmds > 0:
        logging.info(f'Creating jobset for {len(command_list)} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
    else:
        logging.info(f'All output exists. Skipping.')
    logging.info(f'done with stage={stage}...')
    
    
def process_stage_allimages(indir, outdir, bse, stage='background', cp=None, force=False):
    '''
    process any stage that acts on all images singly, batched by cycle directory 
    as arbitrary load balancing.
    NOTE: assumes each input file produces a single output file.   
    
    @arg indir    is top-level input directory (with cycle dirs below)
    @arg outdir   outdir is top-level out directory (with stage dirs below.)
    @arg bse      bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage    which pipeline stage should be executed.  stagdir will be looked up from conf. 

    @return None
    handle all images in all modes in parallel to outdir/stagedir. 
    
    '''
    if cp is None:
        cp = get_default_config()
    logging.info(f'handling stage={stage} indir={indir} outdir={outdir}')
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')
    stagedir = cp.get(stage, 'stagedir')

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )

    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)

    current_env = os.environ['CONDA_DEFAULT_ENV']

    logging.info(f'current_env = {current_env} tool={tool} conda_env={conda_env} script_path={script_path} outdir={outdir}')
    logging.debug(f'script_name={script_name} script_dir={script_dir}')

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
        cyclist = bse.get_cycleset(mode)

        # handle batches of cycle directories...
        for i, dirname in enumerate(dirlist):
            sub_outdir = f'{outdir}/{dirname}'
            logging.debug(f'handling mode={mode} dirname={dirname} sub_outdir={sub_outdir}')
            num_files = 0
            flist = cyclist[i]

            if conda_env == current_env :
                logging.debug(f'same envs needed, run direct...')
                cmd = ['python', script_path,
                           log_arg,
                           '--config' , runconfig ,                            
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
            cmd.append( f'{sub_outdir}')                
            for rname in flist:
                infile = f'{indir}/{rname}'
                outfile = f'{outdir}/{rname}'
                if not os.path.exists(outfile):
                    cmd.append(infile)
                    num_files += 1
                else:
                    logging.debug(f'outfile {outfile} exists. omitting file.')
            if num_files > 0:
                command_list.append(cmd)
                n_cmds += 1
            else:
                logging.debug(f'all outfiles exist. omitting command.')
            
            logging.info(f'handled {indir}/{dirname}')
        logging.info(f'created {n_cmds} commands for mode={mode}')
    if n_cmds > 0:
        logging.info(f'Creating jobset for {len(command_list)} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
    else:
        logging.info(f'All output exists. Skipping.')
    logging.info(f'done with stage={stage}...')


def process_stage_tilelist(indir, outdir, bse, stage='register', cp=None, force=False):
    '''
    process any stage that handles a list of tiles, writing each to parallel output subdirs.
    
    @arg indir          Top-level input directory (with cycle dirs below)
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp.
    @arg cp             ConfigParser object to refer to.    
 
    @return None

    handle all images in a related list, with output to parallel folders.
    Assumes one or more input files to process.
    Optionally allows one template to process input against.     
    
    '''
    if cp is None:
        cp = get_default_config()
    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')
    template_mode = cp.get(stage, 'template_mode')
    if template_mode == 'None':
        template_mode = None
    template_source = cp.get(stage, 'template_source')
    if template_source == 'None':
        template_source = None
    num_cycles = int(cp.get(stage, 'num_cycles'))
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )
    current_env = os.environ['CONDA_DEFAULT_ENV']

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    logging.info(f'handling stage={stage} indir={indir} outdir={outdir} template_mode={template_mode} template_source={template_source} ')
    logging.debug(f'current_env={current_env} tool={tool} conda_env={conda_env} script_dir={script_dir} script_path={script_path} script_name={script_name}')

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
        tilelist = bse.get_tileset(mode)
        
        # Use first cycle as template. 
        if template_mode is not None:    
            template_list = bse.get_cycleset(template_mode)[0]
        else:
            template_list = bse.get_cycleset(mode)[0]
            logging.debug(f'template_list = {template_list}')
        
        # default template source to input directory. 
        template_path = indir
        if template_source == 'input':
            logging.debug(f'template_source={template_source}')
            template_path = indir
        elif template_source == 'output':
            logging.debug(f'template_source={template_source}')
            template_path = outdir
        else:
            logging.warning(f'template_source not specified. defaulting to indir. ')
        
        # Handle batches by tile index. Define template...
        for i, flist in enumerate( tilelist):
            logging.debug(f'handling mode={mode} tile_index={i} n_images={len(flist)} num_cycles={num_cycles}')

            template_rpath = template_list[i]
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
            if template_mode is not None:            
                cmd.append( f'--template')
                cmd.append( f'{template_path}/{template_rpath}')            
            else:
                logging.debug(f'template_mode={template_mode}, omitting --template')
            
            num_files = 0
            for j, rname in enumerate(flist):
                if j < num_cycles:
                    infile = f'{indir}/{rname}'
                    outfile = f'{outdir}/{rname}'
                    if not os.path.exists(outfile):
                        logging.debug(f'outfile {outfile} does not exist. including.') 
                        cmd.append(infile)
                        num_files += 1
                    else:
                        logging.debug(f'outfile {outfile} exists. omitting file.')           
            if num_files > 0:
                command_list.append(cmd)
                cmdstr = ' '.join(cmd)            
                logging.info(f'tileset {i} cmdstr={cmdstr}')
            else:
                logging.info(f'tileset {i} no arguments. skipping command ')    
        n_cmds = len(command_list)
        logging.info(f'created {n_cmds} commands for mode={mode}')
    
    if n_cmds > 0:
        logging.info(f'Creating jobset for {n_cmds} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
    else:
        logging.info(f'All output exits. Skipping.')
    logging.info(f'done with stage={stage}...')
    
def process_stage_positionlist(indir, outdir, bse, stage='stitch', cp=None, force=False):
    '''
    process any stage that handles a list of tiles representing a single position 
    TBD  
        NOTE: optionally allows many-to-many, or many-to-one input to output mapping.     
    
    @arg indir          Top-level input directory (with cycle dirs below)
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp ConfigParser.
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
    num_cycles = int(cp.get(stage, 'num_cycles'))
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
    ddict = bse.ddict
 
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
        positionlists = bse.get_positionset(mode)

        # handle tile sets, batched by position
        for i, poslist in enumerate(positionlists):
            if i < num_cycles:
                (dirname, f) = os.path.split( poslist[0])
                sub_outdir = f'{outdir}/{dirname}'
                logging.debug(f'handling mode={mode} dirname={dirname} sub_outdir={sub_outdir}')
                
                if conda_env == current_env :
                    logging.debug(f'same envs needed, run direct...')
                    cmd = ['python', script_path,
                               log_arg,
                               '--config' , runconfig ,                            
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
                cmd.append( f'{sub_outdir}')               
                for j, rname in enumerate(poslist):
                    infile = f'{indir}/{rname}'
                    cmd.append(infile)
                command_list.append(cmd)
                n_cmds += 1
                logging.info(f'handled {indir}/{dirname}')
            else:
                logging.debug(f'{i} >= {num_cycles}')
        logging.info(f'created {n_cmds} commands for mode={mode}')

    if n_cmds > 0:
        logging.info(f'Creating jobset for {len(command_list)} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
    else:
        logging.info(f'All output exists. Skipping.')
    logging.info(f'done with stage={stage}...')

def process_stage_tilelist_map_works_notemplate(indir, outdir, bse, stage='register', cp=None, force=False):
    '''
    process any stage that handles a list of tiles, following input-output map. 
    
    
    @arg indir          Top-level input directory (with cycle dirs below)
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp.
    @arg cp             ConfigParser object to refer to.    
 
    @return None

    handle all images in a related list, with output to parallel folders.
    Assumes one or more input files to process.
    Optionally allows one template to process input against.     
    
    '''
    logging.info(f'indir={indir}, outdir={outdir} stage={stage} force={force}')
    if cp is None:
        cp = get_default_config()
    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    stagedir = cp.get(stage, 'stagedir')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')
    num_cycles = int(cp.get(stage, 'num_cycles'))

    template_mode = cp.get(stage, 'template_mode')
    if template_mode == 'None':
        template_mode = None
    template_source = cp.get(stage, 'template_source')
    if template_source == 'None':
        template_source = None
    instage = cp.get(stage, 'instage')
    if instage == 'None':
        instage = None

    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )
    current_env = os.environ['CONDA_DEFAULT_ENV']

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    logging.info(f'handling stage={stage} indir={indir} outdir={outdir} template_mode={template_mode} template_source={template_source} ')
    logging.debug(f'current_env={current_env} tool={tool} conda_env={conda_env} script_dir={script_dir} script_path={script_path} script_name={script_name}')

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

        tilelist = bse.get_tileset_map(mode='geneseq', 
                                       stage=stagedir, 
                                       label='spots',
                                       ext='csv',
                                       arity='single',
                                       instage=instage                                       
                                       )
        logging.debug(f'tilelist= {tilelist}')
        
        # Use first cycle as template. 
        if template_mode is not None:    
            template_list = bse.get_cycleset(template_mode)[0]
        else:
            template_list = bse.get_cycleset(mode)[0]
            logging.debug(f'template_list = {template_list}')
        
        # default template source to input directory. 
        #template_path = indir
        #if template_source == 'input':
        #    logging.debug(f'template_source={template_source}')
        #    template_path = indir
        #elif template_source == 'output':
        #    logging.debug(f'template_source={template_source}')
        #    template_path = outdir
        #else:
        #    logging.warning(f'template_source not specified. defaulting to indir. ')
        
        # Handle batches by tile index. Define template...
        for i, fmap in enumerate( tilelist):
            (input_list, output_list) = fmap
    
            logging.debug(f'handling mode={mode} tile_index={i} n_input={len(input_list)} n_output={len(output_list)} num_cycles={num_cycles}')
            logging.info(f'input = {input_list} output = {output_list}')

            #template_rpath = template_list[i]
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
            if len(output_list) > 1:
                cmd.append( '--outdir ' )
                cmd.append( f'{outdir}' )
            elif len(output_list) == 1:
                cmd.append( '--outfile ')
                outfile = os.path.join(outdir, stagedir, output_list[0]   )
                cmd.append( outfile )
            
            if template_mode is not None:            
                cmd.append( f'--template')
                cmd.append( f'{template_path}/{template_rpath}')            
            else:
                logging.debug(f'template_mode={template_mode}, omitting --template')

            for rpath in input_list:
                infile = os.path.join( indir , rpath )
                cmd.append( f' {infile} ')

            # Check for ALL output. Any missing re-runs. 
            output_complete = True 
            for rpath in output_list:
                outfile = os.path.join(outdir, stagedir, rpath )
                output_complete = os.path.exists(outfile) and output_complete
            if not output_complete:
                command_list.append(cmd)
                cmdstr = ' '.join(cmd)            
                logging.info(f'tileset {i} cmdstr={cmdstr}')
            else:
                logging.info(f'tileset {i} output complete. skipping command ')    
        n_cmds = len(command_list)
        logging.info(f'created {n_cmds} commands for mode={mode}')
    
    if n_cmds > 0:
        logging.info(f'Creating jobset for {n_cmds} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
    else:
        logging.info(f'All output exits. Skipping.')
    logging.info(f'done with stage={stage}...')
    
    

    def parse_stage_cycles(self, indir, ddict, stage='basecall', cp=None):
        '''
        make dict of dicts of modes and cycle directory names to all files within. 
        
        .cdict = { <mode> ->  list of cycles -> list of relative filepaths
        
        '''
        if cp is None:
            cp = get_default_config()
                
        cdict = {}
        for mode in self.modes:
            cdict[mode] = []  # list of lists
            for d in ddict[mode]:
                cyclelist = []
                cycledir = f'{indir}/{stage}/{d}'
                logging.debug(f'listing cycle dir {cycledir}')
                flist = os.listdir(cycledir)
                flist.sort()
                fnlist = []
                for f in flist:
                    dp, base, label, ext = split_path(f)
                    rfile = f'{d}/{base}.{ext}'
                    cyclelist.append(rfile)
                cdict[mode].append(cyclelist)
        return cdict    

    def parse_stage_files(self, indir, cdict, stage='basecall', cp=None):
        '''
        sets of files, grouped by position 
        '''
        if cp is None:
            cp = get_default_config()
        image_regex = cp.get('barseq' , 'image_regex')        
        logging.debug(f'image_regex={image_regex}')
        
        
        pdict = {}
        # 
        # pdict[mode] -> cyclist[0] -> posdict['1'] ->  dok_matrix
        #
        for mode in self.modes:
            pdict[mode] = []
            cycfilelist = cdict[mode]
            for i, cycle in enumerate( cycfilelist ):
                logging.debug(f'creating cycle dict for {mode}[{i}]')
                cycdict = {}
                for rfile in cycle:
                    posarray = None
                    afile = os.path.abspath(f'{self.expdir}/{rfile}')
                    dp, base, label, ext = split_path(afile)
                    # Take base as string up to first dot. 
                    base = base.split('.', 1)[0]
                    logging.debug(f'dp={dp} base={base} ext={ext} for file={afile}')
                    m = re.search(image_regex, base)
                    if m is not None:
                        pos = m.group(1)
                        x = m.group(2)
                        y = m.group(3)
                        x = int(x)
                        y = int(y)
                        logging.debug(f'mode={mode} cycle={i} pos={pos} x={x} y={y} type(pos)={type(pos)}')
                        logging.debug(f'cycdict.keys() = {list( cycdict.keys() )}')
                        pos = str(pos).strip()
                        try:    
                            posarray = cycdict[pos] 
                            logging.debug(f'success. got posarray for cycle[{i}] position {pos}')
                        
                        except KeyError:
                            logging.debug(f'KeyError: creating new position dict for {pos} type(pos)={type(pos)}')
                            #cycdict[pos] = lil_matrix( (50,50), dtype='S128' )
                            #cycdict[pos] = coo_matrix( (50,50), dtype='S128' )
                            cycdict[pos] = SimpleMatrix()
                            logging.debug(f'type = {type( cycdict[pos]) }')
                                
                        fname = f'{rfile}'
                        logging.debug(f"saving posarray[{x},{y}] = '{rfile}'")                            
                        cycdict[pos][x,y] = fname 
                    else:
                        logging.warning(f'File {afile} fails a regex match.')
                pdict[mode].append(cycdict)
                
            logging.debug(f'fixing sparse matrices...')
            for i, cycdict in enumerate( pdict[mode]):
                pkeys = list(cycdict.keys())
                pkeys.sort()
                for p in pkeys:
                    sm = cycdict[p]
                    logging.debug(f"fixing sarray {mode} cycle[{i}] position '{p}' type={type(sm)} ")
                    #pnew = self._fix_sparse(sarray)
                    pnew = sm.to_ndarray()
                    logging.debug(f"pnew type={type(pnew)} ")
                    cycdict[p] = pnew
        return pdict
        

    def parse_stage_indirs(self, indir, stage='basecall', cp=None):
        '''
        make a map of a stage output directory, for use
        in generating stage-by-stage processing commands. 
        
        '''    
        if cp is None:
            cp = get_default_config()
        
        re_list = []
        pdict = {}
        ddict = {}
        modes = [ x.strip() for x in cp.get('experiment','modes').split(',') ]
        for mode in modes:
            p = re.compile( cp.get( 'barseq',f'{mode}_regex'))
            re_list.append(p)
            pdict[p] = mode 
            ddict[mode] = []
                            
        stagedir = os.path.join(indir, stage)
        logging.debug(f'scanning stagedir={stagedir}')
        dlist = os.listdir(stagedir)
        dlist.sort()
        for d in dlist:
            for p in pdict.keys():
                if p.search(d) is not None:
                    k = pdict[p]
                    ddict[k].append(d)
        return ddict    
    


def parse_exp_indir(indir, cp=None):
    '''
    determine input data structure and files. 
    return dict of lists of dirs by cycle 
    
     geneseq | bcseq | hyb
    
    As probe types are added, expand this list. 
    
    ddict = { 'geneseq' : 
                [ { 'geneseq01' : ['MAX_Pos1_002_001.tif', 'MAX_Pos1_000_003.tif'] }'
    
    '''    
    if cp is None:
        cp = get_default_config()
        
    bcp = re.compile( cp.get('barseq','bc_regex'))
    gsp = re.compile( cp.get('barseq','gene_regex'))
    hyp = re.compile( cp.get('barseq','hyb_regex'))
    tif = re.compile( cp.get('barseq','tif_regex'))
            
    pdict = { bcp : 'bcseq',
              gsp : 'geneseq',
              hyp : 'hyb'            
             }
        
    ddict = { 'bcseq'   : [],
              'geneseq' : [],
              'hyb'     : []
            } 
    
    dlist = os.listdir(indir)
    dlist.sort()
    for d in dlist:
        for p in pdict.keys():
            if p.search(d) is not None:
                k = pdict[p]
                ddict[k].append(d)
    return ddict


#
#. BSE methods
# 

class BarseqExperimentDummy(self):
    '''
    placeholder.
    '''

    def get_cycleset(self, mode=None, stage=None):
        '''
         get ordered list of cycles where elements are flat lists of relative 
         paths of ALL images in that cycle  
         optionally restrict to single mode.
         if modes must be handled differently, then 
         caller must cycle through modes explicitly                  
        '''
        logging.info(f'mode={mode} stage={stage}')
        clist = []
        if mode is None:
            modes = self.modes
        elif type(mode) == str:
            modes = [mode]
        else:
            modes = mode

        if stage is None:
            cdict = self.cdict
        else:
            (ddict, cdict, pdict) = self.parse_stage_dir( stage )

        for mode in modes:
            for c in cdict[mode]:
                    clist.append(c)
        return clist 

    def get_positionset(self, mode=None, stage=None, cycle=None):
        '''  
        Creates list of lists all tile files grouped by position (slice).
        Each position consists of 1 or more tiles. 
        '''
        logging.info(f'mode={mode}, stage={stage} cycle={cycle} ')               
        positionlist = []

        if mode is None:
            modes = self.modes
        elif type(mode) == str:
            modes = [mode]
        else:
            modes = mode

        if stage is None:
            pdict = self.pdict
        else:
            try:
                (ddict, cdict, pdict) = self.stageinfo[stage]
                logging.debug(f'positionset got cached stageinfo stage={stage}')
            except KeyError:
                logging.debug(f'positionset re-parsing stage dir stage={stage}')
                (ddict, cdict, pdict) = self.parse_stage_dir( stage )

        logging.info(f'positionset: mode={mode} stage={stage} cycle={cycle}')    
        for m in modes:
            logging.debug(f'positionset: handling mode={m}')
            mode_cyc_list = pdict[m]
            logging.debug(f'positionset: handling mode_cyc_list len={len(mode_cyc_list)} type={type(mode_cyc_list)}') 
            for position_dict in mode_cyc_list:
                logging.debug(f'positionset: handling position_dict=\n{position_dict}')
                for pos_key in list( position_dict.keys() ):
                    logging.debug(f'handling pos_key={pos_key}')
                    tlist = []
                    for t in position_dict[pos_key].flatten():
                        #t = t.decode('UTF-8')                       
                        t = str(t)
                        tlist.append(t)
                    positionlist.append(tlist)
        return positionlist


    def get_tileset(self, mode='bcseq', stage=None):
        '''
        Get list of (ordered) lists of images for a single tile across cycles for a single mode.
        If multiple modes are called, the sets span modes. 

        '''
        logging.info(f'mode={mode} stage={stage}')       
        if stage is None:
            cdict = self.cdict
        else:
            (ddict, cdict, pdict) = self.parse_stage_dir( stage )
        
        # use first cycle as template
        # handle multiple modes, or None -> all modes 
        if mode is None:
            mode_list = list(cdict.keys())
        elif type(mode) == list:
            mode_list = mode
        else:
            mode_list = [ mode ]

        logging.debug(f'gathering file list for mode_list={mode_list}')
        tile_list = [ ]
        for file_index in range(0, len(cdict[mode_list[0]][0])):
            file_list = []
            for mode in mode_list:    
                for cyc in cdict[mode]:
                    file_list.append(cyc[file_index])
                tile_list.append(file_list)
        return tile_list

   


    def get_filelist(self, mode=None, stage=None, chunksize=None):
        ''' 
        returns FLAT list of ALL files across ALL cycles for mode(s)
        '''
        logging.info(f'mode={mode} stage={stage}')
        tlist = []
        if mode is None:
            modes = self.modes
        else:
            modes = [mode]
        
        if stage is None:
            pdict = self.pdict
        else:
            try:
                (ddict, cdict, pdict) = self.stageinfo[stage]
            except KeyError:
                (ddict, cdict, pdict) = self.parse_stage_dir( stage)

        for m in modes:
            for cyc in pdict[m]:
                for p in list( cyc.keys()):
                    for t in cyc[p].flatten():
                        try:
                            t = t.decode('UTF-8')
                        except:
                            t = str(t)
                        tlist.append(t)
        return tlist     

    def get_cycleset_map(self,                         
                         mode='bcseq', 
                         stage=None, 
                         label=None, 
                         ext=None, 
                         arity='parallel',
                         instage=None,
                         instage_mode=None,
                         strip_base=False,
                          ):
        '''
        @arg mode       Get map for modality, None means all. 
        @arg stage      Output stage name. 
        @arg label      Output extra label before extension. 
        @arg ext        Output file extension. 
        @arg arity      Arity from input to output. Parallel one-to-one, Single = many-to-one
        @arg instage    Use existing cached stage filemap as input. Initial input default. 
        @arg strip_base Remove base filename from output.        
        
         get ordered list of cycles where elements are flat lists of relative 
         paths of ALL images in that cycle  
         optionally restrict to single mode.
         if modes must be handled differently, then 
         caller must cycle through modes explicitly 
                 
        '''
        logging.info(f'mode={mode} stage={stage} label={label} ext={ext} arity={arity} instage={instage} strip_base={strip_base}')
        if mode is None:
            mode_list = list(self.cdict.keys())
            mode_list.sort()
            mode = mode_list
                
        stagedir = self.cp.get(stage, 'stagedir')
        if instage is None:
            instage_mode = mode
        #cycle_list = self.get_cycleset(mode=instage_mode, stage=instage)
        logging.debug(f"get_stage_files(mode={instage_mode}, stage='{instage}', maptype='cycle')")
        cycle_list = self.get_stage_files(mode=instage_mode, stage=instage, maptype='cycle')
        logging.debug(f'got cycle_list: {cycle_list}')
        
        output_list = []
        output_elem = None 

        if arity == 'parallel':
            # for parallel, each cycle list gets output. 
            for cfset in cycle_list:
                infile_list = []
                outfile_list = []
                for rpath in cfset:
                    infile_list.append(rpath)
                    if (ext is not None) or (label is not None):
                        (subdir, base, current_label, current_ext) =  parse_rpath(rpath)
                        if ext is None:
                            ext = current_ext
                        if label is not None:
                            out_rpath = os.path.join( subdir, f'{base}.{label}.{ext}')
                        else:
                            out_rpath = os.path.join( subdir, f'{base}.{ext}')
                        outfile_list.append( out_rpath )
                    else:
                        outfile_list.append( rpath )
                output_list.append( ( infile_list, outfile_list) )        
                
        elif arity == 'single':
            # Use first input rpath as model for output_rpath
            # Assume mode output dir (not numbered cycle dir)
            # Assume stage mode as correct mode directory
            # Flatten inputs to single input list. 
            infile_list = []
            for cfset in cycle_list:
                for rpath in cfset:
                    infile_list.append(rpath)
            (subdir, base, current_label, current_ext) = parse_rpath( cycle_list[0][0])

            if mode is None:
                mode = self.modes[0]

            if (ext is not None) or (label is not None):
                if ext is None:
                    ext = current_ext
                if label is not None:
                    if strip_base:
                        # stripping base only makes sense if there is a label.
                        # and if arity=single
                        output_elem = os.path.join( mode, f'{label}.{ext}')
                    else:
                        output_elem = os.path.join( mode, f'{base}.{label}.{ext}')
                else:
                    output_elem = os.path.join( mode, f'{base}.{ext}')
            else:
                output_elem = os.path.join( mode, f'{base}.{ext}')
                    
            logging.debug(f'filelist output={( infile_list, output_elem)}')        
            output_list.append( ( infile_list , [output_elem] )  )
        logging.debug(f'made list of {len(output_list)} filemaps')     
        return output_list        


    def get_positionset_map(self, 
                        mode='geneseq', 
                        stage=None, 
                        label=None, 
                        ext=None, 
                        arity='single',
                        instage=None,
                        instage_mode=None,
                        strip_base=False,
                        ) :
        '''
        
        @arg mode       Get map for modality, None means all. 
        @arg stage      Output stage name. 
        @arg label      Output extra label before extension. 
        @arg ext        Output file extension. 
        @arg arity      Arity from input to output. Parallel one-to-one, Single = many-to-one
        @arg instage    Use existing cached stage filemap as input.
        @arg strip_base Remove base filename from output.  
        
        return similar format as get_tileset() except each element is a tuple (input_rpath, output_rpath)
        if arity is 'single' output_rpath is a single item, with leading directory set to <mode>
        
        By default both rpaths the same. 
        
        label=None, ext=None, arity='parallel'
       [
          ( ['geneseq01/MAX_Pos1_003_004.tif', 'geneseq02/MAX_Pos1_003_004.tif'],
            ['geneseq01/MAX_Pos1_003_004.tif', 'geneseq02/MAX_Pos1_003_004.tif']
           ),
          ( ['geneseq01/MAX_Pos1_000_001.tif', 'geneseq02/MAX_Pos1_000_001.tif'],
            ['geneseq01/MAX_Pos1_000_001.tif', 'geneseq02/MAX_Pos1_000_001.tif'] 
          )
        ]

        label='spots, ext='csv, arity='single'
       [
          ( ['geneseq01/MAX_Pos1_003_004.tif', 'geneseq02/MAX_Pos1_003_004.tif'],
             'geneseq/MAX_Pos1_003_004.spots.csv'
           ),
          ( ['geneseq01/MAX_Pos1_000_001.tif', 'geneseq02/MAX_Pos1_000_001.tif'],
            'geneseq/MAX_Pos1_000_001.spots.csv' 
          )
        ]        

        '''
        logging.info(f'mode={mode} stage={stage} label={label} ext={ext} arity={arity} instage={instage} instage_mode={instage_mode} strip_base={strip_base}')
        if mode is None:
            mode_list = list(self.cdict.keys())
            mode_list.sort()
            mode = mode_list

        if instage is None:
            instage_mode = mode
        #positionset_list = self.get_positionset(mode=instage_mode, stage=instage)
        positionset_list = self.get_stage_files(mode=instage_mode, stage=instage, maptype='position')
        output_list = []
        output_elem = None 
        
        for ps in positionset_list:
            if arity == 'parallel':
                output_elem = []
                if (ext is not None) or (label is not None):
                    for rpath in ps:
                        (subdir, base, current_label, current_ext) =  parse_rpath(rpath)
                        if ext is None:
                            ext = current_ext
                        if label is not None:
                            out_rpath = os.path.join(subdir, f'{base}.{label}.{ext}')
                        else:
                            out_rpath = os.path.join(subdir, f'{base}.{ext}')
                        output_elem.append(out_rpath) 
                else:
                    output_elem = ps.copy()
                    
            elif arity == 'single':
                (subdir, base, current_label, current_ext) = parse_rpath( ps[0] )
                if (ext is not None) or (label is not None):
                    if ext is None:
                        ext = current_ext
                    if label is not None:
                        if strip_base:
                            # stripping base only makes sense if there is a label.
                            # and if arity=single
                            output_elem = os.path.join( mode, f'{label}.{ext}')
                        else:
                            output_elem = os.path.join( mode, f'{base}.{label}.{ext}')
                    else:
                        output_elem = os.path.join(mode, f'{base}.{ext}')
                else:
                    output_elem = os.path.join(mode , f'{base}.{ext}')
                    
            logging.debug(f'positionset output={(ps, output_elem)}')        
            output_list.append( (ps, [ output_elem ]) )
        logging.debug(f'made list of {len(output_list)} tilesets.')     
        return output_list

    def get_tileset_map(self, 
                        mode='bcseq', 
                        stage=None, 
                        label=None, 
                        ext=None, 
                        arity='parallel',
                        instage=None,
                        instage_mode=None,
                        strip_base=False,        
                        ) :
        '''
        @arg mode       Get map for modality, None means all. 
        @arg stage      Output stage name. 
        @arg label      Output extra label before extension. 
        @arg ext        Output file extension. 
        @arg arity      Arity from input to output. parallel one-to-one, single = many-to-one
        @arg instage    Use existing cached stage filemap as input. 
        @arg strip_base Remove base filename from output.        
        
        return similar format as get_tileset() except each element is a tuple (input_rpath, output_rpath)
        if arity is 'single' output_rpath is a single item, with leading directory set to <mode>
        
        By default both rpaths the same. 
        
        label=None, ext=None, arity='parallel'
       [
          ( ['geneseq01/MAX_Pos1_003_004.tif', 'geneseq02/MAX_Pos1_003_004.tif'],
            ['geneseq01/MAX_Pos1_003_004.tif', 'geneseq02/MAX_Pos1_003_004.tif']
           ),
          ( ['geneseq01/MAX_Pos1_000_001.tif', 'geneseq02/MAX_Pos1_000_001.tif'],
            ['geneseq01/MAX_Pos1_000_001.tif', 'geneseq02/MAX_Pos1_000_001.tif'] 
          )
        ]

        label='spots, ext='csv, arity='single'
       [
          ( ['geneseq01/MAX_Pos1_003_004.tif', 'geneseq02/MAX_Pos1_003_004.tif'],
             'geneseq/MAX_Pos1_003_004.spots.csv'
           ),
          ( ['geneseq01/MAX_Pos1_000_001.tif', 'geneseq02/MAX_Pos1_000_001.tif'],
            'geneseq/MAX_Pos1_000_001.spots.csv' 
          )
        ]        
        '''
        logging.info(f'mode={mode} stage={stage} label={label} ext={ext} arity={arity} instage={instage} strip_base={strip_base}')
        if mode is None:
            mode_list = list(self.cdict.keys())
            mode_list.sort()
            mode = mode_list

        if instage is None:
            instage_mode = mode
        #tileset_list = self.get_tileset(mode=instage_mode, stage=instage)
        tileset_list = self.get_stage_files(mode=instage_mode, stage=instage, maptype='tileset')

        output_list = []
        output_elem = None 
        
        for ts in tileset_list:
            if arity == 'parallel':
                output_elem = []
                if (ext is not None) or (label is not None):
                    for rpath in ts:
                        (subdir, base, current_label, current_ext) =  parse_rpath(rpath)
                        if ext is None:
                            ext = current_ext
                        if label is not None:
                            out_rpath = os.path.join(subdir, f'{base}.{label}.{ext}')
                        else:
                            out_rpath = os.path.join(subdir, f'{base}.{ext}')
                        output_elem.append(out_rpath) 
                else:
                    output_elem = ts.copy()
                    
            elif arity == 'single':
                # with multi-mode input, raises question of what mode outputs should be. 
                # use whatever the first input mode is. 
                (subdir, base, current_label, current_ext) = parse_rpath( ts[0] )

                if type(mode) == list:
                    out_mode = mode[0]
                else:
                    out_mode = mode
                
                if (ext is not None) or (label is not None):
                    if ext is None:
                        ext = current_ext
                    if label is not None:
                        output_elem = os.path.join(out_mode, f'{base}.{label}.{ext}')
                    else:
                        output_elem = os.path.join(out_mode, f'{base}.{ext}')
                else:
                    output_elem = os.path.join(out_mode , f'{base}.{ext}')
                output_elem = [output_elem]
            logging.debug(f'tileset output={(ts, output_elem)}')        
            output_list.append( (ts, output_elem) )

        logging.debug(f'made list of {len(output_list)} tilesets.')     
        return output_list

    def get_filelist_map(self,                         
                         mode='geneseq', 
                         stage=None, 
                         label=None, 
                         ext=None, 
                         arity='parallel',
                         instage=None,
                         instage_mode=None,
                         strip_base=False):
        '''
        
        @arg mode       Get map for modality, None means all. 
        @arg stage      Output stage name. 
        @arg label      Output extra label before extension. 
        @arg ext        Output file extension. 
        @arg arity      Arity from input to output. Parallel one-to-one, Single = many-to-one
        @arg instage    Use existing cached stage filemap as input. Initial input default. 
        
        To remain consistent with tileset handling, output is still a list of tuples, where 
        each element of the tuple is a list. 
        
        by default, for a flat, 2-element file list with parallel logic we get:
        [  
            ( 
                ['geneseq01/MAX_Pos1_000_000.tif',
                 'geneseq01/MAX_Pos1_000_001.tif',],
                ['geneseq01/MAX_Pos1_000_000.tif',
                 'geneseq01/MAX_Pos1_000_001.tif',]
            )
        ]
                 
        '''
        logging.info(f'mode={mode} stage={stage} label={label} ext={ext} arity={arity} instage={instage} strip_base={strip_base}')
        if mode is None:
            mode_list = list(self.cdict.keys())
            mode_list.sort()
            mode = mode_list
            
        if instage is None:
            instage_mode = mode

        #file_list = self.get_filelist(mode=instage_mode, stage=instage)
        file_list = self.get_stage_files(mode=instage_mode, stage=instage, maptype='filelist')
        
        #
        # list for input-output tuples
        output_list = []
        stagedir = self.cp.get(stage, 'stagedir')
        
        if arity == 'parallel':
            infile_list = []
            outfile_list = []
            for rpath in file_list:
                infile_list.append(rpath)
                if (ext is not None) or (label is not None):
                    (subdir, base, current_label, current_ext) =  parse_rpath(rpath)
                    if ext is None:
                        ext = current_ext
                    if label is not None:
                        out_rpath = os.path.join( subdir, f'{base}.{label}.{ext}')
                    else:
                        out_rpath = os.path.join( subdir, f'{base}.{ext}')
                    outfile_list.append( out_rpath )
                else:
                    outfile_list.append( rpath )
            output_list.append( ( infile_list, outfile_list) )        
                
        elif arity == 'single':
            # Use first input rpath as model for output_rpath
            # Assume mode output dir (not numbered cycle dir)
            (subdir, base, current_label, current_ext) = parse_rpath( file_list[0] )
            if (ext is not None) or (label is not None):
                if ext is None:
                    ext = current_ext
                if label is not None:
                    if strip_base:
                        # stripping base only makes sense if there is a label.
                        # and if arity=single
                        output_elem = os.path.join( mode, f'{label}.{ext}')
                    else:
                        output_elem = os.path.join( mode, f'{base}.{label}.{ext}')
                else:
                    output_elem = os.path.join( mode, f'{base}.{ext}')
            else:
                output_elem = os.path.join( mode, f'{base}.{ext}')
            output_list = [ (file_list, [ output_elem] ) ]        
            logging.debug(f'filelist output={output_list}')        
        logging.debug(f'made list of {len(output_list)} filemaps')     
        return output_list        
    

def process_stage_cycle_map(indir, outdir, bse, stage='stitch', cp=None, force=False):
    '''
    process any stage that handles a list of files per entire cycle, following input-output map. 
    
    @arg indir          Top-level experiment input directory, ignored if instage is not None
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp.
    @arg cp             ConfigParser object to refer to.
    @arg force          Execute even if all outputs exist. 
 
    @return None

    handle all files in a related list, with output to parallel folders.
    Assumes one or more input files to process.     
    
    '''
    logging.info(f'handling stage={stage} indir={indir}, outdir={outdir} force={force}')
    if cp is None:
        cp = get_default_config()
    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    stagedir = cp.get(stage, 'stagedir')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = get_config_list(cp, stage, 'modes' )
    arity = cp.get(stage, 'arity')
    num_cycles = int(cp.get(stage, 'num_cycles'))
    strip_base = cp.getboolean(stage, 'strip_base')
    
    # Potential None params
    instage_dir = None
    label = cp.get(stage, 'label')
    if label == 'None':
        label = None
    ext = cp.get( stage, 'ext')
    if ext == 'None':
        ext = None
    template_mode = cp.get(stage, 'template_mode')
    if template_mode == 'None':
        template_mode = None
    template_source = cp.get(stage, 'template_source')
    if template_source == 'None':
        template_source = None
    instage = cp.get(stage, 'instage')
    if instage == 'None':
        instage = None
    else:
        instage_dir = cp.get(instage, 'stagedir')
    instage_mode = get_config_list(cp, stage, 'instage_modes')
    

    # Misc vars. 
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )
    current_env = os.environ['CONDA_DEFAULT_ENV']

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    logging.info(f'handling stage={stage} indir={indir} outdir={outdir} template_mode={template_mode} template_source={template_source} ')
    logging.debug(f'current_env={current_env} tool={tool} conda_env={conda_env} script_dir={script_dir} script_path={script_path} script_name={script_name}')

    # order matters.
    log_arg = ''
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    command_list = []

    # Handle batches by mode by default
    # Introduce by # of files later.    
    for mode in modes:
        logging.info(f'handling mode {mode}')
        n_cmds = 0
        logging.info(f'get_cycleset_map(mode={mode}, stage={stage}, label={label}, ext={ext}, arity={arity}, instage={instage}, instage_mode={instage_mode}')   
        #file_map = bse.get_cycleset_map(mode=mode, 
        file_map = bse.get_processing_map(mode=mode, 
                                       stage=stage, 
                                       label=label,
                                       ext=ext,
                                       arity=arity, 
                                       instage=instage,
                                       instage_mode=instage_mode,
                                       strip_base = strip_base,
                                       maptype='cycle'
                                       ) 
        logging.debug(f'file_map= {file_map}')
              
        for i, fmap in enumerate( file_map):
            (input_list, output_list) = fmap
            logging.debug(f'handling mode={mode} file_index={i} n_input={len(input_list)} n_output={len(output_list)} num_cycles={num_cycles}')
            logging.debug(f'input = {input_list} output = {output_list}')

            #template_rpath = template_list[i]
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

            if template_mode is not None:            
                cmd.append( f'--template')
                cmd.append( f'{template_path}/{template_rpath}')            
            else:
                logging.debug(f'template_mode={template_mode}, omitting --template')

            # build full paths and check for output. 
            # build infiles/outfiles command arguments
            inlist = []
            outlist = []
            if arity == 'parallel':
                logging.debug(f'arity=parallel output_list length={len(output_list)}')
                for i, fname in enumerate( output_list):
                    logging.debug(f'making outfile outdir={outdir} stagedir={stagedir} fname={fname}')
                    outfile = os.path.join(outdir, stagedir, fname)
                    if not os.path.exists(outfile):
                        outlist.append( outfile )
                        rpath = input_list[i]
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)                        
                    else:
                        logging.debug(f'outfile exists, skipping : {outfile}')    
            if arity == 'single':
                logging.debug(f'arity=single output_list length={len(output_list)}')
                fname = output_list[0]
                outfile = os.path.join(outdir, stagedir, fname)
                if not os.path.exists(outfile):
                    outlist.append( outfile )
                    for rpath in input_list:
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)                        

            cmd.append( '--infiles ')
            for fpath in inlist:
                cmd.append(fpath)

            cmd.append( '--outfiles ')    
            for fpath in outlist:
                cmd.append(fpath)

            if len(outlist) > 0:   
                scmd = ' '.join(cmd)
                logging.debug(f'Adding command: {scmd}')
                command_list.append(cmd)
        n_cmds = len(command_list)
        logging.info(f'created {n_cmds} commands for mode={mode}')
    
    if n_cmds > 0:
        logging.info(f'Creating jobset for {n_cmds} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
        if not jset.all_jobs_succeeded():
            logging.error('Job failure. stage={stage}')
            raise NonZeroReturnException(f'stage={stage}')
        else:
            logging.info(f'All jobs succeeded. stage={stage}')
    else:
        logging.info(f'All output exists. Skipping.')
    logging.info(f'done with stage={stage}...')


def process_stage_position_map(indir, outdir, bse, stage='stitch', cp=None, force=False):
    '''
    process any stage that handles a list of tiles, following input-output map. 
    
    @arg indir          Top-level experiment input directory, ignored if instage is not None
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp.
    @arg cp             ConfigParser object to refer to.
    @arg force          Execute even if all outputs exist. 
 
    @return None

    handle all files in a related list, with output to parallel folders.
    Assumes one or more input files to process.     
    
    '''
    logging.info(f'handling stage={stage} indir={indir}, outdir={outdir} force={force}')
    if cp is None:
        cp = get_default_config()
    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    stagedir = cp.get(stage, 'stagedir')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')
    arity = cp.get(stage, 'arity')
    num_cycles = int(cp.get(stage, 'num_cycles'))
    strip_base = cp.getboolean(stage, 'strip_base')
    
    # Potential None params
    instage_dir = None
    label = cp.get(stage, 'label')
    if label == 'None':
        label = None
    ext = cp.get( stage, 'ext')
    if ext == 'None':
        ext = None
    template_mode = cp.get(stage, 'template_mode')
    if template_mode == 'None':
        template_mode = None
    template_source = cp.get(stage, 'template_source')
    if template_source == 'None':
        template_source = None
    instage = cp.get(stage, 'instage')
    if instage == 'None':
        instage = None
    else:
        instage_dir = cp.get(instage, 'stagedir')
    instage_mode = get_config_list(cp, stage, 'instage_modes')

    # Misc vars. 
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )
    current_env = os.environ['CONDA_DEFAULT_ENV']

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    logging.info(f'handling stage={stage} indir={indir} outdir={outdir} template_mode={template_mode} template_source={template_source} ')
    logging.debug(f'current_env={current_env} tool={tool} conda_env={conda_env} script_dir={script_dir} script_path={script_path} script_name={script_name}')

    # order matters.
    log_arg = ''
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    command_list = []

    # Handle batches by mode by default
    # Introduce by # of files later.    
    for mode in modes:
        logging.info(f'handling mode {mode}')
        n_cmds = 0

        logging.info(f'get_positionset_map(mode={mode}, stage={stage}, label={label}, ext={ext}, arity={arity}, instage={instage}, instage_mode={instage_mode}')       
        #file_map = bse.get_positionset_map(mode=mode, 
        file_map = bse.get_processing_map(mode=mode, 
                                       stage=stage, 
                                       label=label,
                                       ext=ext,
                                       arity=arity, 
                                       instage=instage,
                                       instage_mode=instage_mode,
                                       strip_base = strip_base,
                                       maptype = 'position'
                                       )
        logging.debug(f'file_map= {file_map}')
        
        for i, fmap in enumerate( file_map):
            (input_list, output_list) = fmap
            logging.debug(f'handling mode={mode} file_index={i} n_input={len(input_list)} n_output={len(output_list)} num_cycles={num_cycles}')
            logging.info(f'input = {input_list} output = {output_list}')

            #template_rpath = template_list[i]
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

            if template_mode is not None:            
                cmd.append( f'--template')
                cmd.append( f'{template_path}/{template_rpath}')            
            else:
                logging.debug(f'template_mode={template_mode}, omitting --template')

            # build full paths and check for output. 
            # build infiles/outfiles command arguments
            inlist = []
            outlist = []
            if arity == 'parallel':
                for i, fname in enumerate( output_list):
                    outfile = os.path.join(outdir, stagedir, fname)
                    if not os.path.exists(outfile):
                        outlist.append( outfile )
                        rpath = input_list[i]
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)                        
                    else:
                        logging.debug(f'outfile exists, skipping : {outfile}')    
            if arity == 'single':
                logging.debug(f'arity=single output_list length={len(output_list)}')
                fname = output_list[0]
                outfile = os.path.join(outdir, stagedir, fname)
                if not os.path.exists(outfile):
                    logging.debug(f'outfile {outfile} does not exist.')
                    outlist.append( outfile )
                    for rpath in input_list:
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)                        
                else:
                    logging.debug(f'outfile {outfile} does exist.')

            cmd.append( '--infiles ')
            for fpath in inlist:
                cmd.append(fpath)

            cmd.append( '--outfiles ')    
            for fpath in outlist:
                cmd.append(fpath)

            if len(outlist) > 0:   
                scmd = ' '.join(cmd)
                logging.debug(f'Adding command: {scmd}')
                command_list.append(cmd)
        n_cmds = len(command_list)
        logging.info(f'created {n_cmds} commands for mode={mode}')
    
    if n_cmds > 0:
        logging.info(f'Creating jobset for {n_cmds} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
        if not jset.all_jobs_succeeded():
            logging.error('Job failure. stage={stage}')
            raise NonZeroReturnException(f'stage={stage}')
        else:
            logging.info(f'All jobs succeeded. stage={stage}')
    else:
        logging.info(f'All output exists. Skipping.')

    logging.info(f'done with stage={stage}...')


def process_stage_tileset_map(indir, outdir, bse, stage='register', cp=None, force=False):
    '''
    process any stage that handles a list of tiles, following input-output map. 
    
    
    @arg indir          Top-level input directory (with cycle dirs below)
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp.
    @arg cp             ConfigParser object to refer to.    
 
    @return None

    handle all images in a related list, with output to parallel folders.
    Assumes one or more input files to process.
    Optionally allows one template to process input against.     
    
    '''
    logging.info(f'handling stage={stage}  indir={indir}, outdir={outdir} force={force}')
    if cp is None:
        cp = get_default_config()
    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    stagedir = cp.get(stage, 'stagedir')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')
    modes = [ x.strip() for x in modes]
    num_cycles = int(cp.get(stage, 'num_cycles'))
    arity = cp.get(stage, 'arity')
    strip_base = cp.getboolean(stage, 'strip_base')

    # Potential None params
    instage_dir = None
    label = cp.get(stage, 'label')
    if label == 'None':
        label = None
    ext = cp.get( stage, 'ext')
    if ext == 'None':
        ext = None
    template_mode = cp.get(stage, 'template_mode')
    if template_mode == 'None':
        template_mode = None
    template_source = cp.get(stage, 'template_source')
    if template_source == 'None':
        template_source = None
    instage = cp.get(stage, 'instage')
    if instage == 'None':
        instage = None
    else:
        instage_dir = cp.get(instage, 'stagedir')
    instage_mode = get_config_list(cp, stage, 'instage_modes')

    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )
    current_env = os.environ['CONDA_DEFAULT_ENV']

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    logging.info(f'handling stage={stage} indir={indir} outdir={outdir} template_mode={template_mode} template_source={template_source} ')
    logging.debug(f'current_env={current_env} tool={tool} conda_env={conda_env} script_dir={script_dir} script_path={script_path} script_name={script_name}')

    # order matters.
    log_arg = ''
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    command_list = []
    mode = modes
    logging.info(f'handling mode {mode}')
    n_cmds = 0
    logging.info(f'get_tileset_map(mode={mode}, stage={stage}, label={label}, ext={ext}, arity={arity}, instage={instage}, instage_mode={instage_mode}')
    #tileset_list = bse.get_tileset_map(mode=mode,
    tileset_list = bse.get_processing_map(mode=mode, 
                                    stage=stage, 
                                    label=label,
                                    ext=ext,
                                    arity=arity, 
                                    instage=instage,
                                    instage_mode=instage_mode,
                                    strip_base = strip_base,
                                     maptype='tileset'
                                    )    
    logging.debug(f'tileset_list= {tileset_list}')
    
    # Define template files, if requested.
    template_tileset_list = None
    template_stagedir = None 
    if template_mode is not None:
        if template_source == 'input':  
            #template_tileset_list = bse.get_tileset(template_mode, stage=instage)
            template_tileset_list = bse.get_stage_files(template_mode, stage=instage, maptype='tileset')
            template_stagedir = cp.get(instage, 'stagedir')
        else:
            #template_tileset_list = bse.get_tileset(template_mode, stage=template_source)
            template_tileset_list = bse.get_stage_files(template_mode, stage=template_source, maptype='tileset')
            template_stagedir = cp.get(template_source, 'stagedir')
        logging.debug(f'template_stagedir={template_stagedir} template_tileset_list = {template_tileset_list}')
            
    # Handle batches by tile index. Define template if needed.
    for i, fmap in enumerate( tileset_list):
        (input_list, output_list) = fmap    
        logging.debug(f'handling mode={mode} tile_index={i} n_input={len(input_list)} n_output={len(output_list)} num_cycles={num_cycles}')
        logging.info(f'input = {input_list} output = {output_list}')

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
        
        if template_mode is not None:            
            cmd.append( f'--template ')
            if template_source == 'input':
                template_file = os.path.join(indir, template_stagedir, template_tileset_list[i][0])
            else:
                template_file = os.path.join(outdir, template_stagedir, template_tileset_list[i][0])
            logging.debug(f'template_file = {template_file}')
            cmd.append( template_file )            
        else:
            logging.debug(f'template_mode={template_mode}, omitting --template')

        # build full paths and check for output. 
        # build infiles/outfiles command arguments
        inlist = []
        outlist = []
        if arity == 'parallel':
            for i, fname in enumerate( output_list):
                if i >= num_cycles:
                    break
                outfile = os.path.join(outdir, stagedir, fname)
                if not os.path.exists(outfile):
                    outlist.append( outfile )
                    rpath = input_list[i]
                    if instage is None:
                        infile = os.path.join(indir, rpath)
                    else:
                        infile = os.path.join(outdir, instage_dir, rpath)
                    inlist.append(infile)                        
                else:
                    logging.debug(f'outfile exists, skipping : {outfile}')    

        if arity == 'single':
            logging.debug(f'arity=single output_list length={len(output_list)}')
            fname = output_list[0]
            outfile = os.path.join(outdir, stagedir, fname)
            if not os.path.exists(outfile):
                outlist.append( outfile )
                for i, rpath in enumerate( input_list):
                    if i >= num_cycles:
                        break
                    if instage is None:
                        infile = os.path.join(indir, rpath)
                    else:
                        infile = os.path.join(outdir, instage_dir, rpath)
                    inlist.append(infile)                        

        cmd.append( '--infiles ')
        for fpath in inlist:
            cmd.append(fpath)

        cmd.append( '--outfiles ')    
        for fpath in outlist:
            cmd.append(fpath)

        if len(outlist) > 0:   
            scmd = ' '.join(cmd)
            logging.debug(f'Adding command: {scmd}')
            command_list.append(cmd)
                            
    n_cmds = len(command_list)
    logging.info(f'created {n_cmds} commands for mode={mode}')
    
    if n_cmds > 0:
        logging.info(f'Creating jobset for {n_cmds} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
        if not jset.all_jobs_succeeded():
            logging.error('Job failure. stage={stage}')
            raise NonZeroReturnException(f'stage={stage}')
        else:
            logging.info(f'All jobs succeeded. stage={stage}')
    else:
        logging.info(f'All output exits. Skipping.')
    logging.info(f'done with stage={stage}...')


def process_stage_file_map(indir, outdir, bse, stage='denoise-geneseq', cp=None, force=False):
    '''
    process any stage that handles a list of tiles, following input-output map. 
    
    @arg indir          Top-level experiment input directory
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp.
    @arg cp             ConfigParser object to refer to.
    @arg force          Execute even if all outputs exist. 
 
    @return None

    handle all images in a related list, with output to parallel folders.
    Assumes one or more input files to process.
    Optionally allows one template to process input against.     
    
    '''
    logging.info(f'indir={indir}, outdir={outdir} stage={stage} force={force}')
    if cp is None:
        cp = get_default_config()
    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)
    
    # general parameters
    script_base = cp.get(stage, 'script_base')
    stagedir = cp.get(stage, 'stagedir')
    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    modes = cp.get(stage, 'modes').split(',')
    arity = cp.get(stage, 'arity')
    num_cycles = int(cp.get(stage, 'num_cycles'))
    
    # Potential None params
    instage_dir = None
    label = cp.get(stage, 'label')
    if label == 'None':
        label = None
    ext = cp.get( stage, 'ext')
    if ext == 'None':
        ext = None
    template_mode = cp.get(stage, 'template_mode')
    if template_mode == 'None':
        template_mode = None
    template_source = cp.get(stage, 'template_source')
    if template_source == 'None':
        template_source = None
    instage = cp.get(stage, 'instage')
    if instage == 'None':
        instage = None
    else:
        instage_dir = cp.get(instage, 'stagedir')
    instage_mode = get_config_list(cp, stage, 'instage_modes')

    # Misc vars. 
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    log_level = logging.getLogger().getEffectiveLevel()
    outdir = os.path.expanduser( os.path.abspath(outdir) )
    current_env = os.environ['CONDA_DEFAULT_ENV']

    # tool-specific parameters 
    n_jobs = int( cp.get(tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    logging.info(f'handling stage={stage} indir={indir} outdir={outdir} template_mode={template_mode} template_source={template_source} ')
    logging.debug(f'current_env={current_env} tool={tool} conda_env={conda_env} script_dir={script_dir} script_path={script_path} script_name={script_name}')

    # order matters.
    log_arg = ''
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    command_list = []

    # Handle batches by mode by default
    # Introduce by # of files later.    
    for mode in modes:
        logging.info(f'handling mode {mode}')
        n_cmds = 0
        
        #file_map = bse.get_filelist_map(mode=mode,
        file_map = bse.get_processing_map(mode=mode,                                 
                                       stage=stage, 
                                       label=label,
                                       ext=ext,
                                       arity=arity, 
                                       instage=instage,
                                       instage_mode=instage_mode,
                                       strip_base = strip_base,
                                       maptype = 'filelist'
                                       )
        logging.debug(f'file_map= {file_map}')
        
        # Use first cycle as template. 
        #if template_mode is not None:    
        #    template_list = bse.get_cycleset(template_mode)[0]
        #else:
        #    template_list = bse.get_cycleset(mode)[0]
        #    logging.debug(f'template_list = {template_list}')
        
        # default template source to input directory. 
        #template_path = indir
        #if template_source == 'input':
        #    logging.debug(f'template_source={template_source}')
        #    template_path = indir
        #elif template_source == 'output':
        #    logging.debug(f'template_source={template_source}')
        #    template_path = outdir
        #else:
        #    logging.warning(f'template_source not specified. defaulting to indir. ')
        
        for i, fmap in enumerate( file_map):
            (input_list, output_list) = fmap
            logging.debug(f'handling mode={mode} file_index={i} n_input={len(input_list)} n_output={len(output_list)} num_cycles={num_cycles}')
            logging.info(f'input = {input_list} output = {output_list}')

            #template_rpath = template_list[i]
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

            if template_mode is not None:            
                cmd.append( f'--template')
                cmd.append( f'{template_path}/{template_rpath}')            
            else:
                logging.debug(f'template_mode={template_mode}, omitting --template')

            # build full paths and check for output. 
            # build infiles/outfiles command arguments
            inlist = []
            outlist = []
            if arity == 'parallel':
                for i, fname in enumerate( output_list):
                    outfile = os.path.join(outdir, stagedir, fname)
                    if not os.path.exists(outfile):
                        outlist.append( outfile )
                        rpath = input_list[i]
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)                        
                    else:
                        logging.debug(f'outfile exists, skipping : {outfile}')    
            if arity == 'single':
                logging.debug(f'arity=single output_list length={len(output_list)}')
                fname = output_list[0]
                outfile = os.path.join(outdir, stagedir, fname)
                if not os.path.exists(outfile):
                    outlist.append( outfile )
                    for rpath in input_list:
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)                        

            cmd.append( '--infiles ')
            for fpath in inlist:
                cmd.append(fpath)

            cmd.append( '--outfiles ')    
            for fpath in outlist:
                cmd.append(fpath)

            if len(outlist) > 0:   
                scmd = ' '.join(cmd)
                logging.debug(f'Adding command: {scmd}')
                command_list.append(cmd)
        n_cmds = len(command_list)
        logging.info(f'created {n_cmds} commands for mode={mode}')
    
    if n_cmds > 0:
        logging.info(f'Creating jobset for {n_cmds} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.debug(f'running jobs...')
        jset.runjobs()
        time.sleep(2)
        if not jset.all_suceeded:
            logging.error('Job failure. stage={stage}')
            raise NonZeroReturnException(f'stage={stage}')
    else:
        logging.info(f'All output exists. Skipping.')
    logging.info(f'done with stage={stage}...')

def run_workflow_old(indir, outdir=None, expid=None, cp=None, halt=None):
    '''
    CSHL BARseq pipeline invocation

    Top level function to call into sub-steps...
    @arg indir          Top level input directory. Cycle directories below.  
    @arg outdir         Top-level output directory. Stage directories created below.  
    @arg cp             ConfigParser object defining stage and implementation behavior.
    @arg halt           Stop processing when halt stage is reached.  
     
    '''
    if cp is None:
        cp = get_default_config()
        
    expid = cp.get('project','project_id')
    logging.info(f'Processing experiment {expid} directory={indir} to {outdir}')
    bse = BarseqExperiment(indir, outdir, cp)
    logging.debug(f'got BarseqExperiment metadata: {bse}')
    
    # In sequence, perform all pipeline processing steps
    # maptypes are tileset, cycle, position
    try:
        stage_list = get_config_list(cp, 'experiment','stages')
        n_stages = len(stage_list)
        logging.info(f'got stage_list={stage_list}')
        for i, stage in enumerate( stage_list ):
            maptype = cp.get(stage, 'maptype')
            modes = get_config_list(cp, stage, 'modes')
            stage_no = i + 1
            logging.info(f'[ {stage_no}/{n_stages} ] Running stage={stage} maptype={maptype} modes={modes}')
            if maptype == 'position':
                process_stage_position_map(indir, outdir, bse, stage=stage, cp=cp)
            elif maptype == 'cycle':
                process_stage_cycle_map(indir, outdir, bse, stage=stage, cp=cp )
            elif maptype == 'tileset':
                process_stage_tileset_map(indir, outdir, bse, stage=stage, cp=cp)
            elif maptype == 'filelist':
                process_stage_file_map(indir, outdir, bse, stage=stage, cp=cp)
            else:
                logging.error(f'maptype {maptype} not valid. Exitting.')
                sys.exit(1)
            logging.info(f'[ {stage_no}/{n_stages} ] Done stage={stage}')
            if stage == halt:
                logging.info(f'halt stage= {halt}, current stage {stage}, Done.')
                sys.exit(0)
        logging.info(f'Done running workflow.') 

    except Exception as ex:
        logging.error(f'got exception {ex}')
        logging.error(traceback.format_exc(None))

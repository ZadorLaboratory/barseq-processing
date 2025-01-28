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
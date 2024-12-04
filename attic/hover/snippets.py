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
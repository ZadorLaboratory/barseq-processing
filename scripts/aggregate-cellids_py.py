






# NOTEBOOK CODE
def aggregate_cell_ids_gene_hyb(pth,is_bc):
    """
    Global transformation function:
    1. Combines gene and hyb basecalls and assigns original per tile position and cell id to each
    2. Writes the combined rolony properties file
    """
    gene_rol=load(os.path.join(pth,'processed','basecalls.joblib'))
    seg=load(os.path.join(pth,'processed','all_segmentation.joblib'))
    hyb_rol=load(os.path.join(pth,'processed','genehyb.joblib'))
    if is_bc:
        bc_rol=load(os.path.join(pth,'processed','bc.joblib'))
    
    [folders,_,_,_]=get_folders(pth)
    T={}
    Tbc={}
    for i,folder in enumerate(folders):
        print(f'Operating on {folder}')
        t={}
        mask=seg[folder]['dilated_labels']
        coord_xg=gene_rol['lroi_x'][i]
        coord_yg=gene_rol['lroi_y'][i]
        coord_xh=hyb_rol['lroi_x'][i][0]
        coord_yh=hyb_rol['lroi_y'][i][0]
        t['cellid']=assign_rolony_to_cell(mask,coord_xg,coord_yg)
        t['cellidhyb']=assign_rolony_to_cell(mask,coord_xh,coord_yh)
        T[folder]=t
        if is_bc:
            tbc={}
            coord_xb=bc_rol['lroi_x_all'][i][0]
            coord_yb=bc_rol['lroi_y_all'][i][0]
            tbc['cellidbc']=assign_rolony_to_cell(mask,coord_xb,coord_yb)
            Tbc[folder]=tbc
    dump(T,os.path.join(pth,'processed','cell_id.joblib'))
    if is_bc:
        dump(Tbc,os.path.join(pth,'processed','bccellid.joblib'))
    print('ALL ROLONIES ASSIGNED TO CELLS')


def assign_rolony_to_cell(mask,coord_x,coord_y):
    """
    Global transformation function:
    1. Calls get_cellid function if there are rolonies detected in this tile or else assigns empty cell id to this tile
    """
    if len(coord_x):
        cell_id=get_cellid(mask,coord_x,coord_y)
    else:
        cell_id=[] # earlier this was [] and was causing error later
    return cell_id

def get_cellid(mask,coord_x,coord_y):
    """
    Global transformation function:
    1. For any detected rolony-assigns it to a cell
    2. Returns the cell ids for all rolonies in this tile
    """
    coord_xl=[int(np.round(x)) for x in coord_x]
    coord_yl=[int(np.round(x)) for x in coord_y]
    cell_id=mask[coord_xl,coord_yl]
    return cell_id
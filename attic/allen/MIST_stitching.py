#!/usr/bin/env python
#
# Copied over from MIST_stitching.ipynb from Allen
#
#
import os
import numpy as np
import tifffile
import PIL
from PIL import Image, ImageSequence
from ScanImageTiffReader import ScanImageTiffReader, ScanImageTiffReaderContext
import imagej
import pandas as pd
import skimage
import shutil



def get_file_name(path,kind):  
    os.chdir(path)
    files=[]
    for file in os.listdir():
        if file.endswith(kind):
            files.append(file)
    return files

def get_pos(image_file_name):
    start=image_file_name.find('_')+1
    end=image_file_name.find('_',4)
    return image_file_name[start:end]

def get_row(image_file_name):
    start=image_file_name.find('_',10)+1
    end=start+3
    return int(image_file_name[start:end])

def get_col(image_file_name):
    start=image_file_name.find('_',7)+1
    end=start+3
    return int(image_file_name[start:end])

def reverse_both_row_col_name(file,Max_row,Max_col,pos_name):
    for i in range(len(pos_name)):
        batch=[name for name in file if pos_name[i] in name]
        for name in batch:
            new_r=Max_row[i]-get_row(name) #reverse row number
            new_c=Max_col[i]-get_col(name)
            row_str=str(new_r).zfill(3)
            col_str=str(new_c).zfill(3)
            new_name=name[0:name.find('P')]+pos_name[i]+'_'+col_str+'_'+ row_str+'.tif'#name[0:name.find('0')]='MAX_Pos1_'
            src_dir=os.path.join(pos_path,image_folder,name)
            dst_dir=os.path.join(pos_path,image_folder+'_fixedname',new_name)
            shutil.copy(src_dir,dst_dir)

def reverse_row_name(file,Max_row,pos_name):
    for i in range(len(pos_name)):
        batch=[name for name in file if pos_name[i] in name]
        for name in batch:
            new_r=Max_row[i]-get_row(name) #reverse row number
            new_c=get_col(name)
            row_str=str(new_r).zfill(3)
            col_str=str(new_c).zfill(3)
            new_name=name[0:name.find('P')]+pos_name[i]+'_'+col_str+'_'+ row_str+'.tif'#name[0:name.find('0')]='MAX_Pos1_'
            src_dir=os.path.join(pos_path,image_folder,name)
            dst_dir=os.path.join(pos_path,image_folder+'_fixedname',new_name)
            shutil.copy(src_dir,dst_dir)
            
def reverse_col_name(file,Max_row,pos_name):
    for i in range(len(pos_name)):
        batch=[name for name in file if pos_name[i] in name]
        for name in batch:
            new_r=get_row(name) #reverse row number
            new_c=Max_col[i]-get_col(name)
            row_str=str(new_r).zfill(3)
            col_str=str(new_c).zfill(3)
            new_name=name[0:name.find('P')]+pos_name[i]+'_'+col_str+'_'+ row_str+'.tif'#name[0:name.find('0')]='MAX_Pos1_'
            src_dir=os.path.join(pos_path,image_folder,name)
            dst_dir=os.path.join(pos_path,image_folder+'_fixedname',new_name)
            shutil.copy(src_dir,dst_dir)
            
            
def get_global_trans_matrix(pos):
    img_names = list()
    pixel_x_position = list()
    pixel_y_position = list()
    with open(global_positions_filepath, 'r') as fh:
        for line in fh:
            line = line.strip()
            toks = line.split(';')

            # name loading
            fn_tok = toks[0]
            fn = fn_tok.split(':')[1].strip()
            img_names.append(fn)

            # position loading
            pos_tok = toks[2]
            pos_pair = pos_tok.split(':')[1].strip()
            pos_pair = pos_pair.replace(')', '')
            pos_pair = pos_pair.replace('(', '')
            pos_pairs = pos_pair.split(',')
            x = int(pos_pairs[0].strip())
            y = int(pos_pairs[1].strip())
            pixel_x_position.append(x)
            pixel_y_position.append(y)
    d={'img_names':img_names,'pixel_x_position':pixel_x_position,'pixel_y_position':pixel_y_position}
    trans = pd.DataFrame(data=d)
    trans.to_csv(os.path.join(pos_path,target_channel,'stitched_'+pos,pos+'_transformation.csv'), index=False)
    return img_names,pixel_x_position,pixel_y_position


def create_empty_stitched_img(images_dirpath,img_names):
    first_tile = skimage.io.imread(os.path.join(images_dirpath, img_names[0]))
    tile_shape = first_tile.shape
    n_channels = 1
    if len(tile_shape) == 3:
        n_channels = tile_shape[2]
    tile_h = tile_shape[0]
    tile_w = tile_shape[1]

    stitched_img_h = tile_h + np.max(pixel_y_position)
    stitched_img_w = tile_w + np.max(pixel_x_position)
    stitched_img = np.zeros((stitched_img_h, stitched_img_w), dtype=first_tile.dtype)
    return stitched_img,tile_shape,first_tile,tile_h,tile_w


if __name__ == '__main__':

    # presetting
    pos_path=os.path.join("E:\\Allenwork\\Project\\","stitch_image_with_MIST","20230911_post_xenium_ZM")
    image_folder='hyb01'
    stitch_image_folder=image_folder
    
    file=get_file_name(os.path.join(pos_path,image_folder),'.tif')
    channal_number=6
    c_ls=["ch1","ch2","ch3","ch4","ch5","ch6"]
    #file

    # 
    Max_row=[]
    Max_col=[]
    pos=[1]
    pos_name=['Pos'+str(n) for n in pos]
    for i in pos_name:
        n1=max([get_row(name) for name in file if i in name])
        n2=max([get_col(name) for name in file if i in name])
        Max_row.append(n1)
        Max_col.append(n2)

    # rename the image (optional, in case the file name row and col reversed
    ##rename the image (optional, in case the file name row and col reversed )
    os.mkdir(os.path.join(pos_path,image_folder+'_fixedname'))
    reverse_row_name(file,Max_row,pos_name)
    #reverse_col_name(file,Max_col,pos_name)
    #reverse_both_row_col_name(file,Max_row,Max_col,pos_name)
    stitch_image_folder=image_folder+'_fixedname'    

    # Arrange folders (Ignore here, it is just for our pipeline)
    trim=0.02
    file=get_file_name(os.path.join(pos_path,stitch_image_folder),'.tif')
    img=tifffile.imread(os.path.join(pos_path,stitch_image_folder,file[0]))
    Img_w=img.shape[1]
    Img_h=img.shape[1]
    #stitch_image_folder
    
    
    file=get_file_name(os.path.join(pos_path,stitch_image_folder),'.tif')
    pos=[get_pos(f) for f in file]
    pos_ls=np.unique(pos)
    c_ls=["ch1","ch2","ch3","ch4","ch5","ch6"]
    for i in c_ls:
        if not os.path.exists(os.path.join(pos_path,i)):
            os.mkdir(os.path.join(pos_path,i))        
    for i in c_ls:
        for j in pos_ls:
            if not os.path.exists(os.path.join(pos_path,i,j)):
                os.mkdir(os.path.join(pos_path,i,j)) 
            if not os.path.exists(os.path.join(pos_path,i,"stitched_"+j)):
                os.mkdir(os.path.join(pos_path,i,"stitched_"+j))
    pos_index_ls=[]
    for pos_name in pos_ls:
        index=[i for i, x in enumerate(pos) if x == pos_name]
        pos_index_ls.append(index)
    for i in range(len(pos_index_ls)):
        folder=pos_ls[i]
        name=[file[index] for index in pos_index_ls[i]]
        for n in name:
            img=tifffile.imread(os.path.join(pos_path,stitch_image_folder,n))
            img_crop=img[:,int(Img_w*trim):int(Img_w-(Img_w*trim)),int(Img_h*trim):int(Img_h-(Img_h*trim))]
            for j in range(len(img)):
                img_single=Image.fromarray(img_crop[j])
                
                img_single.save(os.path.join(pos_path,c_ls[j],folder,n))
                print(os.path.join(pos_path,c_ls[j],folder,n))
    
    # config with imageJ
    ij=imagej.init('E:\\Allenwork\\Software\\Fiji.app')
    ij.getVersion()

    # stitch with MIST in ImageJ
    for stitch_channal in c_ls:
        for i in pos_ls:
            image_file_name_ls=os.listdir(os.path.join(pos_path,stitch_channal,i))
            image_file_name_ls=os.listdir(os.path.join(pos_path,stitch_channal,i))
            row=[get_row(i) for i in image_file_name_ls]
            col=[get_col(i) for i in image_file_name_ls]
            Height=int(max(row)+1)
            Width=int(max(col)+1)
            Starting_point="Upper Right"
            img_directory=os.path.join(pos_path,stitch_channal,i)
            img_save_directory=os.path.join(pos_path,stitch_channal,"stitched_"+i)
            filenamePattern='Max_'+i+'_\{ccc\}_\{rrr\}.tif'
            img_directory = img_directory.replace('\\','\\\\')
            img_save_directory=img_save_directory.replace('\\','\\\\')
            macro = f'''run("MIST",\
            "gridwidth={Width} \
            gridheight={Height} \
            starttilerow=0 \
            starttilecol=0 \
            imagedir={img_directory} \
            filenamepattern={filenamePattern} \
            filenamepatterntype=ROWCOL \
            gridorigin=UR \
            assemblefrommetadata=false \
            assemblenooverlap=false \
            globalpositionsfile=[] \
            numberingpattern=HORIZONTALCOMBING \
            startrow=0 \
            startcol=0 \
            extentwidth={Width} \
            extentheight={Height} \
            timeslices=0 \
            istimeslicesenabled=false \
            outputpath={img_save_directory} \
            displaystitching=true outputfullimage=true outputmeta=true outputimgpyramid=false \
            blendingmode=AVERAGE blendingalpha=NaN compressionmode=UNCOMPRESSED outfileprefix=img- \
            unit=MICROMETER unitx=1.0 unity=1.0 programtype=JAVA numcputhreads=8 loadfftwplan=true \
            savefftwplan=true fftwplantype=MEASURE fftwlibraryname=libfftw3 \
            fftwlibraryfilename=libfftw3.dll planpath=C:\\\\Fiji.app\\\\lib\\\\fftw\\\\fftPlans \
            fftwlibrarypath=C:\\\\Fiji.app\\\\lib\\\\fftw stagerepeatability=0 \
            horizontaloverlap=20.0 \
            verticaloverlap=20.0 \
            numfftpeaks=0 \
            overlapuncertainty=NaN \
            isusedoubleprecision=false \
            isusebioformats=false \
            issuppressmodelwarningdialog=false \
            isenablecudaexceptions=false \
            translationrefinementmethod=SINGLE_HILL_CLIMB \
            numtranslationrefinementstartpoints=16 \
            headless=false \
            loglevel=MANDATORY \
            debuglevel=NONE");'''
            ij.py.run_macro(macro)    
            
            
    #get the stitch matrix and apply to each channel
    global_positions_filepath=os.path.join(pos_path,stitch_channal,'stitched_Pos1','img-global-positions-0.txt')
    target_channel='ch6'
    images_dirpath=os.path.join(pos_path,target_channel,'Pos1')
    stitch_images_dirpath=os.path.join(pos_path,target_channel,'Pos1')
    stitch_images_savedirpath=os.path.join(pos_path,target_channel,'stitched_Pos1')

    for ch in c_ls:
        for pos in pos_ls:
            global_positions_filepath=os.path.join(pos_path,stitch_channal,'stitched_'+pos,'img-global-positions-0.txt')
            target_channel=ch
            images_dirpath=os.path.join(pos_path,target_channel,pos)
            stitch_images_dirpath=os.path.join(pos_path,target_channel,pos)
            stitch_images_savedirpath=os.path.join(pos_path,target_channel,'stitched_'+pos)
            img_names,pixel_x_position,pixel_y_position=get_global_trans_matrix(pos)
            stitched_img,tile_shape,first_tile,tile_h,tile_w=create_empty_stitched_img(images_dirpath,img_names)
            for i in range(0, len(img_names)):
                fn = img_names[i]
                x = pixel_x_position[i]
                y = pixel_y_position[i]
    
                tile = skimage.io.imread(os.path.join(stitch_images_dirpath, fn))
                if tile.shape != tile_shape:
                    raise RuntimeError('All images must be the same shape. Image {} is {}, expected {}'.format(fn, tile.shape, tile_shape))
                if tile.dtype != first_tile.dtype:
                    raise RuntimeError('Img {} has type: {}, expected {}.'.format(fn, tile.dtype, first_tile.dtype))
    
                stitched_img[y:y+tile_h, x:x+tile_w] = tile
            skimage.io.imsave(os.path.join(stitch_images_savedirpath,'stitched_'+pos+'.tif'),stitched_img,photometric='minisblack')
            
            
    
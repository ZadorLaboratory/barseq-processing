#!/usr/bin/env bash

#dataset_path='/grid/zador/data/Cristian/cellpose/CS19/fun1/'
dataset_path='/grid/zador/data/Cristian/cellpose/CS100/barcodes/'

do_3d_bool=0
flow_threshold=0.5
cellprob_threshold=0
#cell_diameter='None'
cell_diameter=27.02

username='soitu'
server='bamdev1'
memory_per_thread='4G'
no_threads=4
#job_name='cposeCS100'
job_name='test'

script_origin1='/Users/soitu/Desktop/code/funseq/helpers.py'
script_origin2='/Users/soitu/Desktop/code/funseq/cellpose_segmentation.py'
script_destination='/grid/zador/home/soitu/pyscripts/cellpose/'

echo "Uploading scp files to remote server...."
scp $script_origin1 $script_origin2 $username@$server:$script_destination
echo "File uploaded to remote server completed! ;)"

echo "Submitting job..."
ssh $username@$server 'bash -s'  << EOF

qsub -N $job_name
{
#!/bin/bash
#$ -cwd
#$ -l m_mem_free=$memory_per_thread
#$ -pe threads $no_threads
python3 pyscripts/cellpose/cellpose_segmentation.py -data $dataset_path -do_3d_img $do_3d_bool -diameter $cell_diameter -flow_thresh=$flow_threshold -cellprob_thresh=$cellprob_threshold
}
EOF
echo "Job submitted"
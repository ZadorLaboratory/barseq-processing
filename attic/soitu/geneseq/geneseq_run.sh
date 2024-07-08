#!/usr/bin/env bash

username='soitu'
server='bamdev1'

script_origin1='/Users/soitu/Desktop/code/funseq/geneseq_helpers.py'
script_origin2='/Users/soitu/Desktop/code/funseq/geneseq.py'
script_origin3='/Users/soitu/Desktop/code/funseq/helpers.py'
script_origin4='/Users/soitu/Desktop/code/funseq/N2V.py'
script_destination='/grid/zador/home/soitu/pyscripts/barseq_analysis/'
codebook_path_origin='/Users/soitu/Desktop/code/bardensr/helper/*'
codebook_path_destination='/grid/zador/home/soitu/helper_files'

dataset_path='/grid/zador/data_norepl/Cristian/BCM27393/geneseq/'

echo "Uploading scp files to remote server...."
scp $script_origin1 $script_origin2 $script_origin3 $script_origin4 $username@$server:$script_destination
scp -r $codebook_path_origin $username@$server:$codebook_path_destination
echo "File upload to remote server completed! ;)"

job_name='BCM227393'
#output_file=$logs/job_name+'.o'

ssh $username@$server 'bash -s'  << EOF
qsub -N $job_name
{
#!/bin/bash
#$ -cwd
#$ -l m_mem_free=4G
#$ -pe threads 16
python3 pyscripts/barseq_analysis/geneseq.py -lcl_path $dataset_path -codebook_path $codebook_path_destination
}
EOF

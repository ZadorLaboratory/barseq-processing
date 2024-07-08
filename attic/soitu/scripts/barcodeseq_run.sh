#!/usr/bin/env bash



username='soitu'
server='bamdev1'

script_origin1='/Users/soitu/Desktop/code/funseq/barseq_helpers.py'
script_origin2='/Users/soitu/Desktop/code/funseq/barcodeseq.py'
script_origin3='/Users/soitu/Desktop/code/funseq/helpers.py'
script_destination='/grid/zador/home/soitu/pyscripts/barseq_analysis/'

stardist_model_origin='/Users/soitu/Desktop/code/stardist/models2D/'
stardist_model_destination='/grid/zador/home/soitu/pyscripts/models/'

dataset_path='/grid/zador/data/Cristian/CS100R/barseq/'


echo "Uploading scp files to remote server...."
scp $script_origin1 $script_origin2 $script_origin3 $username@$server:$script_destination
scp -r $stardist_model_origin $username@$server:$stardist_model_destination
echo "File upload to remote server completed! ;)"

job_name='cs100'
#output_file=$logs/job_name+'.o'

ssh $username@$server 'bash -s'  << EOF
qsub -N $job_name
{
#!/bin/bash
#$ -cwd
#$ -l m_mem_free=4G
#$ -pe threads 8
python3 pyscripts/barseq_analysis/barcodeseq.py -lcl_path $dataset_path -star_path $stardist_model_destination
}
EOF

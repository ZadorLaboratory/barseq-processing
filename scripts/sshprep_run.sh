#!/bin/bash

remote_server='zadorstorage2.cshl.edu'
remote_username='cristiansoitu'
remote_password='cristians'
memory_per_thread='4G'
no_threads=4
job_name='sshprep'

remote_path='/home/cristiansoitu/totransfer/raw/bcseq_1/A/'
local_path='/grid/zador/data_norepl/Cristian/test/'
downsample_factor=2
overwrite_images=0

server='bamdev1'
username='soitu'


script_origin1='/Users/soitu/Desktop/code/funseq/fun_helpers.py'
script_origin2='/Users/soitu/Desktop/code/funseq/sshprep.py'
script_origin3='/Users/soitu/Desktop/code/funseq/helpers.py'
script_destination='/grid/zador/home/soitu/pyscripts/dataPrep/'

echo "Uploading scp files to remote server...."
scp $script_origin1 $script_origin2 $script_origin3 $username@$server:$script_destination
echo "File uploaded to remote server completed! ;)"

echo "Submitting job..."
ssh $username@$server 'bash -s'  << EOF

qsub -N $job_name
{
#!/bin/bash
#$ -cwd
#$ -l m_mem_free=$memory_per_thread
#$ -pe threads $no_threads
python3 pyscripts/dataPrep/sshprep.py -srv $remote_server -usr $remote_username -pwd $remote_password -rmt_path $remote_path -lcl_path $local_path -down_factor $downsample_factor -overwrite $overwrite_images
}
EOF
echo "Job submitted"
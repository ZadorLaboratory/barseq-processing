# BARseq Processing
This protocol will guide you through setup, configuration, and running that takes max-projection BARseq image data and provides reduced neurons x genes and neurons x barcodes matrices, along with additional information. 

The pipeline consists of a top-level runner script, core code, and a set of scripts that implement the low-level logic of each stage of the pipeline. Initial input is the raw image data organized in subdirectories by modality and cycle. Final output is a set of dataframes. 

## Install software and dependencies
These instructions assume familiarity with running bioinformatics pipelines...

* Clone the barseq-processing software from the repository to the standard location. (This assumes you already have git installed. If not, install it first). 

```
mkdir ~/git
git clone https://github.com/ZadorLaboratory/barseq-processing.git 
```
All the code is currently organized so it is run directly from the git directory (rather than installed). 

* Install Conda. 
[https://docs.conda.io/projects/miniconda/en/latest/index.html](https://docs.conda.io/projects/miniconda/en/latest/index.html)

* Create an environment for the BARseq pipeline.
```
conda env create --file ~/git/barseq-processing/envs/barseq.environment.yaml 
```

* Activate the environment
```
conda activate barseq
```

* For the standard pipeline, create the sub-environments needed to run stages that require specialize software
```
conda env create --file  ~/git/barseq-processing/envs/ashlar.environment.yaml
conda env create --file  ~/git/barseq-processing/envs/bardensr.environment.yaml
conda env create --file  ~/git/barseq-processing/envs/cellpose.environment.yaml
conda env create --file  ~/git/barseq-processing/envs/n2v.environment.yaml
```

* Create a working directory for your experiment, and copy in the default configuration file. We assume that your max projection data is in a separate location, which we will link via symlink.  

```
mkdir ~/project/barseq/BC12345 ; cd ~/project/barseq/BC12345
cp ~/git/barseq/etc/barseq.conf ./BC12345.barseq.conf
ln -s ~/data/barseq/BC12345 
```

## Experiment Data Layout, Initial Configuration
By default, commands in the pipeline will take their defaults from a single configuration file, included in the distribution ~/git/barseq-processing/etc/barseq.conf.



## Running the pipeline 

### run_barseq.py

```
~/git/barseq-processing/scripts/run_barseq.py 
	-v  				            # give verbose output.
	-c  BC12345.barseq.conf         # customized configuration file.   
	-O  BC12345.run1.out  	        # all output to this sub-directory
	./BC12345	                    # BARseq max projection images organized by cycle.  
```

## Auxiliary Utilities

### calib_XYZ.py and qc_XYZ.py

calib_ utilities generate the artifacts and calculate parameters that are needed for the main pipeline to run. Some may be used one time for each microscope, while others may be dataset-specific. 

qc_ utilities perform checks against generated data. 

##  Next Steps


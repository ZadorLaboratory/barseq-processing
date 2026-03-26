# BARseq Processing
This protocol will guide you through setup, configuration, and running that takes max-projection BARseq image data and provides reduced neurons x genes and neurons x barcodes matrices, along with additional information. 

The pipeline consists of a top-level runner script, core code, and a set of scripts that implement the low-level logic of each stage of the pipeline. Initial input is the raw image data organized in subdirectories by modality and cycle. Final output is a set of dataframes. 

## Install software and dependencies
These instructions assume familiarity with running bioinformatics pipelines. They are regularly run and tested on MacOS and Linux.  

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

* For the standard pipeline, create the sub-environments needed to run stages that require specialize software.
```
conda env create --file  ~/git/barseq-processing/envs/ashlar.environment.yaml
conda env create --file  ~/git/barseq-processing/envs/bardensr.environment.yaml
conda env create --file  ~/git/barseq-processing/envs/cellpose.environment.yaml
```
The n2v conda environment may need to be installed manually. See:
```
~/git/barseq-processing/envs/n2v.softenv.txt
``` 

* Create a working directory for your experiment, and copy in the default configuration file. We assume that your max projection data is in a separate location, which we will link via symlink. E.g.

```
mkdir ~/project/barseq/BC12345 ; cd ~/project/barseq/BC12345
cp ~/git/barseq/etc/barseq.conf ./BC12345.barseq.conf
ln -s ~/data/barseq/BC12345 
```

## Experiment Data Layout, Initial Configuration
By default, commands in the pipeline will take their defaults from a single configuration file, included in the distribution ~/git/barseq-processing/etc/barseq.conf.


## Running the standard pipelines 

### run_workflow.py
To run the standard workflows for barseq or geneseq, run the run_workflow.py script pointed at the appropriate configuration file. A typical invocation would redirect logging output to a file, e.g.
```
~/git/barseq-processing/scripts/run_workflow.py -v -c BC12345.geneseq.conf -O BC12345.run1.out  ./BC12345 > run_geneseq.run1.log 2>&1
```
run_workflow.py will get the stages and their order from the configuration file. 


## Customization and non-standard usage

To do a more ad-hoc pipeline, examine the run_barseq.py or run_geneseq.py scripts, which explicitly call the processing steps for each stage.  

### Configuration
To do non-trivial pipeline alterations, or to handle novel input filenames, it is necessary to understand the parameters of the stage configuration variables.

### Experiment


#### Modes


#### Stages
|  option                     |   valid values       |       meaning      |
|  -----------------------    | -------------------- | ------------------ | 
| [regchannels]               |                      |                       |
| modes = geneseq,bcseq,hyb.  |   mode sections      |                           |
| maptype = cycle             | cycle, tileset, position  |  how to group map inputs                     |
| arity= parallel             |                      |                    |
| template_mode = None        |                      |                    |
| template_source = None      |                      |                    |
| num_cycles = 99             |                      |                    | 
| stagedir = regchannels      |                      |                    |
| file_regex = MAX_Pos(\d*)_(\d*)_(\d*) |                      |                    |
| instage = background        |                      |                    |
| instage_modes = geneseq,bcseq,hyb |                      |                    |
| script_base = regchannels   |                       |                    |
| tool = ski                  |                      |                    |
| label = None                |                      |                    |
| ext = None                  |                      |                    |
| strip_base = False          |                      |                    |






## Auxiliary Utilities

### calib_XYZ.py and qc_XYZ.py

calib_ utilities generate the artifacts and calculate parameters that are needed for the main pipeline to run. Some may be used one time for each microscope, while others may be dataset-specific. 

qc_ utilities perform checks against generated data. 

##  Next Steps


# BARseq Processing
This protocol will guide you through setup, configuration, and running that takes max-projection BARseq image data and provides reduced neurons x genes and neurons x barcodes matrices, along with additional information. 

The pipeline consists of a top-level runner script, core code, and a set of scripts that implement the low-level logic of each stage of the pipeline. Initial input is the raw image data organized in subdirectories by modality and cycle. Final output is a set of dataframes. 

## Install software and dependencies
These instructions assume familiarity with running bioinformatics pipelines. They are regularly run and tested on MacOS and Linux.  

* Install Conda. 
[https://docs.conda.io/projects/miniconda/en/latest/index.html](https://docs.conda.io/projects/miniconda/en/latest/index.html)

* Create an environment for the BARseq pipeline framework. 
```
conda env create --file ~/git/barseq-processing/envs/barseq.environment.yaml 
```

This environment includes the barseq-processing framework code. 

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

## Experiment working directory

Create a working directory for your experiment, and copy in the default configuration file. We assume that your max projection input data is in a separate location, which we will link via symlink. E.g.

```
mkdir ~/project/barseq/BC12345 ; cd ~/project/barseq/BC12345
cp ~/git/barseq/etc/barseq.conf ./BC12345.barseq.conf
ln -s ~/data/barseq/BC12345 
```

## Resource directory

Establish a directory for pipeline resources (e.g. microscope channel profiles, shifts, sequence codebooks, etc.) and ensure that it is pointed to in the configuration file. A resource directory can be shared by multiple experiments. 

```
mkdir ~/project/barseq/resource 
cp ~/git/barseq-processing/resource/*  ~/project/barseq/resource
```

Ensure:
```
[DEFAULT]
resource_dir = ~/project/barseq/resource
```

## Experiment Data Layout, Initial Configuration
By default, commands in the pipeline will take their defaults from a single configuration file. Examples are included in the distribution, e.g. ~/git/barseq-processing/etc/geneseq.conf  ~/git/barseq-processing/etc/barseq.conf.

Ensure that experiment-specific labels, directories, and resources exist and are correct. 


## Running the standard pipelines 

### process_workflow
To run the standard workflows for barseq or geneseq, run process_workflow pointed at the appropriate configuration file. A typical invocation would redirect logging output to a file, e.g.
```
process_workflow 
    -v 
    -c BC12345.geneseq.conf 
    -O BC12345.run1.out  
    ./BC12345 > run_geneseq.run1.log 2>&1
    
```
process_workflow will get the stages and their order from the configuration file. 

## Testing and Validation

For geneseq fuctionality, we created a slimmed-down test dataset, YWT011357_4T that can be found at:

https://labshare.cshl.edu/shares/mbseq/barseq/test_data/

This should be used, along with the relevant `geneseq.conf` file in `/etc` and the resources in `/resource` to confirm that your installation works. After running the pipeline, output can be compared to known-correct output in `/misc/validation/YWT011357_4T/`. The exact output values have been confirmed to match those produced by Xiaoyin's original MATLAB analysis code on the same dataset. 

The `positions.txt` files in `<output>/stitch/hyb/` will confirm that ASHLAR stitching is correct.

The `bardensrparams` files in `<output>/basecall/geneseq/` can confirm bardensr is working normally.  

The `cellsbygenes.tsv` and `filt_cellsbygenes.tsv` output will confirm final data aggregation is correct. 


## Customization and non-standard usage

To do a more ad-hoc pipeline, examine the run_barseq.py or run_geneseq.py scripts, which explicitly call the processing steps for each stage.  

### Configuration
To do non-trivial pipeline alterations, or to handle novel input filenames, it is necessary to understand the parameters of the various types of configuration variables. Configs are standard **[section] option=val** formatted files, handled by the standard Python ConfigParser class.   

Currently all sections are in a single config file. As long as the section names do not collide, they can serve different functions.  

### Experiment

#### Modes and Cycles

#### Tools 

#### Stages
|  section/option             |   valid values       |       meaning      				     |
|  -----------------------    | -------------------- | -----------------------------         |  
| [regchannels]               |                      |  stage name       				     |
| modes = geneseq,bcseq,hyb.  |   mode sections      |  this stage's output modes            |
| maptype = cycle             | cycle, tileset, position  |  how to group map inputs         |
| arity= parallel             |   parallel, single   |  many-to-many, many-to-one            |
| template_mode = None        |   a valid mode       |  mode to draw template arg from       |
| template_source = None      |   a valid stage      |  stage to draw template arg from      |
| num_cycles = 99             | usually 1 or all(99) |  number of input cycles to include in map      | 
| stagedir = regchannels      |   arbitrary          |  subdirectory to put output in        |
| file_regex = MAX_Pos(\d*)_(\d*)_(\d*) | regex(s) match filename base   |  ignore stray files, groups allow variable retrieval   |
| instage = background        |   a valid stage      |  stage to draw input from             |
| instage_modes = geneseq,bcseq,hyb |    valid modes |  mode(s) to draw input from           |
| script_base = regchannels   |   arbitrary          |  script name stem                     |
| tool = ski                  |   a valid tool       |  script name = <stem>_<tool>.py       |
| label = None                |   arbitrary          |  inserted output names before extension. <base>.<label>.<ext>   |
| ext = None                  |   arbitrary          |  output extension (if different from input)  |
| strip_base = False          |   True|False         |  output will be only <label>.<ext>    |



## Auxiliary Utilities

### calib_XYZ.py and qc_XYZ.py

calib_ utilities generate the artifacts and calculate parameters that are needed for the main pipeline to run. Some may be used one time for each microscope, while others may be dataset-specific. 

qc_ utilities perform checks against generated data. 

##  Next Steps


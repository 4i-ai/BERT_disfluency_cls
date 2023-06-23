
#Tested with Ubuntu 20.04 and CUDA 11.6

#create a conda environment for python 3.8
conda create --name discls python=3.8
#activate the environment
conda activate discls

#install requirements
conda install pytorch cudatoolkit=11.6 -c pytorch -c conda-forge
conda install numpy transformers 
conda install -c conda-forge datasets


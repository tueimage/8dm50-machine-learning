# Setting up the Python environment

To get started with setting up a Python environment, follow the instructions in the [Essential Skills](https://github.com/tueimage/essential-skills/python-essentials) module.

In addition, you have to install the packages `matplotlib`, `jupyter`, `scikit-learn`, `scipy`, `pandas` and `tensorflow-cpu`. It is recommended to install these packages in a separate Conda environment. A Conda environment is a directory in which you can install files and packages such that their dependencies will not interact with other environments, which is very useful if you develop code for different courses or research projects. If you want to know more about how to use Conda, read the [docs](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

To create a new Conda environment and install the required packages, run the following commands from the Anaconda Prompt application.

````bash
conda create --name 8dm50 python=3.6				# create a new environment called `myenv`
conda activate 8dm50						# activate this environment
conda install matplotlib jupyter scikit-learn scipy pandas tensorflow	spyder # install the required packages
````
Note that you have to activate the `8dm50` environment every time you start working on the assignments. In order to start Jupyter Notebook type `jupyter notebook` in Anadonda Prompt (after activating the `8dm50` environment with `conda activate 8dm50`). It is best if you change directory to the directory containing the code before starting Jupyter Notebook. Similarly, you can start the Spyder integrated development environment by typing `spyder` in the Anaconda Prompt.

# Working with git

For the practicals, you have to work as a group on the same code. Things are further complicated this year because you will likely not be able to meet in person and have to work remotely. While it is not essential, using git can make thing easier. The [Essential Skills](https://github.com/tueimage/essential-skills/blob/master/version-control-with-git.md) module also contains a section on git basics.

````bash
# first, you have to go to https://github.com/tueimage/8dm40-machine-learning and fork
# the 'official' course repository by clicking 'fork' in the top right corner
# of the page

# then, make a local copy of your fork (note that you will have to replace the username mitkovetta with your own)
git clone https://github.com/mitkovetta/8dm40-machine-learning.git

# change the working directory to the repository
cd 8dm40-machine-learning

# add the 'official' course repository as upstream
git remote add upstream https://github.com/tueimage/8dm40-machine-learning.git

# to pull from the forked student version (e.g. to get updates pushed by other group members):
git pull

# to pull from the 'official' course repository (e.g. in case some assignments have been updated):
git pull upstream master
````

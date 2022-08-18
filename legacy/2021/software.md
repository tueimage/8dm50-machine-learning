# Setting up the Python environment

You can download the source code for the course from this repository [here](https://github.com/tueimage/8dm50-machine-learning/archive/master.zip) (or use Git, see below).

To get started with setting up a Python environment, follow the instructions in the Getting Started section of the [Essential Skills](https://github.com/tueimage/essential-skills/blob/master/python-essentials.md) Python module.

In addition, you have to install the packages `matplotlib`, `jupyter`, `scikit-learn`, `scipy`, `pandas`, `tensorflow` and `spyder`. It is recommended to install these packages in a separate Conda environment. A Conda environment is a directory in which you can install files and packages such that their dependencies will not interact with other environments, which is very useful if you develop code for different courses or research projects. If you want to know more about how to use Conda, read the [docs](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

To create a new Conda environment and install the required packages, run the following commands from the Anaconda Prompt application.

````bash
conda create --name 8dm50 python=3.6				# create a new environment called `myenv`
conda activate 8dm50						# activate this environment
conda install matplotlib jupyter scikit-learn scipy pandas tensorflow spyder # install the required packages
````
Note that you have to activate the `8dm50` environment every time you start working on the assignments. In order to start Jupyter Notebook type `jupyter notebook` in Anadonda Prompt (after activating the `8dm50` environment with `conda activate 8dm50`). It is best if you change directory to the directory containing the code before starting Jupyter Notebook. Similarly, you can start the Spyder integrated development environment by typing `spyder` in the Anaconda Prompt.

## Video instructions
We have also prepared a short video that you can use as a guide for this process:

[![](http://img.youtube.com/vi/AxSwTvnwCUU/0.jpg)](http://www.youtube.com/watch?v=AxSwTvnwCUU "")

Note that the `8DM50-Code` folder at the end of the video should be the folder containing the course source code that you have downloaded.


# Working with Git

**IMPORTANT: this part is highly optional. If you do not want to use Git, think it is an added burden and have found a better workflow for your group, feel free to ignore this section.**

For the practicals, you have to work as a group on the same code. Things are further complicated this year because you will likely not be able to meet in person and have to work remotely. While it is not essential, using a version-control system such as Git can make thing easier. The [Essential Skills](https://github.com/tueimage/essential-skills/blob/master/version-control-with-git.md) Git module goes over the basics of using Git.

Here is an example Git workflow for this course:

````bash
# first, you have to go to https://github.com/tueimage/8dm50-machine-learning and fork
# the 'official' course repository by clicking 'fork' in the top right corner of the page

# then, make a local copy of your fork (note that you will have to replace the username mitkovetta with your own)
git clone https://github.com/mitkovetta/8dm50-machine-learning.git

# change the working directory to the repository
cd 8dm50-machine-learning

# add the 'official' course repository as upstream
git remote add upstream https://github.com/tueimage/8dm50-machine-learning.git

# to commit and push changes that you made locally
git commit -a -m 'did some stuff'
git push

# to pull from the forked student version (e.g. to get updates pushed by other group members)
git pull

# to pull from the 'official' course repository (e.g. in case some assignments have been updated)
git pull upstream master
````

Only one person per group should fork the repository, and the other group members can be added as collaborators (instructions [here](https://docs.github.com/en/github/setting-up-and-managing-your-github-user-account/inviting-collaborators-to-a-personal-repository)).

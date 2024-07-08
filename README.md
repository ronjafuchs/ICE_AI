First install poetry (Python environment manager)

Then use `poetry install` to set up the virtual environment

`poetry shell` spawns a new shell inside of the virtual environment

# An up2date installation of Java is necessary, if not installed yet, run
apt install default-jre

# installing poetry on Ubuntu (debian-based systems) 
apt install python3-poetry

# make sure to use python >=3.9 <3.12
# if not available to your system try installing a different python version using pyenv

## if pyenv is not yet installed install prerequisites first
apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl

## install pyenv (if curl is not installed yet install it using "apt install curl")
curl https://pyenv.run | bash

## add pyenv to autoload by running
nano ~/.bashrc

## add the following lines at the end of the file
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

## restart computer or shell
exec "$SHELL"

## install python version 3.11 using pyenv
pyenv install 3.11.9

## make installed pyenv globally available
pyenv global 3.11.9

## let poetry switch to newly installed python version
poetry env use 3.11.9

# navigate to the root folder of this project and run
poetry install

## if the installation gets stuck on "pending" for all packages, try deleting the lock file in your user cache and restart the previous command
find ~/.cache/pypoetry -name '*.lock' -type f -delete

## alternatively it may be because of a wrong keyring setup. try to deactivate it and run poetry install again
poetry config keyring.enabled false

## if poetry fails in installing some packages, run poetry install again to see if it is a problem caused by the order of dependencies

## in Pycharm or console: activate the environment
go to Setting -> Python Interpreter -> Add Interpreter -> Add Local Interpreter -> Poetry Environment -> use existing environment -> select fightingice-ai

# when facing the following errors do:

## libcublas.so.*[0-9] not found in the system path  ...
## -> install cuda
sudo apt-get install nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc
## if this still returns a similar error use the following line to downgrade torch (known issue with poetry)
poetry add torch=2.0.0



Create a virtual environment ending in `_venv` so that it’s ignored by `.gitignore`.

```bash
python3 -m venv openBCI_venv
```

Add the following to `./openBCI_venv/bin/activate`

```bash
export PYTHONPATH=<abosolute_path_to_current_dir>/src:$PYTHONPATH # need this to be able to do `from utils import *`
export ROOT_DIR=<abosolute_path_to_current_dir>
```

Then activate the environment

```bash
source ./openBCI_venv/bin/activate
```

If you want to use debugger in vscode, add the following to `launch.json`

```bash
"env": {
    "ROOT_DIR": "${workspaceFolder}",
    "PYTHONPATH": "${workspaceFolder}/src",
},
```

Install the required package for this project using 

```bash
pip install -r src/requirements.txt
```
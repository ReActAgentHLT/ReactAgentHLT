# ReAct Agent HLT

A _tool using agent_ built on top of the **ReAct** and **focusedReAct**
paradigms. The project focuses on the implementation of a basic agent capable of
using _search and retrieval_ tools to expand its knowledge if necessary and
Python as a calculator to solve simple arithmetic word problems.

## Installation and Usage

To use or develop the package create and activate a Python virtual environment
in the root directory

```bash
python -m venv .venv
source .venv/bin/activate # on Mac and Linux
.venv/Scripts/activate.bat # on Windows
```

and install the package with pip in the virtual environment

```bash
pip install .
# or
pip install -e . # for edit mode (developer)
```

All the modules will be available through the `reactagenthlt` package:

```python
from reactagenthlt.agents import ReActAgent
from reactagenthlt.tools import SearchEngine
```

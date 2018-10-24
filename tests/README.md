
## Idea of testing
Check out [lytest](https://github.com/atait/lytest) for a better readme.

## One time setup
Run these commands:
```
pip install --upgrade lytest
pip install --upgrade pytest
```
If it says something about permissions, use the `--user` flag.

If you want to write tests, I recommend [lyipc](https://github.com/atait/klayout-ipc). `lyipc` is *no longer available on PyPI*. You must get it via the klayout package manager.


## How to run the tests
In terminal,
```
pytest
```
where you can specify a filename optionally.


## Dev and debugging
I use a pretty neat development flow using
[lyipc](https://github.com/atait/klayout-ipc),
[lytest](https://github.com/atait/lytest),
and [ipython](https://ipython.readthedocs.io/en/stable/).

First, activate lyipc server in klayout GUI. Hit Command-I. Read the disclaimer.

There is a trick because we are messing with the test environment so much. You must launch ipython shell with
```
ipython -i conftest.py
```

After that,
```python
[1] %load_ext autoreload
[2] %autoreload 2
[3] from test_layout import *
[4] Basic()
```
will send this layout to the GUI window. When you change the code of `Basic` at all, ipython will autodetect and reload it (usually, otherwise just rerun line [3]).

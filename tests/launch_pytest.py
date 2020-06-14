'''
This file is made to wrap pytest within the klayout GSI
and then launch pytest upon fake_test_python.py

Use it to prove that klayout pya and klayout.db behave the same

You can make testfile an argument with

    klayout -rd testfile=test_layout.py -r launch_pytest.py
'''
import pytest

global testfile
testfile_default = '.'
try:
    testfile
except NameError:
    testfile = testfile_default

pytest.main([testfile])
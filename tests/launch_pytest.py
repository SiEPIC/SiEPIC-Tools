'''
This file is made to wrap pytest within the klayout GSI
and then launch pytest upon fake_test_python.py
'''
import pytest

testfile_default = 'test_0.py'
global testfile
try:
    testfile
except NameError:
    testfile = testfile_default

pytest.main([testfile])
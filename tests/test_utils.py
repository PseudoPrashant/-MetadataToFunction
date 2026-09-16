import sys
import os

# Add src to sys.path so we can import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils import format_metadata

def test_format_metadata():
    desc = "Adds two integers"
    params = "int a int b"
    ret = "int"
    lib = "MathUtils"
    kw = "add sum"
    cnt = 2
    
    result = format_metadata(desc, params, ret, lib, kw, cnt)
    assert result == "Adds two integers int a int b int MathUtils add sum 2"

def test_format_metadata_empty_strings():
    result = format_metadata("", "", "", "", "", 0)
    assert result == "      0"

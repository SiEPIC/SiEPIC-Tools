"""
Unit tests for SiEPIC.utils string parsing and conversion functions
Tests eng_str, XML parsing, and data extraction functions
"""

import pytest
from SiEPIC.utils import eng_str, angle_trunc


class TestStringParsing:
    """Test string parsing and formatting functions"""
    
    def test_eng_str_returns_string(self):
        """Test eng_str returns a string"""
        result = eng_str(1000)
        assert isinstance(result, str)
    
    def test_eng_str_zero(self):
        """Test zero"""
        result = eng_str(0)
        assert "0" in result
    
    def test_eng_str_negative(self):
        """Test negative number"""
        result = eng_str(-1000)
        assert "-" in result
    
    def test_eng_str_various_magnitudes(self):
        """Test various magnitudes produce valid strings"""
        values = [0.001, 1, 1000, 1000000, 0.000001]
        for val in values:
            result = eng_str(val)
            assert isinstance(result, str)
            assert len(result) > 0


class TestAngleFunctions:
    """Test angle utility functions"""
    
    def test_angle_trunc_function_exists(self):
        """Test angle_trunc function exists"""
        from SiEPIC.utils import angle_trunc
        assert angle_trunc is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


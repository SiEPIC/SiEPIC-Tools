"""
Unit tests for SiEPIC.extend module
Tests type conversion and extension functions
"""

import pytest
from SiEPIC.extend import to_itype, to_dtype


class TestTypeConversion:
    """Test type conversion functions"""
    
    def test_to_itype_float(self):
        """Test converting float to integer type"""
        # These functions convert from µm to database units (typically nm)
        result = to_itype(1.5, 0.001)  # 1.5 µm with dbu=1nm
        assert result == 1500  # Should be 1500 nm
    
    def test_to_itype_zero(self):
        """Test converting zero"""
        result = to_itype(0, 0.001)
        assert result == 0
    
    def test_to_itype_negative(self):
        """Test converting negative values"""
        result = to_itype(-2.5, 0.001)
        assert result == -2500
    
    def test_to_dtype_int(self):
        """Test converting integer to float type"""
        result = to_dtype(1000, 0.001)  # 1000 nm with dbu=1nm
        assert result == pytest.approx(1.0, abs=1e-6)  # Should be 1.0 µm
    
    def test_to_dtype_zero(self):
        """Test converting zero"""
        result = to_dtype(0, 0.001)
        assert result == pytest.approx(0, abs=1e-6)
    
    def test_to_dtype_negative(self):
        """Test converting negative values"""
        result = to_dtype(-5000, 0.001)
        assert result == pytest.approx(-5.0, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



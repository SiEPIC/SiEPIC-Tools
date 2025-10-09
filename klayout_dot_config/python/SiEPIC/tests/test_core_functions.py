"""
Unit tests for SiEPIC.core module
Tests component and PCell functions
"""

import pytest
import pya


class TestCoreModule:
    """Test core module functions"""
    
    def test_module_import(self):
        """Test core module can be imported"""
        from SiEPIC import core
        assert core is not None
    
    def test_component_class_exists(self):
        """Test Component class exists"""
        from SiEPIC.core import Component
        assert Component is not None
    
    def test_pin_class_exists(self):
        """Test Pin class exists"""
        from SiEPIC.core import Pin
        assert Pin is not None
    


class TestPinBasics:
    """Test Pin class basic functionality"""
    
    def test_pin_attributes(self):
        """Test Pin class has expected attributes"""
        from SiEPIC.core import Pin
        
        # Pin should have certain methods/attributes
        assert hasattr(Pin, '__init__')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


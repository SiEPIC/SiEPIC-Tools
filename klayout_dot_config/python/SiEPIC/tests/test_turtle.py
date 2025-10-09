"""
Unit tests for Turtle class in SiEPIC.scripts
Tests Manhattan-style path generation
"""

import pytest
import pya
from SiEPIC.scripts import Turtle


class TestTurtleClass:
    """Test the Turtle class for Manhattan paths"""
    
    def test_turtle_creation_left_turn(self):
        """Test creating a Turtle with left turn"""
        turtle = Turtle([10, 90])  # forward 10, turn left 90
        assert turtle.forward == 10
        assert turtle.turn == 90
    
    def test_turtle_creation_right_turn(self):
        """Test creating a Turtle with right turn"""
        turtle = Turtle([5, -90])  # forward 5, turn right 90
        assert turtle.forward == 5
        assert turtle.turn == -90
    
    def test_turtle_invalid_turn(self):
        """Test that invalid turn raises exception"""
        with pytest.raises(Exception):
            Turtle([10, 45])  # 45 degrees not allowed
    
    def test_turtle_has_cplxtrans(self):
        """Test Turtle has complex transform"""
        turtle = Turtle([10, 90])
        assert hasattr(turtle, 'cplxtrans')
        assert isinstance(turtle.cplxtrans, pya.CplxTrans)
    
    def test_turtle_has_vector(self):
        """Test Turtle has vector"""
        turtle = Turtle([15, -90])
        assert hasattr(turtle, 'vector')
        assert isinstance(turtle.vector, pya.Vector)
        assert turtle.vector.x == 15
        assert turtle.vector.y == 0
    
    def test_turtle_display(self):
        """Test Turtle display method doesn't crash"""
        turtle = Turtle([20, 90])
        # Should not raise an exception
        turtle.display()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


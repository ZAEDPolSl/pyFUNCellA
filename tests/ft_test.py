import numpy as np
import pytest

import source.normalize.freeman_tukey as ft

def test_ft():
    sq = np.array([[1+(2)**(1/2), 2+(5)**(1/2), 3+(10)**(1/2)], 
                  [4+(17)**(1/2), 5+(26)**(1/2), 6+(37)**(1/2)],
                  [1, 0.5+(1.25)**(1/2), 0.1+(1.01)**(1/2)]])
    data = np.array([[1, 4, 9], 
                  [16, 25, 36],
                  [0, 0.25, 0.01]])
    res = ft.freeman_tukey(data)
    assert np.array_equal(sq, res)
    
def test_ft_raises_exception():
    data = np.array([[-1,2,0,4,0],
                     [6,0,6,2,0],
                     [0,7,3,6,0],
                     [0,9,9,0,0],
                     [0,5,0,5,0]])
    with pytest.raises(Exception) as e_info:
        res = ft.freeman_tukey(data)
                
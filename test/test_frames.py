
import vamp
import numpy as np

def to_lists(arrs):
    return [list([list(r) for r in f]) for f in arrs]

def test_frames_from_1d_buffer():
    buf = np.arange(6)
    ff = to_lists(vamp.frames_from_array(buf, 2, 2))
    assert(ff == [[[0,1]],[[2,3]],[[4,5]]])
    ff = to_lists(vamp.frames_from_array(buf, 1, 2))
    assert(ff == [[[0,1]],[[1,2]],[[2,3]],[[3,4]],[[4,5]],[[5,0]]])

def test_frames_from_2d_buffer():
    buf = np.array([np.arange(6),np.arange(6,12)])
    ff = to_lists(vamp.frames_from_array(buf, 2, 2))
    assert(ff == [[[0,1],[6,7]],[[2,3],[8,9]],[[4,5],[10,11]]])
    ff = to_lists(vamp.frames_from_array(buf, 1, 2))
    assert(ff == [[[0,1],[6,7]],[[1,2],[7,8]],[[2,3],[8,9]],
                  [[3,4],[9,10]],[[4,5],[10,11]],[[5,0],[11,0]]])

    

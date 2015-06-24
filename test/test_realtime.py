
import vamp

def test_basic_conf_compare_sec():
    r1 = vamp.vampyhost.RealTime('seconds', 0)
    r2 = vamp.vampyhost.RealTime('seconds', 0)
    assert r1 == r2
    r2a = vamp.vampyhost.RealTime()
    assert r1 == r2a
    r3 = vamp.vampyhost.RealTime('seconds', 1.5)
    assert r1 != r3
    assert r2 != r3
    assert r1 < r3
    assert r3 > r2
    assert r1 <= r3
    assert r3 >= r2
    assert r1 >= r2

def test_basic_conf_compare_msec():
    r1 = vamp.vampyhost.RealTime('milliseconds', 0)
    r2 = vamp.vampyhost.RealTime('milliseconds', 0)
    assert r1 == r2
    r3 = vamp.vampyhost.RealTime('milliseconds', 1500)
    assert r1 != r3
    assert r2 != r3
    assert r1 < r3
    assert r3 > r2
    assert r1 <= r3
    assert r3 >= r2
    assert r1 >= r2

def test_basic_conf_compare_sec_msec():
    r1 = vamp.vampyhost.RealTime('milliseconds', 0)
    r2 = vamp.vampyhost.RealTime('seconds', 0)
    assert r1 == r2
    r3 = vamp.vampyhost.RealTime('milliseconds', 1500)
    r4 = vamp.vampyhost.RealTime('seconds', 1.5)
    assert r3 == r4
    assert r1 != r3
    assert r2 != r3
    assert r1 < r3
    assert r3 > r2
    assert r1 <= r3
    assert r3 >= r2
    assert r4 >= r2
    assert r1 >= r2
    assert r4 <= r3

def test_basic_conf_compare_int_float():
    r1 = vamp.vampyhost.RealTime('seconds', 100)
    r2 = vamp.vampyhost.RealTime('seconds', 100.0)
    assert r1 == r2
    r2n = vamp.vampyhost.RealTime('seconds', 100.00001)
    assert r1 != r2n
    assert r2 != r2n
    r1 = vamp.vampyhost.RealTime('milliseconds', 100)
    r2 = vamp.vampyhost.RealTime('milliseconds', 100.0)
    r2n = vamp.vampyhost.RealTime('milliseconds', 100.00001)
    r3 = vamp.vampyhost.RealTime('seconds', 0.1)
    assert r1 == r2
    assert r1 != r2n
    assert r2 != r2n
    assert r1 == r3
    assert r2 == r3
    
def test_basic_conf_compare_tuple():
    r1 = vamp.vampyhost.RealTime(0, 0)
    r2 = vamp.vampyhost.RealTime(0, 0)
    assert r1 == r2
    r3 = vamp.vampyhost.RealTime(1, 500000000)
    r4 = vamp.vampyhost.RealTime('seconds', 1.5)
    assert r3 == r4
    assert r1 != r3
    assert r2 != r3
    assert r1 < r3
    assert r3 > r2
    assert r1 <= r3
    assert r3 >= r2
    assert r4 >= r2
    assert r1 >= r2
    assert r4 <= r3

def test_conv_float():
    r = vamp.vampyhost.RealTime('seconds', 0)
    assert float(r) == 0.0

def test_conv_float():
    r = vamp.vampyhost.RealTime('seconds', 0)
    assert float(r) == 0.0

def test_conv_str():    
    r = vamp.vampyhost.RealTime('seconds', 0)
    assert str(r) == " 0.000000000"
    r = vamp.vampyhost.RealTime('seconds', 1.5)
    assert str(r) == " 1.500000000"
    r = vamp.vampyhost.RealTime('seconds', -2)
    assert str(r) == "-2.000000000"
    r = vamp.vampyhost.RealTime(-1, -500000000)
    assert str(r) == "-1.500000000"

def test_add_subtract():
    r1 = vamp.vampyhost.RealTime('milliseconds', 400)
    r2 = vamp.vampyhost.RealTime('milliseconds', 600)
    r3 = vamp.vampyhost.RealTime('seconds', 1)
    assert r1 + r2 == r3
    assert r3 - r2 - r1 == vamp.vampyhost.RealTime()
    assert r2 - r1 == vamp.vampyhost.RealTime('milliseconds', 200)
    assert r1 - r2 == vamp.vampyhost.RealTime('milliseconds', -200)
    

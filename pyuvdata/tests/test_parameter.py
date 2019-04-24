# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

import nose.tools as nt
import numpy as np
import astropy.units as units

from pyuvdata import parameter as uvp
from pyuvdata.uvbase import UVBase


def test_class_inequality():
    """Test equality error for different uvparameter classes."""
    param1 = uvp.UVParameter(name='p1', value=1)
    param2 = uvp.AngleParameter(name='p2', value=1)
    nt.assert_not_equal(param1, param2)


def test_value_class_inequality():
    """Test equality error for different uvparameter classes."""
    param1 = uvp.UVParameter(name='p1', value=3)
    param2 = uvp.UVParameter(name='p2', value=np.array([3, 4, 5]))
    nt.assert_not_equal(param1, param2)
    nt.assert_not_equal(param2, param1)
    param3 = uvp.UVParameter(name='p2', value='Alice')
    nt.assert_not_equal(param1, param3)


def test_array_inequality():
    """Test equality error for different array values."""
    param1 = uvp.UVParameter(name='p1', value=np.array([0, 1, 3]))
    param2 = uvp.UVParameter(name='p2', value=np.array([0, 2, 4]))
    nt.assert_not_equal(param1, param2)
    param3 = uvp.UVParameter(name='p3', value=np.array([0, 1]))
    nt.assert_not_equal(param1, param3)


def test_string_inequality():
    """Test equality error for different string values."""
    param1 = uvp.UVParameter(name='p1', value='Alice')
    param2 = uvp.UVParameter(name='p2', value='Bob')
    nt.assert_not_equal(param1, param2)


def test_string_list_inequality():
    """Test equality error for different string values."""
    param1 = uvp.UVParameter(name='p1', value=['Alice', 'Eve'])
    param2 = uvp.UVParameter(name='p2', value=['Bob', 'Eve'])
    nt.assert_not_equal(param1, param2)


def test_string_equality():
    """Test equality error for different string values."""
    param1 = uvp.UVParameter(name='p1', value='Alice')
    param2 = uvp.UVParameter(name='p2', value='Alice')
    nt.assert_equal(param1, param2)


def test_integer_inequality():
    """Test equality error for different non-array, non-string values."""
    param1 = uvp.UVParameter(name='p1', value=1)
    param2 = uvp.UVParameter(name='p2', value=2)
    nt.assert_not_equal(param1, param2)


def test_dict_equality():
    """Test equality for dict values."""
    param1 = uvp.UVParameter(name='p1', value={'v1': 1})
    param2 = uvp.UVParameter(name='p2', value={'v1': 1})
    nt.assert_equal(param1, param2)


def test_dict_inequality_int():
    """Test equality error for integer dict values."""
    param1 = uvp.UVParameter(name='p1', value={'v1': 1, 's1': 'test'})
    param2 = uvp.UVParameter(name='p2', value={'v1': 2, 's1': 'test'})
    nt.assert_not_equal(param1, param2)


def test_dict_inequality_str():
    """Test equality error for string dict values."""
    param1 = uvp.UVParameter(name='p1', value={'v1': 1, 's1': 'test'})
    param4 = uvp.UVParameter(name='p3', value={'v1': 1, 's1': 'foo'})
    nt.assert_not_equal(param1, param4)


def test_dict_inequality_keys():
    """Test equality error for different keys."""
    param1 = uvp.UVParameter(name='p1', value={'v1': 1, 's1': 'test'})
    param3 = uvp.UVParameter(name='p3', value={'v3': 1, 's1': 'test'})
    nt.assert_not_equal(param1, param3)


def test_equality_check_fail():
    """Test equality error for non string, dict or array values."""
    param1 = uvp.UVParameter(name='p1', value=uvp.UVParameter(name='p1', value='Alice'))
    param2 = uvp.UVParameter(name='p2', value=uvp.UVParameter(name='p1', value='Bob'))
    nt.assert_not_equal(param1, param2)


def test_notclose():
    """Test equality error for values not with tols."""
    param1 = uvp.UVParameter(name='p1', value=1.0)
    param2 = uvp.UVParameter(name='p2', value=1.001)
    nt.assert_not_equal(param1, param2)


def test_close():
    """Test equality error for values within tols."""
    param1 = uvp.UVParameter(name='p1', value=1.0)
    param2 = uvp.UVParameter(name='p2', value=1.000001)
    nt.assert_equal(param1, param2)


def test_acceptability():
    """Test check_acceptability function."""
    param1 = uvp.UVParameter(name='p1', value=1000, acceptable_range=(1, 10))
    nt.assert_false(param1.check_acceptability()[0])

    param1 = uvp.UVParameter(name='p1', value=np.random.rand(100), acceptable_range=(.1, .9))
    nt.assert_true(param1.check_acceptability()[0])
    param1 = uvp.UVParameter(name='p1', value=np.random.rand(100) * 1e-4, acceptable_range=(.1, .9))
    nt.assert_false(param1.check_acceptability()[0])

    param2 = uvp.UVParameter(name='p2', value=5, acceptable_range=(1, 10))
    nt.assert_true(param2.check_acceptability()[0])
    param2 = uvp.UVParameter(name='p2', value=5, acceptable_vals=[1, 10])
    nt.assert_false(param2.check_acceptability()[0])


def test_string_acceptability():
    """Test check_acceptability function with strings."""
    param1 = uvp.UVParameter(name='p1', value='Bob', form='str',
                             acceptable_vals=['Alice', 'Eve'])
    nt.assert_false(param1.check_acceptability()[0])
    param2 = uvp.UVParameter(name='p2', value='Eve', form='str',
                             acceptable_vals=['Alice', 'Eve'])
    nt.assert_true(param2.check_acceptability()[0])


def test_expected_shape():
    """Test missing shape param."""
    class TestUV(UVBase):
        def __init__(self):
            self._p1 = uvp.UVParameter(name='p1', required=False)
            self._p2 = uvp.UVParameter(name='p2', form=('p1',))
            self._p3 = uvp.UVParameter(name='p3', form=(2,))
            super(TestUV, self).__init__()
    obj = TestUV()
    obj.p2 = np.array([0, 5, 8])
    obj.p3 = np.array([4, 9])
    nt.assert_raises(ValueError, obj.check)
    nt.assert_equal(obj._p3.expected_shape(obj), (2,))


def test_angle_set_degree_none():
    param1 = uvp.AngleParameter(name='p2', value=1)
    param1.set_degrees(None)

    nt.assert_equal(None, param1.value)


def test_location_set_lat_lon_alt_none():
    param1 = uvp.LocationParameter(name='p2', value=1)
    param1.set_lat_lon_alt(None)

    nt.assert_equal(None, param1.value)


def test_location_set_lat_lon_alt_degrees_none():
    param1 = uvp.LocationParameter(name='p2', value=1)
    param1.set_lat_lon_alt_degrees(None)

    nt.assert_equal(None, param1.value)


def test_location_acceptable_none():
    param1 = uvp.LocationParameter(name='p2', value=1, acceptable_range=None)

    nt.assert_true(param1.check_acceptability())


def test_UnitParameter_error_flag_not_set():
    """Test an error is raised if the value_not_parameter flag is not set with non Quantity."""
    nt.assert_raises(ValueError, uvp.UnitParameter, name='p1', value=3)


def test_UnitParameter_error_flag_not_set_none_value():
    """Test an error is raised if the value_not_parameter flag is not set with non Quantity."""
    nt.assert_raises(ValueError, uvp.UnitParameter, name='p1', value=None)


def test_UnitParameter_no_expected_units():
    """Test error is raised if no expected units are supplied to UnitParameter."""
    nt.assert_raises(ValueError, uvp.UnitParameter, name='p1',
                     value=1 * units.m)


def test_UnitParameter_tolerance_wrong_sized_Quantity():
    """Test an error is raised if a UnitParameter is initialized with tolerance being a Quantity with size > 1."""
    nt.assert_raises(ValueError, uvp.UnitParameter, name='p1',
                     value=3 * units.m, tols=(1e-9, 1) * units.m,
                     expected_units=units.m)


def test_UnitParameter_single_Quantity():
    """Test that a single tolerance quantity is upgraded to tuple and units preserved."""
    param1 = uvp.UnitParameter(name='p1', value=3 * units.m, tols=1e-3 * units.m,
                               expected_units=units.m)
    nt.assert_true(isinstance(param1.tols, tuple))
    nt.assert_equal(param1.tols[1].unit, units.m)


def test_UnitParameter_single_flaot():
    """Test that a single tolerance float is upgraded to tuple and units preserved."""
    param1 = uvp.UnitParameter(name='p1', value=3 * units.m, tols=1e-3,
                               expected_units=units.m)
    nt.assert_true(isinstance(param1.tols, tuple))
    nt.assert_equal(param1.tols[1].unit, units.m)


def test_UnitParameter_tolerance_incompatible_units():
    """Test an error is raised if a UnitParameter is initialized with tolerance having incompatible units."""
    nt.assert_raises(units.UnitConversionError, uvp.UnitParameter, name='p1',
                     value=3 * units.m, tols=(0, 1 * units.s),
                     expected_units=units.m)


def test_UnitParameter_from_incompatible_list():
    """Test an error is raised when attempting to initialize UnitParameter with list of incompatible Quantites."""
    nt.assert_raises(ValueError, uvp.UnitParameter, name='p1',
                     value=[1 * units.m, 2 * units.s], expected_units=units.m,
                     tols=(0, 1e-2 * units.m)
                     )


def test_UnitParameter_incompatible_expected_units():
    """Test Error is raised when value and expected_units are incompatible."""
    nt.assert_raises(ValueError, uvp.UnitParameter, name='p1',
                     value=1 * units.m, expected_units=units.s,
                     tols=(0, 1e-2 * units.s)
                     )


def test_class_inequality_UnitParameter():
    """Test object is not equal to general other object."""
    param1 = uvp.UnitParameter(name='p1', value=3, value_not_quantity=True)
    param2 = uvp.UVParameter(name='p2', value=1)
    nt.assert_not_equal(param1, param2)


def test_value_class_inequality_UnitParameter():
    """Test false is returned if values have different classes for UnitParameter."""
    param1 = uvp.UnitParameter(name='p2', value=np.array(3) * units.m,
                               expected_units=units.m)
    param2 = uvp.UnitParameter(name='p3', value='test string',
                               value_not_quantity=True)
    nt.assert_not_equal(param1, param2)
    nt.assert_not_equal(param2, param1)


def test_two_non_quantity_UnitParameters_equl():
    """Test two UnitParameters with non-quanitites are equated."""
    param1 = uvp.UnitParameter(name='p1', value=3,
                               value_not_quantity=True)
    param2 = uvp.UnitParameter(name='p2', value=3,
                               value_not_quantity=True)
    nt.assert_equal(param1, param2)
    nt.assert_equal(param2, param1)


def test_shape_inequality_UnitParameter():
    """Test false is returned if values have different shapes for UnitParameter."""
    # since param1 is initialized from a list it has shape (1,)
    param1 = uvp.UnitParameter(name='p1', value=[3 * units.m],
                               expected_units=units.m)
    # param2 has shape ()
    param2 = uvp.UnitParameter(name='p2', value=np.array(3) * units.m,
                               expected_units=units.m)
    nt.assert_not_equal(param1, param2)
    nt.assert_not_equal(param2, param1)


def test_incompatible_units_inequality():
    """Test UnitParameters with incompatible units are not equal."""
    param1 = uvp.UnitParameter(name='p1', value=3 * units.s,
                               expected_units=units.s)
    param2 = uvp.UnitParameter(name='p2', value=np.array(3) * units.m,
                               expected_units=units.m)
    nt.assert_not_equal(param1, param2)
    nt.assert_not_equal(param2, param1)


def test_incompatible_value_inequality():
    """Test UnitParameters with different values are not equal."""
    param1 = uvp.UnitParameter(name='p1', value=5 * units.m,
                               expected_units=units.m)
    param2 = uvp.UnitParameter(name='p2', value=3 * units.m,
                               expected_units=units.m)
    nt.assert_not_equal(param1, param2)
    nt.assert_not_equal(param2, param1)


def test_tolerance_give_equality_UnitParameter():
    """Test UnitParameters which are equal within tolerance are equal."""
    param1 = uvp.UnitParameter(name='p1', value=5 * units.m,
                               tols=(0, 3 * units.m),
                               expected_units=units.m)
    param2 = uvp.UnitParameter(name='p2', value=3 * units.m,
                               expected_units=units.m)
    nt.assert_equal(param1, param2)

    # Also test that tolerances must be set on both to not be a two sided test.
    nt.assert_not_equal(param2, param1)


def test_incompatible_units_fails_check_acceptability():
    """Test check check_acceptability retruns false if value of UnitParameter changed to incompatible units."""
    param1 = uvp.UnitParameter(name='p1', value=5 * units.m,
                               tols=(0, 3 * units.m),
                               expected_units=units.m)
    nt.assert_true(param1.check_acceptability()[0])

    param1.value = 5 * units.s
    nt.assert_false(param1.check_acceptability()[0])

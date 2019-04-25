# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""
Define UVParameters: data and metadata objects for interferometric data sets.

UVParameters are objects to hold specific data and metadata associated with
interferometric data sets. They are used as attributes for classes based on
UVBase. This module also includes specialized sublasses for particular types
of metadata.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import six
import astropy.units as units

from . import utils


class UVParameter(object):
    """
    Data and metadata objects for interferometric data sets.

    Attributes:
        name: A string giving the name of the attribute. Used as the associated
            property name in classes based on UVBase.

        required: A boolean indicating whether this is required metadata for
            the class with this UVParameter as an attribute. Default is True.

        value: The value of the data or metadata.

        spoof_val: A fake value that can be assigned to a non-required
            UVParameter if the metadata is required for a particular file-type.
            This is not an attribute of required UVParameters.

        form: Either 'str' or a tuple giving information about the expected
            shape of the value. Elements of the tuple may be the name of other
            UVParameters that indicate data shapes. \n
            Examples:\n
                'str': a string value\n
                ('Nblts', 3): the value should be an array of shape: Nblts (another UVParameter name), 3

        description: A string description of the data or metadata in the object.

        expected_type: The type that the data or metadata should be.
            Default is np.int or str if form is 'str'

        acceptable_vals: Optional. List giving allowed values for elements of value.

        acceptable_range: Optional. Tuple giving a range of allowed magnitudes for elements of value.

        tols: Tolerances for testing the equality of UVParameters. Either a
            single absolute value or a tuple of relative and absolute values to
            be used by np.isclose()
    """

    def __init__(self, name, required=True, value=None, spoof_val=None,
                 form=(), description='', expected_type=int, acceptable_vals=None,
                 acceptable_range=None, tols=(1e-05, 1e-08)):
        """Init UVParameter object."""
        self.name = name
        self.required = required
        # cannot set a spoof_val for required parameters
        if not self.required:
            self.spoof_val = spoof_val
        self.value = value
        self.description = description
        self.form = form
        if self.form == 'str':
            self.expected_type = str
        else:
            self.expected_type = expected_type
        self.acceptable_vals = acceptable_vals
        self.acceptable_range = acceptable_range
        if np.size(tols) == 1:
            # Only one tolerance given, assume absolute, set relative to zero
            self.tols = (0, tols)
        else:
            self.tols = tols  # relative and absolute tolerances to be used in np.isclose

    def __eq__(self, other):
        """Equal if classes match and values are identical."""
        if isinstance(other, self.__class__):
            # only check that value is identical
            if not isinstance(self.value, other.value.__class__):
                print('{name} parameter value classes are different. Left is '
                      '{lclass}, right is {rclass}'.format(name=self.name,
                                                           lclass=self.value.__class__,
                                                           rclass=other.value.__class__))
                return False
            if isinstance(self.value, np.ndarray) and not isinstance(self.value[0], six.string_types):
                if self.value.shape != other.value.shape:
                    print('{name} parameter value is array, shapes are '
                          'different'.format(name=self.name))
                    return False
                elif not np.allclose(self.value, other.value,
                                     rtol=self.tols[0], atol=self.tols[1]):
                    print('{name} parameter value is array, values are not '
                          'close'.format(name=self.name))
                    return False
            else:
                str_type = False
                if isinstance(self.value, six.string_types):
                    str_type = True
                if isinstance(self.value, (list, np.ndarray)):
                    if isinstance(self.value[0], str):
                        str_type = True

                if not str_type:
                    try:
                        if not np.allclose(np.array(self.value),
                                           np.array(other.value),
                                           rtol=self.tols[0], atol=self.tols[1]):
                            print('{name} parameter value can be cast to an array'
                                  ' and tested with np.allclose. The values are '
                                  'not close'.format(name=self.name))
                            return False
                    except(TypeError):
                        if self.value != other.value:
                            if isinstance(self.value, dict):
                                # check to see if they are equal other than upper/lower case keys
                                self_lower = {k.lower(): v for k, v in self.value.items()}
                                other_lower = {k.lower(): v for k, v in other.value.items()}
                                if self_lower != other_lower:
                                    message_str = '{name} parameter is a dict'.format(name=self.name)
                                    if set(self_lower.keys()) != set(other_lower.keys()):
                                        message_str += ', keys are not the same.'
                                    else:
                                        # need to check if values are close, not just equal
                                        values_close = True
                                        for key in self_lower.keys():
                                            try:
                                                if not np.isclose(self_lower[key], other_lower[key]):
                                                    message_str += (', key {key} is not '
                                                                    'equal'.format(key=key))
                                                    values_close = False
                                            except(TypeError):
                                                # this isn't a type that can be handled by np.isclose, test for equality
                                                if self_lower[key] != other_lower[key]:
                                                    message_str += (', key {key} is not '
                                                                    'equal'.format(key=key))
                                                    values_close = False
                                        if values_close is False:
                                            print(message_str)
                                            return False
                                        else:
                                            return True
                                else:
                                    return True
                            else:
                                print('{name} parameter value is not a string '
                                      'or a dict and cannot be cast as a numpy '
                                      'array. The values are not equal.'.format(name=self.name))

                            return False

                else:
                    if isinstance(self.value, (list, np.ndarray)):
                        if [s.strip() for s in self.value] != [s.strip() for s in other.value]:
                            print('{name} parameter value is a list of strings, '
                                  'values are different'.format(name=self.name))
                            return False
                    else:
                        if self.value.strip() != other.value.strip():
                            if (self.value.replace('\n', '').replace(' ', '')
                                    != other.value.replace('\n', '').replace(' ', '')):
                                print('{name} parameter value is a string, '
                                      'values are different'.format(name=self.name))
                                return False

            return True
        else:
            print('{name} parameter classes are different'.format(name=self.name))
            return False

    def __ne__(self, other):
        """Not equal."""
        return not self.__eq__(other)

    def apply_spoof(self):
        """Set value to spoof_val for non-required UVParameters."""
        self.value = self.spoof_val

    def expected_shape(self, uvbase):
        """
        Get the expected shape of the value based on the form.

        Args:
            uvbase: object with this UVParameter as an attribute. Needed
                because the form can refer to other UVParameters on this object.

        Returns:
            The expected shape of the value.
        """
        if self.form == 'str':
            return self.form
        elif isinstance(self.form, np.int):
            # Fixed shape, just return the form
            return (self.form, )
        else:
            # Given by other attributes, look up values
            eshape = ()
            for p in self.form:
                if isinstance(p, np.int):
                    eshape = eshape + (p,)
                else:
                    val = getattr(uvbase, p)
                    if val is None:
                        raise ValueError('Missing UVBase parameter {p} needed to '
                                         'calculate expected shape of parameter'.format(p=p))
                    eshape = eshape + (val,)
            return eshape

    def check_acceptability(self):
        """Check that values are acceptable."""
        if self.acceptable_vals is None and self.acceptable_range is None:
            return True, 'No acceptability check'
        else:
            # either acceptable_vals or acceptable_range is set. Prefer acceptable_vals
            if self.acceptable_vals is not None:
                # acceptable_vals are a list of allowed values
                if self.expected_type is str:
                    # strings need to be converted to lower case
                    if isinstance(self.value, str):
                        value_set = set([self.value.lower()])
                    else:
                        # this is a list or array of strings, make them all lower case
                        value_set = set([x.lower() for x in self.value])
                    acceptable_vals = [x.lower() for x in self.acceptable_vals]
                else:
                    if isinstance(self.value, (list, np.ndarray)):
                        value_set = set(list(self.value))
                    else:
                        value_set = set([self.value])
                    acceptable_vals = self.acceptable_vals
                for elem in value_set:
                    if elem not in acceptable_vals:
                        message = ('Value {val}, is not in allowed values: '
                                   '{acceptable_vals}'.format(val=elem, acceptable_vals=acceptable_vals))
                        return False, message
                return True, 'Value is acceptable'
            else:
                # acceptable_range is a tuple giving a range of allowed magnitudes
                testval = np.mean(np.abs(self.value))
                if (testval >= self.acceptable_range[0]) and (testval <= self.acceptable_range[1]):
                    return True, 'Value is acceptable'
                else:
                    message = ('Mean of abs values, {val}, is not in allowed range: '
                               '{acceptable_range}'.format(val=testval, acceptable_range=self.acceptable_range))
                    return False, message


class AntPositionParameter(UVParameter):
    """
    Subclass of UVParameter for antenna positions.

    Overrides apply_spoof method to generate an array of the correct shape based
    on the number of antennas on the object with this UVParameter as an attribute.
    """

    def apply_spoof(self, uvbase, antnum_name):
        """
        Set value to zeroed array of shape: number of antennas, 3.

        Args:
            uvbase: object with this UVParameter as an attribute. Needed
                to get the number of antennas.
            antnum_name: A string giving the name of the UVParameter containing
                the number of antennas.
        """
        self.value = np.zeros((getattr(uvbase, antnum_name), 3))


class AngleParameter(UVParameter):
    """
    Subclass of UVParameter for Angle type parameters.

    Adds extra methods for conversion to & from degrees (used by UVBase objects
    for _degrees properties associated with these parameters).
    """

    def degrees(self):
        """Get value in degrees."""
        if self.value is None:
            return None
        else:
            return self.value * 180. / np.pi

    def set_degrees(self, degree_val):
        """
        Set value in degrees.

        Args:
            degree_val: value in degrees to use to set the value attribute.
        """
        if degree_val is None:
            self.value = None
        else:
            self.value = degree_val * np.pi / 180.


class LocationParameter(UVParameter):
    """
    Subclass of UVParameter for Earth location type parameters.

    Adds extra methods for conversion to & from lat/lon/alt in radians or
    degrees (used by UVBase objects for _lat_lon_alt and _lat_lon_alt_degrees
    properties associated with these parameters).
    """
    def __init__(self, name, required=True, value=None, spoof_val=None, description='',
                 acceptable_range=(6.35e6, 6.39e6), tols=1e-3):
        super(LocationParameter, self).__init__(name, required=required, value=value,
                                                spoof_val=spoof_val, form=3,
                                                description=description,
                                                expected_type=np.float,
                                                acceptable_range=acceptable_range, tols=tols)

    def lat_lon_alt(self):
        """Get value in (latitude, longitude, altitude) tuple in radians."""
        if self.value is None:
            return None
        else:
            return utils.LatLonAlt_from_XYZ(self.value)

    def set_lat_lon_alt(self, lat_lon_alt):
        """
        Set value from (latitude, longitude, altitude) tuple in radians.

        Args:
            lat_lon_alt: tuple giving the latitude (radians), longitude (radians)
                and altitude to use to set the value attribute.
        """
        if lat_lon_alt is None:
            self.value = None
        else:
            self.value = utils.XYZ_from_LatLonAlt(lat_lon_alt[0], lat_lon_alt[1],
                                                  lat_lon_alt[2])

    def lat_lon_alt_degrees(self):
        """Get value in (latitude, longitude, altitude) tuple in degrees."""
        if self.value is None:
            return None
        else:
            latitude, longitude, altitude = utils.LatLonAlt_from_XYZ(self.value)
            return latitude * 180. / np.pi, longitude * 180. / np.pi, altitude

    def set_lat_lon_alt_degrees(self, lat_lon_alt_degree):
        """
        Set value from (latitude, longitude, altitude) tuple in degrees.

        Args:
            lat_lon_alt: tuple giving the latitude (degrees), longitude (degrees)
                and altitude to use to set the value attribute.
        """
        if lat_lon_alt_degree is None:
            self.value = None
        else:
            latitude, longitude, altitude = lat_lon_alt_degree
            self.value = utils.XYZ_from_LatLonAlt(latitude * np.pi / 180.,
                                                  longitude * np.pi / 180.,
                                                  altitude)

    def check_acceptability(self):
        """Check that values are acceptable. Special case for location, where
            we want to check the vector magnitude
        """
        if self.acceptable_range is None:
            return True, 'No acceptability check'
        else:
            # acceptable_range is a tuple giving a range of allowed vector magnitudes
            testval = np.sqrt(np.sum(np.abs(self.value)**2))
            if (testval >= self.acceptable_range[0]) and (testval <= self.acceptable_range[1]):
                return True, 'Value is acceptable'
            else:
                message = ('Value {val}, is not in allowed range: '
                           '{acceptable_range}'.format(val=testval, acceptable_range=self.acceptable_range))
                return False, message


class UnitParameter(UVParameter):
    """Subclass of UVParameters with astropy Quantity compatibility.

    Adds checks for Astropy Quantity objects and equality between Quantites.
    """

    def __init__(self, name, required=True, value=None, spoof_val=None,
                 form=(), description='', expected_type=int,
                 acceptable_vals=None, acceptable_range=None,
                 expected_units=None,
                 tols=(1e-05, 1e-08), value_not_quantity=False):
        """Initialize the UnitParameter.

        This object relies on all arguments and keywords from the UVParamter Class.

        The following is a list of additional keywords:
        Extra keywords:
            value_not_quantity: (Boolean, default False)
                                Boolean flag used to specify that input value
                                is not an astropy Quantity object, but a
                                UnitParameter is desired over a UVParameter.

        """
        # First check that the value is exepcted to be a Quantity and if not `value_not_quantity` is True
        if not (value_not_quantity or isinstance(expected_units, units.UnitBase)):
            raise ValueError("Input expected_units are not an astropy.Unit"
                             " object and the user did not specify "
                             "value_not_quantity flag.")

        # Must check if tolerance is given as a quantity first.
        # There are a few cases of possible input tolerances:
        #  - User gives a single value as a quantity.
        #     -- This _must_ bet the absolute error.
        #  - User gives a tuple with neither entry being a quantity
        #     -- This is assumed to be a tuple of (relative, aboslute) errors and the unit from the quantity input is applied to the absolute error
        #  - User provides a tuple (relative, absolute) where the absolute error is a quantity with appropriate units
        #     -- This is the prefered and least ambiguous option.
        #  - User provides a tuple of Quantity objects (relative, absolute)
        #     -- This is non-physical relative error should be unitless
        # These cases must be checked prior to the super().__init__ call to accomodate for the units
        if not value_not_quantity:
            if isinstance(tols, units.Quantity):
                if tols.size > 1:
                    raise ValueError("Tolerance values that are Quantity "
                                     "objects must be a single value to "
                                     "represent the absolute tolerance.")
                else:
                    # Only one tolerance given, assume absolute, set relative to zero
                    tols = tuple((0, tols))

            if len(utils._get_iterable(tols)) == 1:
                # Only one tolerance given, assume absolute, set relative to zero
                tols = tuple((0, tols))

            if not isinstance(tols[1], units.Quantity):
                print("Given absolute tolerance did not all have units. "
                      "Applying expected units from parameter.")
                tols = tuple((tols[0], tols[1] * expected_units))

            if not tols[1].unit.is_equivalent(expected_units):
                raise units.UnitConversionError("Given absolute tolerance "
                                                "did not all have units "
                                                "compatible with expected_units.")
            tols = tuple((tols[0], tols[1].to(expected_units).value))

        super(UnitParameter, self).__init__(name=name, required=required,
                                            value=value,
                                            spoof_val=spoof_val, form=form,
                                            description=description,
                                            expected_type=expected_type,
                                            acceptable_vals=acceptable_vals,
                                            acceptable_range=acceptable_range,
                                            tols=tols)
        self.value_not_quantity = value_not_quantity
        self.expected_units = expected_units
        # For Quantity objects there is some more set-up and checking to do.
        if not self.value_not_quantity:
            # now re-wrap the tolerances with all the necessary information
            self.tols = tuple((self.tols[0], self.tols[1] * self.expected_units))
            if isinstance(self.value, list) and isinstance(self.value[0], units.Quantity):
                try:
                    self.value = units.Quantity(self.value)
                except units.UnitConversionError:
                    raise ValueError("Unable to create UnitParameter objects "
                                     "from lists whose elements have "
                                     "non-comaptible units.")
            if not self.value.unit.is_equivalent(self.expected_units):
                raise units.UnitConversionError("Input value has units {0} "
                                                "which are not equivalent to "
                                                "expected units of {1}"
                                                .format(self.value.unit,
                                                        self.expected_units))
            else:
                # store in the expected unit
                self.value = self.value.to(self.expected_units)

    def __eq__(self, other):
        """Classes must match with identical values for equality."""
        if not isinstance(other, self.__class__):
            print('{name} parameter value classes are different and one '
                  'is not a subclass of the other. Left is '
                  '{lclass}, right is {rclass}'.format(name=self.name,
                                                       lclass=self.__class__,
                                                       rclass=other.__class__))
            return False
        else:
            # If values are different types of object self and other are not equal.
            if not isinstance(self.value, other.value.__class__):
                print('{name} parameter value classes are different. Left is '
                      '{lclass}, right is {rclass}'.format(name=self.name,
                                                           lclass=self.value.__class__,
                                                           rclass=other.value.__class__))
                return False
            if self.value_not_quantity:
                # If value is not a quantity then use the UVParameter __eq__
                return super(UnitParameter, self).__eq__(other)
            else:
                if self.value.shape != other.value.shape:
                    print('{name} parameter value is array, shapes are '
                          'different'.format(name=self.name))
                    return False
                elif not self.value.unit.is_equivalent(other.value.unit):
                    print('{name} parameter is Quantity, but have '
                          'non-compatible units '.format(name=self.name))
                    return False
                # astropy.units has a units.allclose but only for python 3
                # already know the units are compatible so
                # Convert other to self's units and compare values
                if not np.allclose(self.value.to(self.expected_units).value,
                                   other.value.to(self.expected_units).value,
                                   rtol=self.tols[0],
                                   atol=self.tols[1].to(self.expected_units).value):
                    print('{name} parameter value is array, values are not '
                          'close'.format(name=self.name))
                    return False
                else:
                    return True

    def __ne__(self, other):
        """Not Equal."""
        return not self.__eq__(other)

    def check_acceptability(self):
        """Added compatibility tests for units on top of base compatibility."""
        if not self.value.unit.is_equivalent(self.expected_units):
            message = ("Input value has units {0} "
                       "which are not equivalent to "
                       "expected units of {1}"
                       .format(self.value.unit, self.expected_units))
            return False, message
        else:
            return super(UnitParameter, self).check_acceptability()

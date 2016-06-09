from astropy import constants as const
from astropy.time import Time
import os.path as op
import numpy as np
from scipy.io.idl import readsav
import warnings
from itertools import islice
import aipy as a
import os
import ephem
from astropy.utils import iers
import uvdata

data_path = op.join(uvdata.__path__[0], 'data')

iers_a = iers.IERS_A.open(op.join(data_path, 'finals.all'))


class UVProperty:
    def __init__(self, required=True, value=None, spoof_val=None,
                 form=(), description='', expected_type=np.int, sane_vals=None,
                 tols=(1e-05, 1e-08)):
        self.required = required
        # cannot set a spoof_val for required properties
        if not self.required:
            self.spoof_val = spoof_val
        self.value = value
        self.description = description
        self.form = form
        if self.form == 'str':
            self.expected_type = str
        else:
            self.expected_type = expected_type
        self.sane_vals = sane_vals
        if np.size(tols) == 1:
            # Only one tolerance given, assume absolute, set relative to zero
            self.tols = (0, tols)
        else:
            self.tols = tols  # relative and absolute tolerances to be used in np.isclose

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # only check that value is identical
            isequal = True
            if not isinstance(self.value, other.value.__class__):
                isequal = False
            if isinstance(self.value, np.ndarray):
                if self.value.shape != other.value.shape:
                    isequal = False
                elif not np.allclose(self.value, other.value,
                                     rtol=self.tols[0], atol=self.tols[1]):
                    isequal = False
            else:
                str_type = False
                if isinstance(self.value, (str, unicode)):
                    str_type = True
                if isinstance(self.value, list):
                    if isinstance(self.value[0], str):
                        str_type = True

                if not str_type:
                    try:
                        if not np.isclose(np.array(self.value),
                                          np.array(other.value),
                                          rtol=self.tols[0], atol=self.tols[1]):
                            isequal = False
                    except:
                        print self.value, other.value
                        isequal = False
                else:
                    if self.value != other.value:
                        if not isinstance(self.value, list):
                            if self.value.replace('\n', '') != other.value.replace('\n', ''):
                                isequal = False
                        else:
                            isequal = False

            return isequal
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def apply_spoof(self, *args):
        self.value = self.spoof_val

    def expected_size(self, dataobj):
        # Takes the form of the property and returns the size
        # expected, given values in the UVData object
        if self.form == 'str':
            return self.form
        elif isinstance(self.form, np.int):
            # Fixed size, just return the form
            return self.form
        else:
            # Given by other attributes, look up values
            esize = ()
            for p in self.form:
                if isinstance(p, np.int):
                    esize = esize + (p,)
                else:
                    prop = getattr(dataobj, p)
                    if prop.value is None:
                        raise ValueError('Missing UVData property {p} needed to '
                                         'calculate expected size of property'.format(p=p))
                    esize = esize + (prop.value,)
            return esize

    def sanity_check(self):
        # A quick method for checking that values are sane
        # This needs development
        sane = False  # Default to insanity
        if self.sane_vals is None:
            sane = True
        else:
            testval = np.mean(np.abs(self.value))
            if (testval >= self.sane_vals[0]) and (testval <= self.sane_vals[1]):
                sane = True
        return sane


class AntPositionUVProperty(UVProperty):
    def apply_spoof(self, uvdata):
        self.value = np.zeros((len(uvdata.antenna_indices.value), 3))


class ExtraKeywordUVProperty(UVProperty):
    def __init__(self, required=False, value={}, spoof_val={},
                 description=''):
        self.required = required
        # cannot set a spoof_val for required properties
        if not self.required:
            self.spoof_val = spoof_val
        self.value = value
        self.description = description


class AngleUVProperty(UVProperty):
    def degrees(self):
        if self.value is None:
            return None
        else:
            return self.value * 180. / np.pi

    def set_degrees(self, degree_val):
        if degree_val is None:
            self.value = None
        else:
            self.value = degree_val * np.pi / 180.


class UVData:
    supported_file_types = ['uvfits', 'miriad', 'fhd']

    def __init__(self):
        # add the default_required_attributes to the class?
        # dimension definitions
        self.Ntimes = UVProperty(description='Number of times')
        self.Nbls = UVProperty(description='number of baselines')
        self.Nblts = UVProperty(description='Ntimes * Nbls')
        self.Nfreqs = UVProperty(description='number of frequency channels')
        self.Npols = UVProperty(description='number of polarizations')

        desc = ('array of the visibility data, size: (Nblts, Nspws, Nfreqs, '
                'Npols), type = complex float, in units of self.vis_units')
        self.data_array = UVProperty(description=desc,
                                     form=('Nblts', 'Nspws', 'Nfreqs', 'Npols'),
                                     expected_type=np.complex)

        self.vis_units = UVProperty(description='Visibility units, options '
                                    '["uncalib","Jy","K str"]', form='str')

        desc = ('number of data points averaged into each data element, '
                'type = int, same shape as data_array')
        self.nsample_array = UVProperty(description=desc,
                                        form=('Nblts', 'Nspws', 'Nfreqs', 'Npols'),
                                        expected_type=(np.float, np.int))

        self.flag_array = UVProperty(description='boolean flag, True is '
                                     'flagged, same shape as data_array.',
                                     form=('Nblts', 'Nspws', 'Nfreqs', 'Npols'),
                                     expected_type=np.bool)

        self.Nspws = UVProperty(description='number of spectral windows '
                                '(ie non-contiguous spectral chunks)')

        self.spw_array = UVProperty(description='array of spectral window '
                                    'numbers', form=('Nspws',))

        desc = ('Projected baseline vectors relative to phase center, ' +
                '(3,Nblts), units meters')
        self.uvw_array = UVProperty(description=desc, form=(3, 'Nblts'),
                                    expected_type=np.float, sane_vals=(1e-3, 1e8),
                                    tols=.001)

        self.time_array = UVProperty(description='array of times, center '
                                     'of integration, dimension (Nblts), '
                                     'units Julian Date', form=('Nblts',),
                                     expected_type=np.float,
                                     tols=1e-3 / (60.0 * 60.0 * 24.0))  # 1 ms in days

        self.lst_array = UVProperty(description='array of lsts, center '
                                    'of integration, dimension (Nblts), '
                                    'units radians', form=('Nblts',),
                                    expected_type=np.float,
                                    tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 ms in radians

        desc = ('array of first antenna indices, dimensions (Nblts), '
                'type = int, 0 indexed')
        self.ant_1_array = UVProperty(description=desc, form=('Nblts',))
        desc = ('array of second antenna indices, dimensions (Nblts), '
                'type = int, 0 indexed')
        self.ant_2_array = UVProperty(description=desc, form=('Nblts',))

        desc = ('array of baseline indices, dimensions (Nblts), '
                'type = int; baseline = 2048 * (ant2+1) + (ant1+1) + 2^16 '
                '(may this break casa?)')
        self.baseline_array = UVProperty(description=desc, form=('Nblts',))

        # this dimensionality of freq_array does not allow for different spws
        # to have different dimensions
        self.freq_array = UVProperty(description='array of frequencies, '
                                     'dimensions (Nspws,Nfreqs), units Hz',
                                     form=('Nspws', 'Nfreqs'),
                                     expected_type=np.float,
                                     tols=1e-3)  # mHz

        desc = ('array of polarization integers (Npols). '
                'AIPS Memo 117 says: stokes 1:4 (I,Q,U,V);  '
                'circular -1:-4 (RR,LL,RL,LR); linear -5:-8 (XX,YY,XY,YX)')
        self.polarization_array = UVProperty(description=desc, form=('Npols',))

        self.integration_time = UVProperty(description='length of the '
                                           'integration (s)',
                                           expected_type=np.float,
                                           tols=1e-3)  # 1 ms
        self.channel_width = UVProperty(description='width of channel (Hz)',
                                        expected_type=np.float,
                                        tols=1e-3)  # 1 mHz

        # --- observation information ---
        self.object_name = UVProperty(description='source or field '
                                      'observed (string)', form='str')
        self.telescope_name = UVProperty(description='name of telescope '
                                         '(string)', form='str')
        self.instrument = UVProperty(description='receiver or backend.', form='str')
        self.latitude = AngleUVProperty(description='latitude of telescope, '
                                        'units radians', expected_type=np.float,
                                        tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians
        self.longitude = AngleUVProperty(description='longitude of telescope, '
                                         'units degrees', expected_type=np.float,
                                         tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians
        self.altitude = UVProperty(description='altitude of telescope, '
                                   'units meters', expected_type=np.float,
                                   tols=1e-3)  # 1 mm
        self.history = UVProperty(description='string of history, units '
                                  'English', form='str')

        desc = ('epoch year of the phase applied to the data (eg 2000)')
        self.phase_center_epoch = UVProperty(description=desc, expected_type=np.float)

        # --- antenna information ----
        desc = ('number of antennas with data present. May be smaller ' +
                'than the number of antennas in the array')
        self.Nants_data = UVProperty(description=desc)
        desc = ('number of antennas in the array. May be larger ' +
                'than the number of antennas with data')
        self.Nants_telescope = UVProperty(description=desc)
        desc = ('list of antenna names, dimensions (Nants_telescope), '
                'indexed by self.ant_1_array, self.ant_2_array, '
                'self.antenna_indices. There must be one '
                'entry here for each unique entry in self.ant_1_array and '
                'self.ant_2_array, but there may be extras as well.')
        self.antenna_names = UVProperty(description=desc, form=('Nants_telescope',),
                                        expected_type=str)

        desc = ('integer index into antenna_names, dimensions '
                '(Nants_telescope). There must be one '
                'entry here for each unique entry in self.ant_1_array and '
                'self.ant_2_array, but there may be extras as well.')
        self.antenna_indices = UVProperty(description=desc, form=('Nants_telescope',))

        # -------- extra, non-required properties ----------
        desc = ('any user supplied extra keywords, type=dict')
        self.extra_keywords = ExtraKeywordUVProperty(description=desc)

        self.dateobs = UVProperty(required=False,
                                  description='date of observation')

        desc = ('coordinate frame for antenna positions '
                '(eg "ITRF" -also google ECEF). NB: ECEF has x running '
                'through long=0 and z through the north pole')
        self.xyz_telescope_frame = UVProperty(required=False, description=desc,
                                              spoof_val='ITRF', form='str')

        self.x_telescope = UVProperty(required=False,
                                      description='x coordinates of array '
                                      'center in meters in coordinate frame',
                                      spoof_val=0,
                                      tols=1e-3)  # 1 mm
        self.y_telescope = UVProperty(required=False,
                                      description='y coordinates of array '
                                      'center in meters in coordinate frame',
                                      spoof_val=0,
                                      tols=1e-3)  # 1 mm
        self.z_telescope = UVProperty(required=False,
                                      description='z coordinates of array '
                                      'center in meters in coordinate frame',
                                      spoof_val=0,
                                      tols=1e-3)  # 1 mm
        desc = ('array giving coordinates of antennas relative to '
                '{x,y,z}_telescope in the same frame, (Nants_telescope, 3)')
        self.antenna_positions = AntPositionUVProperty(required=False,
                                                       description=desc,
                                                       form=('Nants_telescope', 3),
                                                       tols=1e-3)  # 1 mm

        desc = ('ra of zenith. units: radians, shape (Nblts)')
        self.zenith_ra = AngleUVProperty(required=False, description=desc,
                                         form=('Nblts',),
                                         tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians

        desc = ('dec of zenith. units: radians, shape (Nblts)')
        # in practice, dec of zenith will never change; does not need to
        #  be shape Nblts
        self.zenith_dec = AngleUVProperty(required=False, description=desc,
                                          form=('Nblts',),
                                          tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians

        desc = ('right ascension of phase center (see uvw_array), '
                'units radians')
        self.phase_center_ra = AngleUVProperty(required=False,
                                               description=desc,
                                               tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians

        desc = ('declination of phase center (see uvw_array), '
                'units radians')
        self.phase_center_dec = AngleUVProperty(required=False,
                                                description=desc,
                                                tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians

        # --- other stuff ---
        # the below are copied from AIPS memo 117, but could be revised to
        # merge with other sources of data.
        self.GST0 = UVProperty(required=False,
                               description='Greenwich sidereal time at '
                               'midnight on reference date', spoof_val=0.0)
        self.RDate = UVProperty(required=False,
                                description='date for which the GST0 or '
                                'whatever... applies', spoof_val='')
        self.earth_omega = UVProperty(required=False,
                                      description='earth\'s rotation rate '
                                      'in degrees per day', spoof_val=360.985)
        self.DUT1 = UVProperty(required=False,
                               description='DUT1 (google it) AIPS 117 '
                               'calls it UT1UTC', spoof_val=0.0)
        self.TIMESYS = UVProperty(required=False,
                                  description='We only support UTC',
                                  spoof_val='UTC', form='str')

        desc = ('FHD thing we do not understand, something about the time '
                'at which the phase center is normal to the chosen UV plane '
                'for phasing')
        self.uvplane_reference_time = UVProperty(required=False,
                                                 description=desc,
                                                 spoof_val=0)

    def property_iter(self):
        attribute_list = [a for a in dir(self) if not a.startswith('__') and
                          not callable(getattr(self, a))]
        prop_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if isinstance(attr, UVProperty):
                prop_list.append(a)
        for a in prop_list:
            yield a

    def required_property_iter(self):
        attribute_list = [a for a in dir(self) if not a.startswith('__') and
                          not callable(getattr(self, a))]
        required_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if isinstance(attr, UVProperty):
                if attr.required:
                    required_list.append(a)
        for a in required_list:
            yield a

    def extra_property_iter(self):
        attribute_list = [a for a in dir(self) if not a.startswith('__') and
                          not callable(getattr(self, a))]
        extra_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if isinstance(attr, UVProperty):
                if not attr.required:
                    extra_list.append(a)
        for a in extra_list:
            yield a

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # only check that required properties are identical
            isequal = True
            for p in self.required_property_iter():
                self_prop = getattr(self, p)
                other_prop = getattr(other, p)
                if self_prop != other_prop:
                    print('property {pname} does not match. Left is {lval} '
                          'and right is {rval}'.
                          format(pname=p, lval=str(self_prop.value),
                                 rval=str(other_prop.value)))
                    isequal = False
            return isequal
        else:
            print('Classes do not match')
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def baseline_to_antnums(self, baseline):
        if self.Nants_telescope.value > 2048:
            raise StandardError('error Nants={Nants}>2048 not '
                                'supported'.format(Nants=self.Nants_telescope.value))
        if np.min(baseline) > 2**16:
            i = (baseline - 2**16) % 2048 - 1
            j = (baseline - 2**16 - (i + 1)) / 2048 - 1
        else:
            i = (baseline) % 256 - 1
            j = (baseline - (i + 1)) / 256 - 1
        return np.int32(i), np.int32(j)

    def antnums_to_baseline(self, i, j, attempt256=False):
        # set the attempt256 keyword to True to (try to) use the older
        # 256 standard used in many uvfits files
        # (will use 2048 standard if there are more than 256 antennas)
        i, j = np.int64((i, j))
        if self.Nants_telescope.value > 2048:
            raise StandardError('cannot convert i,j to a baseline index '
                                'with Nants={Nants}>2048.'
                                .format(Nants=self.Nants_telescope))
        if attempt256:
            if (np.max(i) < 255 and np.max(j) < 255):
                return 256 * (j + 1) + (i + 1)
            else:
                print('Max antnums are {} and {}'.format(np.max(i), np.max(j)))
                message = 'antnums_to_baseline: found > 256 antennas, using ' \
                          '2048 baseline indexing. Beware compatibility ' \
                          'with CASA etc'
                warnings.warn(message)

        return np.int64(2048 * (j + 1) + (i + 1) + 2**16)

    def set_LatLonAlt_from_XYZ(self, overwrite=False):
        if (self.xyz_telescope_frame.value == "ITRF" and
            None not in (self.x_telescope.value,
                         self.y_telescope.value,
                         self.z_telescope.value)):
            # see wikipedia geodetic_datum and Datum transformations of
            # GPS positions PDF in docs folder
            gps_b = 6356752.31424518
            gps_a = 6378137
            e_squared = 6.69437999014e-3
            e_prime_squared = 6.73949674228e-3
            gps_p = np.sqrt(self.x_telescope.value**2 +
                            self.y_telescope.value**2)
            gps_theta = np.arctan2(self.z_telescope.value * gps_a,
                                   gps_p * gps_b)
            if self.latitude.value is None or overwrite:
                self.latitude.value = np.arctan2(self.z_telescope.value +
                                                 e_prime_squared * gps_b *
                                                 np.sin(gps_theta)**3,
                                                 gps_p - e_squared * gps_a *
                                                 np.cos(gps_theta)**3)

            if self.longitude.value is None or overwrite:
                self.longitude.value = np.arctan2(self.y_telescope.value,
                                                  self.x_telescope.value)
            gps_N = gps_a / np.sqrt(1 - e_squared *
                                    np.sin(self.latitude.value)**2)
            if self.altitude.value is None or overwrite:
                self.altitude.value = ((gps_p / np.cos(self.latitude.value)) -
                                       gps_N)
        else:
            raise ValueError('No x, y or z_telescope value assigned or '
                             'xyz_telescope_frame is not "ITRF"')

    def set_XYZ_from_LatLonAlt(self, overwrite=False):
        # check that the coordinates we need actually exist
        if None not in (self.latitude.value, self.longitude.value,
                        self.altitude.value):
            # see wikipedia geodetic_datum and Datum transformations of
            # GPS positions PDF in docs folder
            gps_b = 6356752.31424518
            gps_a = 6378137
            e_squared = 6.69437999014e-3
            e_prime_squared = 6.73949674228e-3
            gps_N = gps_a / np.sqrt(1 - e_squared *
                                    np.sin(self.latitude.value)**2)
            if self.x_telescope.value is None or overwrite:
                self.x_telescope.value = ((gps_N + self.altitude.value) *
                                          np.cos(self.latitude.value) *
                                          np.cos(self.longitude.value))
            if self.y_telescope.value is None or overwrite:
                self.y_telescope.value = ((gps_N + self.altitude.value) *
                                          np.cos(self.latitude.value) *
                                          np.sin(self.longitude.value))
            if self.z_telescope.value is None or overwrite:
                self.z_telescope.value = ((gps_b**2 / gps_a**2 * gps_N +
                                          self.altitude.value) *
                                          np.sin(self.latitude.value))
        else:
            raise ValueError('lat, lon or altitude not found')

    def set_lsts_from_time_array(self):
        lsts = []
        curtime = self.time_array.value[0]
        for ind, jd in enumerate(self.time_array.value):
            if ind == 0 or not np.isclose(jd, curtime, atol=1e-6, rtol=1e-12):
                curtime = jd
                t = Time(jd, format='jd', location=(self.longitude.degrees(),
                                                    self.latitude.degrees()))
                t.delta_ut1_utc = iers_a.ut1_utc(t)
            lsts.append(t.sidereal_time('apparent').radian)
        self.lst_array.value = np.array(lsts)
        return True

    def juldate2ephem(self, num):
        """Convert Julian date to ephem date, measured from noon, Dec. 31, 1899."""
        return ephem.date(num - 2415020.)

    def phase(self, ra=None, dec=None, epoch=ephem.J2000, time=None):
        # phase drift scan data to a single ra/dec at the set epoch
        # or time in jd (i.e. ra/dec of zenith at that time in current epoch).
        # ra/dec should be in radians.
        # epoch should be an ephem date, measured from noon Dec. 31, 1899.
        # will not phase already phased data.
        if (self.phase_center_ra.value is not None or
                self.phase_center_dec.value is not None):
            raise ValueError('The data is already phased; can only phase ' +
                             'drift scanning data.')

        obs = ephem.Observer()
        # obs inits with default values for parameters -- be sure to replace them
        obs.lat = self.latitude.value
        obs.lon = self.longitude.value
        if ra is not None and dec is not None and epoch is not None and time is None:
            pass

        elif ra is None and dec is None and time is not None:
            # NB if phasing to a time, epoch does not need to be None, but it is ignored
            obs.date, obs.epoch = self.juldate2ephem(time), self.juldate2ephem(time)

            ra = self.longitude.value - obs.sidereal_time()
            dec = self.latitude.value
            epoch = time

        else:
            raise ValueError('Need to define either ra/dec/epoch or time ' +
                             '(but not both).')

        # create a pyephem object for the phasing position
        precess_pos = ephem.FixedBody()
        precess_pos._ra = ra
        precess_pos._dec = dec
        precess_pos._epoch = epoch

        # calculate RA/DEC in J2000 and write to object
        obs.date, obs.epoch = ephem.J2000, ephem.J2000
        precess_pos.compute(obs)
        self.phase_center_ra.value = precess_pos.ra
        self.phase_center_dec.value = precess_pos.dec

        for ind, jd in enumerate(self.time_array.value):
            # calculate ra/dec of phase center in current epoch
            obs.date, obs.epoch = self.juldate2ephem(jd), self.juldate2ephem(jd)
            precess_pos.compute(obs)
            ra, dec = precess_pos.ra, precess_pos.dec

            # generate rotation matrices
            m0 = a.coord.top2eq_m(self.lst_array.value[ind] - obs.sidereal_time(), self.latitude.value)
            m1 = a.coord.eq2top_m(self.lst_array.value[ind] - ra, dec)

            # rotate and write uvws
            uvw = self.uvw_array.value[:, ind]
            uvw = np.dot(m0, uvw)
            uvw = np.dot(m1, uvw)
            self.uvw_array.value[:, ind] = uvw

            # calculate data and apply phasor
            w_lambda = uvw[2] / const.c.to('m/s').value * self.freq_array.value
            phs = np.exp(-1j * 2 * np.pi * w_lambda)
            phs.shape += (1,)
            self.data_array.value[ind] *= phs

        del(obs)
        return True

    def check(self, run_sanity_check=True):
        # loop through all required properties, make sure that they are filled
        for p in self.required_property_iter():
            prop = getattr(self, p)
            # Check required property exists
            if prop.value is None:
                raise ValueError('Required UVProperty ' + p +
                                 ' has not been set.')

            # Check required property size
            esize = prop.expected_size(self)
            if esize is None:
                raise ValueError('Required UVProperty ' + p +
                                 ' expected size is not defined.')
            elif esize == 'str':
                # Check that it's a string
                if not isinstance(prop.value, str):
                    raise ValueError('UVProperty ' + p + 'expected to be '
                                     'string, but is not')
            else:
                # Check the size of the property value. Note that np.shape
                # returns an empty tuple for single numbers. esize should do the same.
                if not np.shape(prop.value) == esize:
                    raise ValueError('UVProperty ' + p + 'is not expected size.')
                if esize == ():
                    # Single element
                    if not isinstance(prop.value, prop.expected_type):
                        raise ValueError('UVProperty ' + p + ' is not the appropriate'
                                         ' type. Is: ' + str(type(prop.value)) +
                                         '. Should be: ' + str(prop.expected_type))
                else:
                    if isinstance(prop.value, list):
                        # List needs to be handled differently than array (I think)
                        if not isinstance(prop.value[0], prop.expected_type):
                            raise ValueError('UVProperty ' + p + ' is not the'
                                             ' appropriate type. Is: ' +
                                             str(type(prop.value[0])) + '. Should'
                                             ' be: ' + str(prop.expected_type))
                    else:
                        # Array
                        if not isinstance(prop.value.item(0), prop.expected_type):
                            raise ValueError('UVProperty ' + p + ' is not the appropriate'
                                             ' type. Is: ' + str(prop.value.dtype) +
                                             '. Should be: ' + str(prop.expected_type))

            if run_sanity_check:
                if not prop.sanity_check():
                    raise ValueError('UVProperty ' + p + ' has insane values.')

        return True

    def write(self, filename, spoof_nonessential=False, force_phase=False,
              run_check=True, run_sanity_check=True):
        if run_check:
            self.check(run_sanity_check=run_sanity_check)
        status = False
        # filename ending in .uvfits gets written as a uvfits
        if filename.endswith('.uvfits'):
            status = self.write_uvfits(filename, spoof_nonessential=spoof_nonessential, force_phase=force_phase)
        return status

    def read(self, filename, file_type, use_model=False, run_check=True,
             run_sanity_check=True):
        """
        General read function which calls file_type specific read functions
        Inputs:
            filename: string or list of strings
                May be a file name, directory name or a list of file names
                depending on file_type
            file_type: string
                Must be a supported type, see self.supported_file_types
        """
        if file_type not in self.supported_file_types:
            raise ValueError('file_type must be one of ' +
                             ' '.join(self.supported_file_types))
        if file_type == 'uvfits':
            status = self.read_uvfits(filename, run_check=run_check,
                                      run_sanity_check=run_sanity_check)
        elif file_type == 'miriad':
            status = self.read_miriad(filename, run_check=run_check,
                                      run_sanity_check=run_sanity_check)
        elif file_type == 'fhd':
            status = self.read_fhd(filename, use_model=use_model, run_check=run_check,
                                   run_sanity_check=run_sanity_check)
        if run_check:
            self.check(run_sanity_check=run_sanity_check)
        return status

    def convert_from_uvfits(self, uvfits_obj):
        for p in uvfits_obj.property_iter():
            prop = getattr(uvfits_obj, p)
            setattr(self, p, prop)

    def convert_to_uvfits(self):
        uvfits_obj = uvdata.uvfits.UVFITS()
        for p in self.property_iter():
            prop = getattr(self, p)
            setattr(uvfits_obj, p, prop)
        return uvfits_obj

    def read_uvfits(self, filename):
        uvfits_obj = uvdata.uvfits.UVFITS()
        ret_val = uvfits_obj.read_uvfits(filename)
        self.convert_from_uvfits(uvfits_obj)
        return ret_val

    def write_uvfits(self, filename, spoof_nonessential=False,
                     force_phase=False):
        uvfits_obj = self.convert_to_uvfits()
        ret_val = uvfits_obj.write_uvfits(filename,
                                          spoof_nonessential=spoof_nonessential,
                                          force_phase=force_phase)
        return ret_val

    def read_fhd(self, filelist, use_model=False, run_check=True,
                 run_sanity_check=True):
        """
        Read in fhd visibility save files
            filelist: list
                list of files containing fhd-style visibility data.
                Must include at least one polarization file, a params file and
                a flag file.
        """

        datafiles = {}
        params_file = None
        flags_file = None
        settings_file = None
        if use_model:
            data_name = '_vis_model_'
        else:
            data_name = '_vis_'
        for file in filelist:
            if file.lower().endswith(data_name + 'xx.sav'):
                datafiles['xx'] = xx_datafile = file
            elif file.lower().endswith(data_name + 'yy.sav'):
                datafiles['yy'] = yy_datafile = file
            elif file.lower().endswith(data_name + 'xy.sav'):
                datafiles['xy'] = xy_datafile = file
            elif file.lower().endswith(data_name + 'yx.sav'):
                datafiles['yx'] = yx_datafile = file
            elif file.lower().endswith('_params.sav'):
                params_file = file
            elif file.lower().endswith('_flags.sav'):
                flags_file = file
            elif file.lower().endswith('_settings.txt'):
                settings_file = file
            else:
                continue

        if len(datafiles) < 1:
            raise StandardError('No data files included in file list')
        if params_file is None:
            raise StandardError('No params file included in file list')
        if flags_file is None:
            raise StandardError('No flags file included in file list')
        if settings_file is None:
            warnings.warn('No settings file included in file list')

        # TODO: add checking to make sure params, flags and datafiles are
        # consistent with each other

        vis_data = {}
        for pol, file in datafiles.iteritems():
            this_dict = readsav(file, python_dict=True)
            if use_model:
                vis_data[pol] = this_dict['vis_model_ptr']
            else:
                vis_data[pol] = this_dict['vis_ptr']
            this_obs = this_dict['obs']
            data_dimensions = vis_data[pol].shape

        obs = this_obs
        bl_info = obs['BASELINE_INFO'][0]
        meta_data = obs['META_DATA'][0]
        astrometry = obs['ASTR'][0]
        fhd_pol_list = []
        for pol in obs['POL_NAMES'][0]:
            fhd_pol_list.append(pol.decode("utf-8").lower())

        params_dict = readsav(params_file, python_dict=True)
        params = params_dict['params']

        flags_dict = readsav(flags_file, python_dict=True)
        flag_data = {}
        for index, f in enumerate(flags_dict['flag_arr']):
            flag_data[fhd_pol_list[index]] = f

        self.Ntimes.value = int(obs['N_TIME'][0])
        self.Nbls.value = int(obs['NBASELINES'][0])
        self.Nblts.value = data_dimensions[0]
        self.Nfreqs.value = int(obs['N_FREQ'][0])
        self.Npols.value = len(vis_data.keys())
        self.Nspws.value = 1
        self.spw_array.value = np.array([0])
        self.vis_units.value = 'JY'

        lin_pol_order = ['xx', 'yy', 'xy', 'yx']
        linear_pol_dict = dict(zip(lin_pol_order, np.arange(5, 9) * -1))
        pol_list = []
        for pol in lin_pol_order:
            if pol in vis_data:
                pol_list.append(linear_pol_dict[pol])
        self.polarization_array.value = np.asarray(pol_list)

        self.data_array.value = np.zeros((self.Nblts.value, self.Nspws.value,
                                          self.Nfreqs.value, self.Npols.value),
                                         dtype=np.complex_)
        self.nsample_array.value = np.zeros((self.Nblts.value,
                                             self.Nspws.value,
                                             self.Nfreqs.value,
                                             self.Npols.value),
                                            dtype=np.float_)
        self.flag_array.value = np.zeros((self.Nblts.value, self.Nspws.value,
                                          self.Nfreqs.value, self.Npols.value),
                                         dtype=np.bool_)
        for pol, vis in vis_data.iteritems():
            pol_i = pol_list.index(linear_pol_dict[pol])
            self.data_array.value[:, 0, :, pol_i] = vis
            self.flag_array.value[:, 0, :, pol_i] = flag_data[pol] <= 0
            self.nsample_array.value[:, 0, :, pol_i] = np.abs(flag_data[pol])

        # In FHD, uvws are in seconds not meters!
        self.uvw_array.value = np.zeros((3, self.Nblts.value))
        self.uvw_array.value[0, :] = params['UU'][0] * const.c.to('m/s').value
        self.uvw_array.value[1, :] = params['VV'][0] * const.c.to('m/s').value
        self.uvw_array.value[2, :] = params['WW'][0] * const.c.to('m/s').value

        # bl_info.JDATE (a vector of length Ntimes) is the only safe date/time
        # to use in FHD files.
        # (obs.JD0 (float) and params.TIME (vector of length Nblts) are
        #   context dependent and are not safe
        #   because they depend on the phasing of the visibilities)
        # the values in bl_info.JDATE are the JD for each integration.
        # We need to expand up to Nblts.
        int_times = bl_info['JDATE'][0]
        bin_offset = bl_info['BIN_OFFSET'][0]
        self.time_array.value = np.zeros(self.Nblts.value)
        for ii in range(0, self.Ntimes.value):
            if ii < (self.Ntimes.value - 1):
                self.time_array.value[bin_offset[ii]:bin_offset[ii + 1]] = int_times[ii]
            else:
                self.time_array.value[bin_offset[ii]:] = int_times[ii]

        # Note that FHD antenna arrays are 1-indexed so we subtract 1
        # to get 0-indexed arrays
        self.ant_1_array.value = bl_info['TILE_A'][0] - 1
        self.ant_2_array.value = bl_info['TILE_B'][0] - 1

        self.Nants_data.value = np.max([len(np.unique(self.ant_1_array.value)),
                                        len(np.unique(self.ant_2_array.value))])

        self.antenna_names.value = bl_info['TILE_NAMES'][0].tolist()
        self.Nants_telescope.value = len(self.antenna_names.value)
        self.antenna_indices.value = np.arange(self.Nants_telescope.value)

        self.baseline_array.value = \
            self.antnums_to_baseline(self.ant_1_array.value,
                                     self.ant_2_array.value)

        self.freq_array.value = np.zeros((self.Nspws.value, self.Nfreqs.value),
                                         dtype=np.float_)
        self.freq_array.value[0, :] = bl_info['FREQ'][0]

        if not np.isclose(obs['OBSRA'][0], obs['PHASERA'][0]) or \
                not np.isclose(obs['OBSDEC'][0], obs['PHASEDEC'][0]):
            warnings.warn('These visibilities may have been phased '
                          'improperly -- without changing the uvw locations')

        self.phase_center_ra.set_degrees(float(obs['OBSRA'][0]))
        self.phase_center_dec.set_degrees(float(obs['OBSDEC'][0]))

        # this is generated in FHD by subtracting the JD of neighboring
        # integrations. This can have limited accuracy, so it can be slightly
        # off the actual value.
        # (e.g. 1.999426... rather than 2)
        self.integration_time.value = float(obs['TIME_RES'][0])
        self.channel_width.value = float(obs['FREQ_RES'][0])

        # # --- observation information ---
        self.telescope_name.value = str(obs['INSTRUMENT'][0].decode())

        # This is a bit of a kludge because nothing like object_name exists
        # in FHD files.
        # At least for the MWA, obs.ORIG_PHASERA and obs.ORIG_PHASEDEC specify
        # the field the telescope was nominally pointing at
        # (May need to be revisited, but probably isn't too important)
        self.object_name.value = 'Field RA(deg): ' + \
                                 str(obs['ORIG_PHASERA'][0]) + \
                                 ', Dec:' + str(obs['ORIG_PHASEDEC'][0])
        # For the MWA, this can sometimes be converted to EoR fields
        if self.telescope_name.value.lower() == 'mwa':
            if np.isclose(obs['ORIG_PHASERA'][0], 0) and \
                    np.isclose(obs['ORIG_PHASEDEC'][0], -27):
                object_name = 'EoR 0 Field'

        self.instrument.value = self.telescope_name.value
        self.latitude.set_degrees(float(obs['LAT'][0]))
        self.longitude.set_degrees(float(obs['LON'][0]))
        self.altitude.value = float(obs['ALT'][0])

        self.set_lsts_from_time_array()

        # Use the first integration time here
        self.dateobs.value = min(self.time_array.value)

        # history: add the first few lines from the settings file
        if settings_file is not None:
            history_list = ['fhd settings info']
            with open(settings_file) as f:
                # TODO Make this reading more robust.
                head = list(islice(f, 11))
            for line in head:
                newline = ' '.join(str.split(line))
                if not line.startswith('##'):
                    history_list.append(newline)
            self.history.value = '    '.join(history_list)
        else:
            self.history.value = ''

        self.phase_center_epoch.value = astrometry['EQUINOX'][0]

        # TODO Once FHD starts reading and saving the antenna table info from
        #    uvfits, that information should be read into the following optional
        #    parameters:
        # 'xyz_telescope_frame'
        # 'x_telescope'
        # 'y_telescope'
        # 'z_telescope'
        # 'antenna_positions'
        # 'GST0'
        # 'RDate'
        # 'earth_omega'
        # 'DUT1'
        # 'TIMESYS'

        # check if object has all required uv_properties set
        if run_check:
            self.check(run_sanity_check=run_sanity_check)
        return True

    def miriad_pol_to_ind(self, pol):
        if self.polarization_array.value is None:
            raise(ValueError, "Can't index polarization {p} because "
                  "polarization_array is not set".format(p=pol))
        pol_ind = np.argwhere(self.polarization_array.value == pol)
        if len(pol_ind) != 1:
            raise(ValueError, "multiple matches for pol={pol} in "
                  "polarization_array".format(pol=pol))
        return pol_ind

    def read_miriad(self, filepath, FLEXIBLE_OPTION=True, run_check=True,
                    run_sanity_check=True):
        # map uvdata attributes to miriad data values
        # those which we can get directly from the miriad file
        # (some, like n_times, have to be calculated)
        if not os.path.exists(filepath):
            raise(IOError, filepath + ' not found')
        uv = a.miriad.UV(filepath)

        miriad_header_data = {'Nfreqs': 'nchan',
                              'Npols': 'npol',
                              # 'Nspws': 'nspec',  # not always available
                              'integration_time': 'inttime',
                              'channel_width': 'sdf',  # in Ghz!
                              'object_name': 'source',
                              'telescope_name': 'telescop',
                              # same as telescope_name for now
                              'instrument': 'telescop',
                              'latitude': 'latitud',
                              'longitude': 'longitu',  # in units of radians
                              # (get the first time in the ever changing header)
                              'dateobs': 'time',
                              # 'history': 'history',
                              'Nants_telescope': 'nants',
                              'phase_center_epoch': 'epoch',
                              'antenna_positions': 'antpos',  # take deltas
                              }
        for item in miriad_header_data:
            if isinstance(uv[miriad_header_data[item]], str):
                header_value = uv[miriad_header_data[item]].replace('\x00', '')
            else:
                header_value = uv[miriad_header_data[item]]
            getattr(self, item).value = header_value

        if self.telescope_name.value.startswith('PAPER') and \
                self.altitude.value is None:
            print "WARNING: Altitude not found for telescope PAPER. "
            print "setting to 1100m"
            self.altitude.value = 1100.

        self.history.value = uv['history']
        try:
            self.antenna_positions.value = \
                self.antenna_positions.value.reshape(3, self.Nants_telescope.value).T
        except(ValueError):
            self.antenna_positions.value = None
        self.channel_width.value *= 1e9  # change from GHz to Hz

        # read through the file and get the data
        _source = uv['source'] # check source of initial visibility
        data_accumulator = {}
        for (uvw, t, (i, j)), d, f in uv.all(raw=True):
            # control for the case of only a single spw not showing up in
            # the dimension
            if len(d.shape) == 1:
                d.shape = (1,) + d.shape
                self.Nspws.value = d.shape[0]
                self.spw_array.value = np.arange(self.Nspws.value)
            else:
                raise(ValueError, """Sorry.  Files with more than one spectral
                      window (spw) are not yet supported. A great
                      project for the interested student!""")
            try:
                cnt = uv['cnt']
            except(KeyError):
                cnt = np.ones(d.shape, dtype=np.int)
            zenith_ra = uv['ra']
            zenith_dec = uv['dec']
            lst = uv['lst']
            source = uv['source']
            if source != _source:
                raise(ValueError, """This appears to be a multi source file, which
                      is not supported.""")
            else:
                _source = source

            try:
                data_accumulator[uv['pol']].append([uvw, t, i, j, d, f, cnt,
                                                    zenith_ra, zenith_dec])
            except(KeyError):
                data_accumulator[uv['pol']] = [[uvw, t, i, j, d, f, cnt,
                                                zenith_ra, zenith_dec]]
                # NB: flag types in miriad are usually ints
        self.polarization_array.value = np.sort(data_accumulator.keys())
        if len(self.polarization_array.value) > self.Npols.value:
            print "WARNING: npols={npols} but found {l} pols in data file".format(
                npols=self.Npols.value, l=len(self.polarization_array.value))
        if FLEXIBLE_OPTION:
            # makes a data_array (and flag_array) of zeroes to be filled in by
            #   data values
            # any missing data will have zeros

            # use set to get the unique list of all times ever listed in the file
            # iterate over polarizations and all spectra (bls and times) in two
            # nested loops, then flatten into a single vector, then set
            # then list again.
            times = list(set(
                np.ravel([[k[1] for k in d] for d in data_accumulator.values()])))
            times = np.sort(times)

            ant_i_unique = list(set(
                np.ravel([[k[2] for k in d] for d in data_accumulator.values()])))
            ant_j_unique = list(set(
                np.ravel([[k[3] for k in d] for d in data_accumulator.values()])))

            self.Nants_data.value = max(len(ant_i_unique), len(ant_j_unique))
            self.antenna_indices.value = np.arange(self.Nants_telescope.value)
            self.antenna_names.value = self.antenna_indices.value.astype(str).tolist()
            # form up a grid which indexes time and baselines along the 'long'
            # axis of the visdata array
            t_grid = []
            ant_i_grid = []
            ant_j_grid = []
            for t in times:
                for ant_i in ant_i_unique:
                    for ant_j in ant_j_unique:
                        t_grid.append(t)
                        ant_i_grid.append(ant_i)
                        ant_j_grid.append(ant_j)
            ant_i_grid = np.array(ant_i_grid)
            ant_j_grid = np.array(ant_j_grid)
            t_grid = np.array(t_grid)

            # set the data sizes
            self.Nblts.value = len(t_grid)
            self.Ntimes.value = len(times)
            self.time_array.value = t_grid
            self.ant_1_array.value = ant_i_grid
            self.ant_2_array.value = ant_j_grid

            self.baseline_array.value = self.antnums_to_baseline(ant_i_grid,
                                                                 ant_j_grid)
            self.Nbls.value = len(np.unique(self.baseline_array.value))
            # slot the data into a grid
            self.data_array.value = np.zeros((self.Nblts.value,
                                              self.Nspws.value,
                                              self.Nfreqs.value,
                                              self.Npols.value),
                                             dtype=np.complex64)
            self.flag_array.value = np.ones(self.data_array.value.shape, dtype=np.bool)
            self.uvw_array.value = np.zeros((3, self.Nblts.value))
            # NOTE: Using our lst calculator, which uses astropy,
            # instead of aipy values which come from pyephem.
            # The differences are of order 5 seconds.
            self.set_lsts_from_time_array()
            self.nsample_array.value = np.ones(self.data_array.value.shape,
                                               dtype=np.int)
            self.freq_array.value = (np.arange(self.Nfreqs.value) *
                                     self.channel_width.value +
                                     uv['sfreq'] * 1e9)
            # Tile freq_array to dimensions (Nspws, Nfreqs).
            # Currently does not actually support Nspws>1!
            self.freq_array.value = np.tile(self.freq_array.value,
                                            (self.Nspws.value, 1))

            ra_list = np.zeros(self.Nblts.value)
            dec_list = np.zeros(self.Nblts.value)

            for pol, data in data_accumulator.iteritems():
                pol_ind = self.miriad_pol_to_ind(pol)
                for ind, d in enumerate(data):
                    t, ant_i, ant_j = d[1], d[2], d[3]
                    blt_index = np.where(np.logical_and(np.logical_and(t == t_grid,
                                                                       ant_i == ant_i_grid),
                                                        ant_j == ant_j_grid))[0].squeeze()
                    self.data_array.value[blt_index, :, :, pol_ind] = d[4]
                    self.flag_array.value[blt_index, :, :, pol_ind] = d[5]
                    self.nsample_array.value[blt_index, :, :, pol_ind] = d[6]

                    # because there are uvws/ra/dec for each pol, and one pol may not
                    # have that visibility, we collapse along the polarization
                    # axis but avoid any missing visbilities
                    uvw = d[0] * const.c.to('m/ns').value
                    uvw.shape = (1, 3)
                    self.uvw_array.value[:, blt_index] = uvw
                    ra_list[blt_index] = d[7]
                    dec_list[blt_index] = d[8]

            # check if ra is constant throughout file; if it is,
            # file is tracking if not, file is drift scanning
            if np.isclose(np.mean(np.diff(ra_list)), 0.):
                self.phase_center_ra.value = ra_list[0]
                self.phase_center_dec.value = dec_list[0]
            else:
                self.zenith_ra.value = ra_list
                self.zenith_dec.value = dec_list

            # enforce drift scan/ phased convention
            # convert lat/lon to x/y/z_telescope
            #    LLA to ECEF (see pdf in docs)

        if not FLEXIBLE_OPTION:
            pass
            # this option would accumulate things requiring
            # baselines and times are sorted in the same
            #          order for each polarization
            # and that there are the same number of baselines
            #          and pols per timestep
            # TBD impliment

        # NOTES:
        # pyuvdata is natively 0 indexed as is miriad
        # miriad uses the same pol2num standard as aips/casa

        self.vis_units.value = 'UNCALIB'  # assume no calibration

        # things that might not be required?
        # 'GST0'  : None,
        # 'RDate'  : None,  # date for which the GST0 or whatever... applies
        # 'earth_omega'  : 360.985,
        # 'DUT1'  : 0.0,        # DUT1 (google it) AIPS 117 calls it UT1UTC
        # 'TIMESYS'  : 'UTC',   # We only support UTC

        #

        # Phasing rule: if alt/az is set and ra/dec are None,
        #  then its a drift scan

        # check if object has all required uv_properties set
        if run_check:
            self.check(run_sanity_check=run_sanity_check)
        return True

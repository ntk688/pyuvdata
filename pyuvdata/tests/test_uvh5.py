# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for HDF5 object

"""
from __future__ import absolute_import, division, print_function

import os
import six
import numpy as np
import pytest
from astropy.time import Time
import h5py

from pyuvdata import UVData, uvh5
import pyuvdata.utils as uvutils
from pyuvdata.data import DATA_PATH
import pyuvdata.tests as uvtest
from pyuvdata.uvh5 import _hera_corr_dtype


# ignore common file-read warnings
pytestmark = [
    pytest.mark.filterwarnings("ignore:Altitude is not present in Miriad"),
    pytest.mark.filterwarnings("ignore:Telescope EVLA is not"),
]


@pytest.fixture(scope="function")
def uv_miriad():
    # read in a miriad test file
    uv_miriad = UVData()
    miriad_filename = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA")
    uv_miriad.read_miriad(miriad_filename)
    yield uv_miriad

    # clean up when done
    del uv_miriad

    return


@pytest.fixture(scope="function")
def uv_uvfits():
    # read in a uvfits test file
    uv_uvfits = UVData()
    uvfits_filename = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")
    uv_uvfits.read_uvfits(uvfits_filename)
    yield uv_uvfits

    # clean up when done
    del uv_uvfits

    return


@pytest.fixture(scope="function")
def uv_uvh5():
    # read in a uvh5 test file
    uv_uvh5 = UVData()
    uvh5_filename = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')
    uv_uvh5.read_uvh5(uvh5_filename)
    yield uv_uvh5

    # clean up when done
    del uv_uvh5

    return


@pytest.fixture(scope="function")
def uv_partial_write():
    # convert a uvfits file to uvh5, cutting down the amount of data
    uv_uvfits = UVData()
    uvfits_filename = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")
    uv_uvfits.read_uvfits(uvfits_filename)
    uv_uvfits.select(antenna_nums=[3, 7, 24])
    uv_uvfits.lst_array = uvutils.get_lst_for_time(
        uv_uvfits.time_array, *uv_uvfits.telescope_location_lat_lon_alt_degrees
    )

    testfile = os.path.join(DATA_PATH, "test", "outtest.uvh5")
    uv_uvfits.write_uvh5(testfile)
    uv_uvh5 = UVData()
    uv_uvh5.read(testfile)

    yield uv_uvh5

    # clean up when done
    del uv_uvh5
    os.remove(testfile)

    return


def initialize_with_zeros(uvd, filename):
    """
    Make a uvh5 file with all zero values for data-sized arrays.

    This function is a helper function used for tests of partial writing.
    """
    uvd.initialize_uvh5_file(filename, clobber=True)
    data_shape = (uvd.Nblts, 1, uvd.Nfreqs, uvd.Npols)
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    with h5py.File(filename, 'r+') as h5f:
        dgrp = h5f['/Data']
        data_dset = dgrp['visdata']
        flags_dset = dgrp['flags']
        nsample_dset = dgrp['nsamples']
        data_dset = data  # noqa
        flags_dset = flags  # noqa
        nsample_dset = nsamples  # noqa
    return


def initialize_with_zeros_ints(uvd, filename):
    """
    Make a uvh5 file with all zeros for data-sized arrays.

    This function is a helper function used for tests of partial writing with
    integer data types.
    """
    uvd.initialize_uvh5_file(
        filename, clobber=True, data_write_dtype=_hera_corr_dtype
    )
    data_shape = (uvd.Nblts, 1, uvd.Nfreqs, uvd.Npols)
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    with h5py.File(filename, 'r+') as h5f:
        dgrp = h5f['/Data']
        data_dset = dgrp['visdata']
        flags_dset = dgrp['flags']
        nsample_dset = dgrp['nsamples']
        with data_dset.astype(_hera_corr_dtype):
            data_dset[:, :, :, :, 'r'] = data.real
            data_dset[:, :, :, :, 'i'] = data.imag
        flags_dset = flags  # noqa
        nsample_dset = nsamples  # noqa
    return


def test_read_miriad_write_uvh5_read_uvh5(uv_miriad):
    """
    Test a miriad file round trip.
    """
    uv_in = uv_miriad
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_miriad.uvh5')
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)
    assert uv_in == uv_out

    # also test round-tripping phased data
    uv_in.phase_to_time(Time(np.mean(uv_in.time_array), format='jd'))
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)

    assert uv_in == uv_out

    # clean up
    os.remove(testfile)

    return


def test_read_uvfits_write_uvh5_read_uvh5(uv_uvfits):
    """
    Test a uvfits file round trip.
    """
    uv_in = uv_uvfits
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)
    assert uv_in == uv_out

    # also test writing double-precision data_array
    uv_in.data_array = uv_in.data_array.astype(np.complex128)
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)
    assert uv_in == uv_out

    # clean up
    os.remove(testfile)

    return


def test_read_uvh5_errors():
    """
    Test raising errors in read function.
    """
    uv_in = UVData()
    fake_file = os.path.join(DATA_PATH, 'fake_file.uvh5')
    with pytest.raises(IOError) as cm:
        uv_in.read_uvh5(fake_file)
    assert str(cm.value).startswith("{} not found".format(fake_file))

    return


def test_write_uvh5_errors(uv_uvfits):
    """
    Test raising errors in write_uvh5 function.
    """
    uv_in = uv_uvfits
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')
    with open(testfile, 'a'):
        os.utime(testfile, None)

    # assert IOError if file exists
    with pytest.raises(IOError) as cm:
        uv_in.write_uvh5(testfile, clobber=False)
    assert str(cm.value).startswith("File exists; skipping")

    # use clobber=True to write out anyway
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)
    assert uv_in == uv_out

    # clean up
    os.remove(testfile)

    return


def test_uvh5_optional_parameters(uv_uvfits):
    """
    Test reading and writing optional parameters not in sample files.
    """
    uv_in = uv_uvfits
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')

    # set optional parameters
    uv_in.x_orientation = 'east'
    uv_in.antenna_diameters = np.ones_like(uv_in.antenna_numbers) * 1.
    uv_in.uvplane_reference_time = 0

    # reorder_blts
    uv_in.reorder_blts()

    # write out and read back in
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)
    assert uv_in == uv_out

    # test with blt_order = bda as well (single entry in tuple)
    uv_in.reorder_blts(order='bda')

    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)
    assert uv_in == uv_out

    # clean up
    os.remove(testfile)

    return


def test_uvh5_compression_options(uv_uvfits):
    """
    Test writing data with compression filters.
    """
    uv_in = uv_uvfits
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits_compression.uvh5')

    # write out and read back in
    uv_in.write_uvh5(
        testfile,
        clobber=True,
        data_compression="lzf",
        flags_compression=None,
        nsample_compression=None
    )
    uv_out.read(testfile)
    assert uv_in == uv_out

    # clean up
    os.remove(testfile)

    return


def test_uvh5_read_multiple_files(uv_uvfits):
    """
    Test reading multiple uvh5 files.
    """
    uv_in = uv_uvfits
    testfile1 = os.path.join(DATA_PATH, 'test/uv1.uvh5')
    testfile2 = os.path.join(DATA_PATH, 'test/uv2.uvh5')
    uv1 = uv_in.copy()
    uv2 = uv_in.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_uvh5(testfile1, clobber=True)
    uv2.write_uvh5(testfile2, clobber=True)
    uv1.read([testfile1, testfile2])
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(uv_in.history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency axis using'
                                    ' pyuvdata.', uv1.history)
    uv1.history = uv_in.history
    assert uv1 == uv_in

    # clean up
    os.remove(testfile1)
    os.remove(testfile2)

    return


def test_uvh5_read_multiple_files_metadata_only(uv_uvfits):
    """
    Test reading multiple uvh5 files with metadata only.
    """
    uv_in = uv_uvfits
    testfile1 = os.path.join(DATA_PATH, 'test/uv1.uvh5')
    testfile2 = os.path.join(DATA_PATH, 'test/uv2.uvh5')
    uv1 = uv_in.copy()
    uv2 = uv_in.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_uvh5(testfile1, clobber=True)
    uv2.write_uvh5(testfile2, clobber=True)

    uvfits_filename = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")
    uv_full = UVData()
    uv_full.read_uvfits(uvfits_filename, read_data=False)
    uv1.read([testfile1, testfile2], read_data=False)
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency axis using'
                                    ' pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # clean up
    os.remove(testfile1)
    os.remove(testfile2)

    return


def test_uvh5_rea_multiple_files_axis(uv_uvfits):
    """
    Test reading multiple uvh5 files with setting axis.
    """
    uv_in = uv_uvfits
    testfile1 = os.path.join(DATA_PATH, 'test/uv1.uvh5')
    testfile2 = os.path.join(DATA_PATH, 'test/uv2.uvh5')
    uv1 = uv_in.copy()
    uv2 = uv_in.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_uvh5(testfile1, clobber=True)
    uv2.write_uvh5(testfile2, clobber=True)
    uv1.read([testfile1, testfile2], axis="freq")
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(uv_in.history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency axis using'
                                    ' pyuvdata.', uv1.history)
    uv1.history = uv_in.history
    assert uv1 == uv_in

    # clean up
    os.remove(testfile1)
    os.remove(testfile2)

    return


def test_uvh5_partial_read_antennas(uv_uvfits):
    """
    Test reading in only certain antennas from disk.
    """
    uv_in = uv_uvfits
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    # change telescope name to avoid errors
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)
    uvh5_uv.read(testfile)

    # select on antennas
    ants_to_keep = np.array([0, 19, 11, 24, 3, 23, 1, 20, 21])
    uvh5_uv.read(testfile, antenna_nums=ants_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep)
    assert uvh5_uv == uvh5_uv2

    # clean up
    os.remove(testfile)

    return


def test_uvh5_partial_read_freqs(uv_uvfits):
    """
    Test reading in only certain frequencies from disk.
    """
    uv_in = uv_uvfits
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    # change telescope name to avoid errors
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)
    uvh5_uv.read(testfile)

    # select on frequency channels
    chans_to_keep = np.arange(12, 22)
    uvh5_uv.read(testfile, freq_chans=chans_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(freq_chans=chans_to_keep)
    assert uvh5_uv == uvh5_uv2

    # clean up
    os.remove(testfile)

    return


def test_uvh5_partial_read_pols(uv_uvfits):
    """
    Test reading in only certain polarizations from disk.
    """
    uv_in = uv_uvfits
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    # change telescope name to avoid errors
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)
    uvh5_uv.read(testfile)

    # select on pols
    pols_to_keep = [-1, -2]
    uvh5_uv.read(testfile, polarizations=pols_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(polarizations=pols_to_keep)
    assert uvh5_uv == uvh5_uv2

    return


def test_uvh5_partial_read_times(uv_uvfits):
    """
    Test reading in only certain times from disk.
    """
    uv_in = uv_uvfits
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    # change telescope name to avoid errors
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)
    uvh5_uv.read(testfile)

    # select on read using time_range
    unique_times = np.unique(uvh5_uv.time_array)
    uvtest.checkWarnings(
        uvh5_uv.read,
        [testfile],
        {'time_range': [unique_times[0], unique_times[1]]},
        message=['Warning: "time_range" keyword is set']
    )
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(times=unique_times[0:2])
    assert uvh5_uv == uvh5_uv2

    # clean up
    os.remove(testfile)

    return


def test_uvh5_partial_read_multi1(uv_uvfits):
    """
    Test select-on-read for multiple axes, frequencies being smallest fraction.
    """
    uv_in = uv_uvfits
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    # change telescope name to avoid errors
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)
    uvh5_uv.read(testfile)

    # now test selecting on multiple axes
    # read frequencies first
    ants_to_keep = np.array([0, 19, 11, 24, 3, 23, 1, 20, 21])
    chans_to_keep = np.arange(12, 22)
    pols_to_keep = [-1, -2]
    uvh5_uv.read(testfile, antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                 polarizations=pols_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    assert uvh5_uv == uvh5_uv2

    # clean up
    os.remove(testfile)

    return


def test_uvh5_partial_read_multi2(uv_uvfits):
    """
    Test select-on-read for multiple axes, baselines being smallest fraction.
    """
    uv_in = uv_uvfits
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    # change telescope name to avoid errors
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)
    uvh5_uv.read(testfile)

    # now test selecting on multiple axes
    # read baselines first
    ants_to_keep = np.array([0, 1])
    chans_to_keep = np.arange(12, 22)
    pols_to_keep = [-1, -2]
    uvh5_uv.read(testfile, antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                 polarizations=pols_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    assert uvh5_uv == uvh5_uv2

    # clean up
    os.remove(testfile)

    return


def test_uvh5_partial_read_multi3(uv_uvfits):
    """
    Test select-on-read for multiple axes, polarizations being smallest fraction.
    """
    uv_in = uv_uvfits
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    # change telescope name to avoid errors
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)
    uvh5_uv.read(testfile)

    # now test selecting on multiple axes
    # read polarizations first
    ants_to_keep = np.array([0, 1, 2, 3, 6, 7, 8, 11, 14, 18, 19, 20, 21, 22])
    chans_to_keep = np.arange(12, 64)
    pols_to_keep = [-1, -2]
    uvh5_uv.read(
        testfile,
        antenna_nums=ants_to_keep,
        freq_chans=chans_to_keep,
        polarizations=pols_to_keep,
    )
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(
        antenna_nums=ants_to_keep, freq_chans=chans_to_keep, polarizations=pols_to_keep
    )
    assert uvh5_uv == uvh5_uv2

    # clean up
    os.remove(testfile)

    return


def test_uvh5_partial_write_antpairs(uv_partial_write):
    """
    Test writing an entire UVH5 file in pieces by antpairs.
    """
    full_uvh5 = uv_partial_write
    partial_uvh5 = UVData()

    # delete data arrays in partial file
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)

    # write to file by iterating over antpairpol
    antpairpols = full_uvh5.get_antpairpols()
    for key in antpairpols:
        data = full_uvh5.get_data(key, squeeze='none')
        flags = full_uvh5.get_flags(key, squeeze='none')
        nsamples = full_uvh5.get_nsamples(key, squeeze='none')
        partial_uvh5.write_uvh5_part(
            partial_testfile, data, flags, nsamples, bls=key
        )

    # now read in the full file and make sure that it matches the original
    partial_uvh5.read(partial_testfile)
    assert full_uvh5 == partial_uvh5

    # test add_to_history
    key = antpairpols[0]
    data = full_uvh5.get_data(key, squeeze='none')
    flags = full_uvh5.get_flags(key, squeeze='none')
    nsamples = full_uvh5.get_nsamples(key, squeeze='none')
    partial_uvh5.write_uvh5_part(
        partial_testfile, data, flags, nsamples, bls=key, add_to_history="foo"
    )
    partial_uvh5.read(partial_testfile, read_data=False)
    assert 'foo' in partial_uvh5.history

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_frequencies(uv_partial_write):
    """
    Test writing an entire UVH5 file in pieces by frequencies.
    """
    full_uvh5 = uv_partial_write
    partial_uvh5 = UVData()

    # start over, and write frequencies
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)

    Nfreqs = full_uvh5.Nfreqs
    Hfreqs = Nfreqs // 2
    freqs1 = np.arange(Hfreqs)
    freqs2 = np.arange(Hfreqs, Nfreqs)
    data = full_uvh5.data_array[:, :, freqs1, :]
    flags = full_uvh5.flag_array[:, :, freqs1, :]
    nsamples = full_uvh5.nsample_array[:, :, freqs1, :]
    partial_uvh5.write_uvh5_part(
        partial_testfile, data, flags, nsamples, freq_chans=freqs1
    )
    data = full_uvh5.data_array[:, :, freqs2, :]
    flags = full_uvh5.flag_array[:, :, freqs2, :]
    nsamples = full_uvh5.nsample_array[:, :, freqs2, :]
    partial_uvh5.write_uvh5_part(
        partial_testfile, data, flags, nsamples, freq_chans=freqs2
    )

    # read in the full file and make sure it matches
    partial_uvh5.read(partial_testfile)
    assert full_uvh5 == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_blts(uv_partial_write):
    """
    Test writing an entire UVH5 file in pieces by blt.
    """
    full_uvh5 = uv_partial_write
    partial_uvh5 = UVData()

    # start over, write chunks of blts
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)

    Nblts = full_uvh5.Nblts
    Hblts = Nblts // 2
    blts1 = np.arange(Hblts)
    blts2 = np.arange(Hblts, Nblts)
    data = full_uvh5.data_array[blts1, :, :, :]
    flags = full_uvh5.flag_array[blts1, :, :, :]
    nsamples = full_uvh5.nsample_array[blts1, :, :, :]
    partial_uvh5.write_uvh5_part(
        partial_testfile, data, flags, nsamples, blt_inds=blts1
    )
    data = full_uvh5.data_array[blts2, :, :, :]
    flags = full_uvh5.flag_array[blts2, :, :, :]
    nsamples = full_uvh5.nsample_array[blts2, :, :, :]
    partial_uvh5.write_uvh5_part(
        partial_testfile, data, flags, nsamples, blt_inds=blts2
    )

    # read in the full file and make sure it matches
    partial_uvh5.read(partial_testfile)
    assert full_uvh5 == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_pols(uv_partial_write):
    """
    Test writing an entire UVH5 file in pieces by pol.
    """
    full_uvh5 = uv_partial_write
    partial_uvh5 = UVData()

    # start over, write groups of pols
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)

    Npols = full_uvh5.Npols
    Hpols = Npols // 2
    pols1 = np.arange(Hpols)
    pols2 = np.arange(Hpols, Npols)
    data = full_uvh5.data_array[:, :, :, pols1]
    flags = full_uvh5.flag_array[:, :, :, pols1]
    nsamples = full_uvh5.nsample_array[:, :, :, pols1]
    partial_uvh5.write_uvh5_part(
        partial_testfile,
        data,
        flags,
        nsamples,
        polarizations=full_uvh5.polarization_array[:Hpols]
    )
    data = full_uvh5.data_array[:, :, :, pols2]
    flags = full_uvh5.flag_array[:, :, :, pols2]
    nsamples = full_uvh5.nsample_array[:, :, :, pols2]
    partial_uvh5.write_uvh5_part(
        partial_testfile,
        data,
        flags,
        nsamples,
        polarizations=full_uvh5.polarization_array[Hpols:]
    )

    # read in the full file and make sure it matches
    partial_uvh5.read(partial_testfile)
    assert full_uvh5 == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_irregular_blt(uv_partial_write):
    """
    Test writing a uvh5 file using irregular intervals for single blt.
    """
    full_uvh5 = uv_partial_write
    partial_uvh5 = UVData()

    # delete data arrays in partial file
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # write a single blt to file
    blt_inds = np.arange(1)
    data = full_uvh5.data_array[blt_inds, :, :, :]
    flags = full_uvh5.flag_array[blt_inds, :, :, :]
    nsamples = full_uvh5.nsample_array[blt_inds, :, :, :]
    partial_uvh5.write_uvh5_part(partial_testfile, data, flags, nsamples, blt_inds=blt_inds)

    # also write the arrays to the partial object
    partial_uvh5.data_array[blt_inds, :, :, :] = data
    partial_uvh5.flag_array[blt_inds, :, :, :] = flags
    partial_uvh5.nsample_array[blt_inds, :, :, :] = nsamples

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_irregular_freq(uv_partial_write):
    """
    Test writing a uvh5 file using irregular intervals for single frequency.
    """
    full_uvh5 = uv_partial_write
    partial_uvh5 = UVData()

    # delete data arrays in partial file
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # write a single freq to file
    freq_inds = np.arange(1)
    data = full_uvh5.data_array[:, :, freq_inds, :]
    flags = full_uvh5.flag_array[:, :, freq_inds, :]
    nsamples = full_uvh5.nsample_array[:, :, freq_inds, :]
    partial_uvh5.write_uvh5_part(
        partial_testfile, data, flags, nsamples, freq_chans=freq_inds
    )

    # also write the arrays to the partial object
    partial_uvh5.data_array[:, :, freq_inds, :] = data
    partial_uvh5.flag_array[:, :, freq_inds, :] = flags
    partial_uvh5.nsample_array[:, :, freq_inds, :] = nsamples

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_irregular_pol(uv_partial_write):
    """
    Test writing a uvh5 file using irregular intervals for single polarization.
    """
    full_uvh5 = uv_partial_write
    partial_uvh5 = UVData()

    # delete data arrays in partial file
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # write a single pol to file
    pol_inds = np.arange(1)
    data = full_uvh5.data_array[:, :, :, pol_inds]
    flags = full_uvh5.flag_array[:, :, :, pol_inds]
    nsamples = full_uvh5.nsample_array[:, :, :, pol_inds]
    partial_uvh5.write_uvh5_part(
        partial_testfile,
        data,
        flags,
        nsamples,
        polarizations=partial_uvh5.polarization_array[pol_inds],
    )

    # also write the arrays to the partial object
    partial_uvh5.data_array[:, :, :, pol_inds] = data
    partial_uvh5.flag_array[:, :, :, pol_inds] = flags
    partial_uvh5.nsample_array[:, :, :, pol_inds] = nsamples

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_irregular_multi1(uv_partial_write):
    """
    Test writing a uvh5 file using irregular intervals for blts and freqs.
    """
    full_uvh5 = uv_partial_write
    full_uvh5.telescope_name = "PAPER"
    partial_uvh5 = UVData()

    # delete data arrays in partial file
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # define blts and freqs
    blt_inds = [0, 1, 2, 7]
    freq_inds = [0, 2, 3, 4]
    data_shape = (len(blt_inds), 1, len(freq_inds), full_uvh5.Npols)
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    for iblt, blt_idx in enumerate(blt_inds):
        for ifreq, freq_idx in enumerate(freq_inds):
            data[iblt, :, ifreq, :] = full_uvh5.data_array[blt_idx, :, freq_idx, :]
            flags[iblt, :, ifreq, :] = full_uvh5.flag_array[blt_idx, :, freq_idx, :]
            nsamples[iblt, :, ifreq, :] = full_uvh5.nsample_array[blt_idx, :, freq_idx, :]
    uvtest.checkWarnings(
        partial_uvh5.write_uvh5_part,
        [partial_testfile, data, flags, nsamples],
        {'blt_inds': blt_inds, 'freq_chans': freq_inds},
        message='Selected frequencies are not evenly spaced',
    )

    # also write the arrays to the partial object
    for iblt, blt_idx in enumerate(blt_inds):
        for ifreq, freq_idx in enumerate(freq_inds):
            partial_uvh5.data_array[blt_idx, :, freq_idx, :] = data[iblt, :, ifreq, :]
            partial_uvh5.flag_array[blt_idx, :, freq_idx, :] = flags[iblt, :, ifreq, :]
            partial_uvh5.nsample_array[blt_idx, :, freq_idx, :] = nsamples[iblt, :, ifreq, :]

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_irregular_multi2(uv_partial_write):
    """
    Test writing a uvh5 file using irregular intervals for freqs and pols.
    """
    full_uvh5 = uv_partial_write
    full_uvh5.telescope_name = "PAPER"
    partial_uvh5 = UVData()

    # delete data arrays in partial file
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # define freqs and pols
    freq_inds = [0, 1, 2, 7]
    pol_inds = [0, 1, 3]
    data_shape = (full_uvh5.Nblts, 1, len(freq_inds), len(pol_inds))
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    for ifreq, freq_idx in enumerate(freq_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            data[:, :, ifreq, ipol] = full_uvh5.data_array[:, :, freq_idx, pol_idx]
            flags[:, :, ifreq, ipol] = full_uvh5.flag_array[:, :, freq_idx, pol_idx]
            nsamples[:, :, ifreq, ipol] = full_uvh5.nsample_array[:, :, freq_idx, pol_idx]
    uvtest.checkWarnings(
        partial_uvh5.write_uvh5_part,
        [partial_testfile, data, flags, nsamples],
        {'freq_chans': freq_inds, 'polarizations': full_uvh5.polarization_array[pol_inds]},
        nwarnings=2,
        message=[
            'Selected frequencies are not evenly spaced',
            'Selected polarization values are not evenly spaced',
        ]
    )

    # also write the arrays to the partial object
    for ifreq, freq_idx in enumerate(freq_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            partial_uvh5.data_array[:, :, freq_idx, pol_idx] = data[:, :, ifreq, ipol]
            partial_uvh5.flag_array[:, :, freq_idx, pol_idx] = flags[:, :, ifreq, ipol]
            partial_uvh5.nsample_array[:, :, freq_idx, pol_idx] = nsamples[:, :, ifreq, ipol]

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_irregular_multi3(uv_partial_write):
    """
    Test writing a uvh5 file using irregular intervals for blts and pols.
    """
    full_uvh5 = uv_partial_write
    full_uvh5.telescope_name = "PAPER"
    partial_uvh5 = UVData()

    # delete data arrays in partial file
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # define blts and freqs
    blt_inds = [0, 1, 2, 7]
    pol_inds = [0, 1, 3]
    data_shape = (len(blt_inds), 1, full_uvh5.Nfreqs, len(pol_inds))
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    for iblt, blt_idx in enumerate(blt_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            data[iblt, :, :, ipol] = full_uvh5.data_array[blt_idx, :, :, pol_idx]
            flags[iblt, :, :, ipol] = full_uvh5.flag_array[blt_idx, :, :, pol_idx]
            nsamples[iblt, :, :, ipol] = full_uvh5.nsample_array[blt_idx, :, :, pol_idx]
    uvtest.checkWarnings(
        partial_uvh5.write_uvh5_part,
        [partial_testfile, data, flags, nsamples],
        {'blt_inds': blt_inds, 'polarizations': full_uvh5.polarization_array[pol_inds]},
        message='Selected polarization values are not evenly spaced',
    )

    # also write the arrays to the partial object
    for iblt, blt_idx in enumerate(blt_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            partial_uvh5.data_array[blt_idx, :, :, pol_idx] = data[iblt, :, :, ipol]
            partial_uvh5.flag_array[blt_idx, :, :, pol_idx] = flags[iblt, :, :, ipol]
            partial_uvh5.nsample_array[blt_idx, :, :, pol_idx] = nsamples[iblt, :, :, ipol]

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    return


def test_uvh5_partial_write_irregular_multi4(uv_partial_write):
    """
    Test writing a uvh5 file using irregular intervals for all axes.
    """
    full_uvh5 = uv_partial_write
    full_uvh5.telescope_name = "PAPER"
    partial_uvh5 = UVData()

    # delete data arrays in partial file
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # define blts and freqs
    blt_inds = [0, 1, 2, 7]
    freq_inds = [0, 2, 3, 4]
    pol_inds = [0, 1, 3]
    data_shape = (len(blt_inds), 1, len(freq_inds), len(pol_inds))
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    for iblt, blt_idx in enumerate(blt_inds):
        for ifreq, freq_idx in enumerate(freq_inds):
            for ipol, pol_idx in enumerate(pol_inds):
                data[iblt, :, ifreq, ipol] = full_uvh5.data_array[blt_idx, :, freq_idx, pol_idx]
                flags[iblt, :, ifreq, ipol] = full_uvh5.flag_array[blt_idx, :, freq_idx, pol_idx]
                nsamples[iblt, :, ifreq, ipol] = full_uvh5.nsample_array[blt_idx, :, freq_idx, pol_idx]
    uvtest.checkWarnings(
        partial_uvh5.write_uvh5_part,
        [partial_testfile, data, flags, nsamples],
        {
            'blt_inds': blt_inds,
            'freq_chans': freq_inds,
            'polarizations': full_uvh5.polarization_array[pol_inds],
        },
        nwarnings=2,
        message=[
            'Selected frequencies are not evenly spaced',
            'Selected polarization values are not evenly spaced',
        ]
    )

    # also write the arrays to the partial object
    for iblt, blt_idx in enumerate(blt_inds):
        for ifreq, freq_idx in enumerate(freq_inds):
            for ipol, pol_idx in enumerate(pol_inds):
                partial_uvh5.data_array[blt_idx, :, freq_idx, pol_idx] = data[iblt, :, ifreq, ipol]
                partial_uvh5.flag_array[blt_idx, :, freq_idx, pol_idx] = flags[iblt, :, ifreq, ipol]
                partial_uvh5.nsample_array[blt_idx, :, freq_idx, pol_idx] = nsamples[iblt, :, ifreq, ipol]

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_errors(uv_partial_write):
    """
    Test errors in uvh5_write_part method.
    """
    full_uvh5 = uv_partial_write
    partial_uvh5 = UVData()

    # get a waterfall
    antpairpols = full_uvh5.get_antpairpols()
    key = antpairpols[0]
    data = full_uvh5.get_data(key, squeeze='none')
    flags = full_uvh5.get_data(key, squeeze='none')
    nsamples = full_uvh5.get_data(key, squeeze='none')

    # delete data arrays in partial file
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # try to write to a file that doesn't exists
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    if os.path.exists(partial_testfile):
        os.remove(partial_testfile)
    with pytest.raises(AssertionError) as cm:
        partial_uvh5.write_uvh5_part(
            partial_testfile, data, flags, nsamples, bls=key
        )
    assert str(cm.value).startswith("{} does not exist".format(partial_testfile))

    # initialize file on disk
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)

    # pass in arrays that are different sizes
    with pytest.raises(AssertionError) as cm:
        partial_uvh5.write_uvh5_part(
            partial_testfile, data, flags[:, :, :, 0], nsamples, bls=key
        )
    assert str(cm.value).startswith("data_array and flag_array must have the same shape")
    with pytest.raises(AssertionError) as cm:
        partial_uvh5.write_uvh5_part(
            partial_testfile, data, flags, nsamples[:, :, :, 0], bls=key
        )
    assert str(cm.value).startswith("data_array and nsample_array must have the same shape")

    # pass in arrays that are the same size, but don't match expected shape
    with pytest.raises(AssertionError) as cm:
        partial_uvh5.write_uvh5_part(
            partial_testfile, data[:, :, :, 0], flags[:, :, :, 0], nsamples[:, :, :, 0]
        )
    assert str(cm.value).startswith("data_array has shape")

    # initialize a file on disk, and pass in a different object so check_header fails
    empty_uvd = UVData()
    with pytest.raises(AssertionError) as cm:
        empty_uvd.write_uvh5_part(
            partial_testfile, data, flags, nsamples, bls=key
        )
    assert str(cm.value).startswith(
        "The object metadata in memory and metadata on disk are different"
    )

    # clean up
    os.remove(partial_testfile)

    return


def test_initialize_uvh5_file(uv_partial_write):
    """
    Test initializing a UVH5 file on disk.
    """
    full_uvh5 = uv_partial_write
    full_uvh5.data_array = None
    full_uvh5.flag_array = None
    full_uvh5.nsample_array = None

    # initialize file
    partial_uvh5 = full_uvh5.copy()
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)

    # read it in and make sure that the metadata matches the original
    partial_uvh5.read(partial_testfile, read_data=False)
    assert partial_uvh5 == full_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_initialize_uvh5_file_errors(uv_partial_write):
    """
    Test errors in initializing a UVH5 file on disk.
    """
    full_uvh5 = uv_partial_write
    full_uvh5.data_array = None
    full_uvh5.flag_array = None
    full_uvh5.nsample_array = None

    # initialize file
    partial_uvh5 = full_uvh5.copy()
    partial_testfile = os.path.join(DATA_PATH, "test", "outtest_partial.uvh5")
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)

    # check that IOError is raised then when clobber == False
    with pytest.raises(IOError) as cm:
        partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=False)
    assert str(cm.value).startswith("File exists; skipping")

    # clean up
    os.remove(partial_testfile)

    return


def test_initialize_uvh5_file_compression_opts(uv_partial_write):
    """
    Test initializing a uvh5 file with compression options.
    """
    full_uvh5 = uv_partial_write
    full_uvh5.data_array = None
    full_uvh5.flag_array = None
    full_uvh5.nsample_array = None

    # add options for compression
    partial_uvh5 = full_uvh5.copy()
    partial_testfile = os.path.join(DATA_PATH, "test", "outtest_partial.uvh5")
    partial_uvh5.initialize_uvh5_file(
        partial_testfile,
        clobber=True,
        data_compression="lzf",
        flags_compression=None,
        nsample_compression=None,
    )
    partial_uvh5.read(partial_testfile, read_data=False)
    assert partial_uvh5 == full_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_single_integration_time(uv_uvfits):
    """
    Check backwards compatibility warning for files with a single integration time.
    """
    uv_in = uv_uvfits
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)

    # change integration_time in file to be a single number
    with h5py.File(testfile, 'r+') as h5f:
        int_time = h5f['/Header/integration_time'][0]
        del(h5f['/Header/integration_time'])
        h5f['/Header/integration_time'] = int_time
    uvtest.checkWarnings(
        uv_out.read_uvh5,
        [testfile],
        message='outtest_uvfits.uvh5 appears to be an old uvh5 format',
        category=DeprecationWarning,
    )
    assert uv_in == uv_out

    # clean up
    os.remove(testfile)

    return


def test_uvh5_lst_array(uv_uvfits):
    """
    Test different cases of the lst_array.
    """
    uv_in = uv_uvfits
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)

    # remove lst_array from file; check that it's correctly computed on read
    with h5py.File(testfile, 'r+') as h5f:
        del(h5f['/Header/lst_array'])
    uv_out.read_uvh5(testfile)
    assert uv_in == uv_out

    # now change what's in the file and make sure a warning is raised
    uv_in.write_uvh5(testfile, clobber=True)
    with h5py.File(testfile, 'r+') as h5f:
        lst_array = h5f['/Header/lst_array'][:]
        del(h5f['/Header/lst_array'])
        h5f['/Header/lst_array'] = 2 * lst_array
    uvtest.checkWarnings(
        uv_out.read_uvh5,
        [testfile],
        message='LST values stored in outtest_uvfits.uvh5 are not self-consistent',
    )
    uv_out.lst_array = lst_array
    assert uv_in == uv_out

    # clean up
    os.remove(testfile)

    return


def test_uvh5_string_back_compat(uv_uvfits):
    """
    Test backwards compatibility handling of strings.
    """
    uv_in = uv_uvfits
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)

    # write a string-type data as-is, without casting to np.string_
    with h5py.File(testfile, 'r+') as h5f:
        del(h5f['Header/instrument'])
        h5f['Header/instrument'] = uv_in.instrument
    uvtest.checkWarnings(
        uv_out.read_uvh5,
        [testfile],
        message='Strings in metadata of outtest_uvfits.uvh5 are not the correct type',
        category=DeprecationWarning,
    )
    assert uv_in == uv_out

    # clean up
    os.remove(testfile)

    return


def test_uvh5_read_header_special_cases(uv_uvfits):
    """
    Test special cases values when reading files.
    """
    uv_in = uv_uvfits
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)

    # change some of the metadata to trip certain if/else clauses
    with h5py.File(testfile, 'r+') as h5f:
        del(h5f['Header/history'])
        del(h5f['Header/vis_units'])
        del(h5f['Header/phase_type'])
        del(h5f['Header/latitude'])
        del(h5f['Header/longitude'])
        h5f['Header/history'] = np.string_('blank history')
        h5f['Header/phase_type'] = np.string_('blah')
        h5f['Header/latitude'] = uv_in.telescope_location_lat_lon_alt[0]
        h5f['Header/longitude'] = uv_in.telescope_location_lat_lon_alt[1]
    uvtest.checkWarnings(
        uv_out.read_uvh5,
        [testfile],
        category=DeprecationWarning,
        message='It seems that the latitude and longitude are in radians',
    )

    # make input and output values match now
    uv_in.history = uv_out.history
    uv_in.set_unknown_phase_type()
    uv_in.phase_center_ra = None
    uv_in.phase_center_dec = None
    uv_in.phase_center_epoch = None
    uv_in.vis_units = 'UNCALIB'
    assert uv_in == uv_out

    # clean up
    os.remove(testfile)

    return


def test_uvh5_read_ints(uv_uvh5):
    """
    Test reading visibility data saved as integers.
    """
    uv_in = uv_uvh5
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    uv_in.write_uvh5(testfile, clobber=True)

    # read it back in to make sure data is the same
    uv_out.read_uvh5(testfile)
    assert uv_in == uv_out

    # now read in as np.complex128
    uvh5_filename = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')
    uv_in.read_uvh5(uvh5_filename, data_array_dtype=np.complex128)
    assert uv_in == uv_out
    assert uv_in.data_array.dtype == np.dtype(np.complex128)

    # clean up
    os.remove(testfile)

    return


def test_uvh5_read_ints_error():
    """
    Test raising an error for passing in an unsupported data_array dtype.
    """
    uv_in = UVData()
    uvh5_filename = os.path.join(DATA_PATH, "zen.2458432.34569.uvh5")

    # raise error for bogus data_array_dtype
    with pytest.raises(ValueError) as cm:
        uv_in.read_uvh5(uvh5_filename, data_array_dtype=np.int32)
    assert str(cm.value).startswith("data_array_dtype must be np.complex64 or np.complex128")

    return


def test_uvh5_write_ints(uv_uvh5):
    """
    Test writing visibility data as integers.
    """
    uv_in = uv_uvh5
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    uv_in.write_uvh5(testfile, clobber=True, data_write_dtype=_hera_corr_dtype)

    # read it back in to make sure data is the same
    uv_out.read_uvh5(testfile)
    assert uv_in == uv_out

    # also check that the datatype on disk is the right type
    with h5py.File(testfile, 'r') as h5f:
        visdata_dtype = h5f['Data/visdata'].dtype
        assert 'r' in visdata_dtype.names
        assert 'i' in visdata_dtype.names
        assert visdata_dtype['r'].kind == 'i'
        assert visdata_dtype['i'].kind == 'i'

    # clean up
    os.remove(testfile)

    return


def test_uvh5_partial_read_ints_antennas():
    """
    Test reading in only some antennas from disk with integer data type.
    """
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    uvh5_file = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')

    # select on antennas
    ants_to_keep = np.array([0, 1])
    uvh5_uv.read(uvh5_file, antenna_nums=ants_to_keep)
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(antenna_nums=ants_to_keep)
    assert uvh5_uv == uvh5_uv2

    return


def test_uvh5_partial_read_ints_freqs():
    """
    Test reading in only some frequencies from disk with integer data type.
    """
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    uvh5_file = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')

    # select on frequency channels
    chans_to_keep = np.arange(12, 22)
    uvh5_uv.read(uvh5_file, freq_chans=chans_to_keep)
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(freq_chans=chans_to_keep)
    assert uvh5_uv == uvh5_uv2

    return


def test_uvh5_partial_read_ints_pols():
    """
    Test reading in only some polarizations from disk with integer data type.
    """
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    uvh5_file = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')

    # select on pols
    pols_to_keep = [-5, -6]
    uvh5_uv.read(uvh5_file, polarizations=pols_to_keep)
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(polarizations=pols_to_keep)
    assert uvh5_uv == uvh5_uv2

    return


def test_uvh5_partial_read_ints_times():
    """
    Test reading in only some times from disk with integer data type.
    """
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    uvh5_file = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')

    # select on read using time_range
    uvh5_uv.read_uvh5(uvh5_file, read_data=False)
    unique_times = np.unique(uvh5_uv.time_array)
    uvtest.checkWarnings(
        uvh5_uv.read,
        [uvh5_file],
        {'time_range': [unique_times[0], unique_times[1]]},
        message=['Warning: "time_range" keyword is set'],
    )
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(times=unique_times[0:2])
    assert uvh5_uv == uvh5_uv2

    return


def test_uvh5_partial_read_ints_multi1():
    """
    Test select-on-read for multiple axes, frequencies being smallest fraction.
    """
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    uvh5_file = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')

    # read frequencies first
    ants_to_keep = np.array([0, 1])
    chans_to_keep = np.arange(12, 22)
    pols_to_keep = [-5, -6]
    uvh5_uv.read(
        uvh5_file,
        antenna_nums=ants_to_keep,
        freq_chans=chans_to_keep,
        polarizations=pols_to_keep,
    )
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(
        antenna_nums=ants_to_keep, freq_chans=chans_to_keep, polarizations=pols_to_keep
    )
    assert uvh5_uv == uvh5_uv2

    return


def test_uvh5_partial_read_ints_multi2():
    """
    Test select-on-read for multiple axes, baselines being smallest fraction.
    """
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    uvh5_file = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')

    # read baselines first
    ants_to_keep = np.array([0, 1])
    chans_to_keep = np.arange(12, 22)
    pols_to_keep = [-5, -6, -7]
    uvh5_uv.read(
        uvh5_file,
        antenna_nums=ants_to_keep,
        freq_chans=chans_to_keep,
        polarizations=pols_to_keep,
    )
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(
        antenna_nums=ants_to_keep,
        freq_chans=chans_to_keep,
        polarizations=pols_to_keep,
    )
    assert uvh5_uv == uvh5_uv2

    return


def test_uvh5_partial_read_ints_multi3():
    """
    Test select-on-read for multiple axes, polarizations being smallest fraction.
    """
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    uvh5_file = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')

    # read polarizations first
    ants_to_keep = np.array([0, 1, 12])
    chans_to_keep = np.arange(12, 64)
    pols_to_keep = [-5, -6]
    uvh5_uv.read(
        uvh5_file,
        antenna_nums=ants_to_keep,
        freq_chans=chans_to_keep,
        polarizations=pols_to_keep,
    )
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(
        antenna_nums=ants_to_keep, freq_chans=chans_to_keep, polarizations=pols_to_keep
    )
    assert uvh5_uv == uvh5_uv2

    return


def test_uvh5_partial_write_ints_antapirs(uv_uvh5):
    """
    Test writing an entire UVH5 file in pieces by antpairs using ints.
    """
    full_uvh5 = uv_uvh5

    # delete data arrays in partial file
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    partial_uvh5.initialize_uvh5_file(
        partial_testfile,
        clobber=True,
        data_write_dtype=_hera_corr_dtype,
    )

    # write to file by iterating over antpairpol
    antpairpols = full_uvh5.get_antpairpols()
    for key in antpairpols:
        data = full_uvh5.get_data(key, squeeze='none')
        flags = full_uvh5.get_flags(key, squeeze='none')
        nsamples = full_uvh5.get_nsamples(key, squeeze='none')
        partial_uvh5.write_uvh5_part(
            partial_testfile, data, flags, nsamples, bls=key
        )

    # now read in the full file and make sure that it matches the original
    partial_uvh5.read(partial_testfile)
    assert full_uvh5 == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_ints_frequencies(uv_uvh5):
    """
    Test writing an entire UVH5 file in pieces by frequency using ints.
    """
    full_uvh5 = uv_uvh5

    # delete data arrays in partial file
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    partial_uvh5.initialize_uvh5_file(
        partial_testfile, clobber=True, data_write_dtype=_hera_corr_dtype
    )

    # only write certain frequencies
    Nfreqs = full_uvh5.Nfreqs
    Hfreqs = Nfreqs // 2
    freqs1 = np.arange(Hfreqs)
    freqs2 = np.arange(Hfreqs, Nfreqs)
    data = full_uvh5.data_array[:, :, freqs1, :]
    flags = full_uvh5.flag_array[:, :, freqs1, :]
    nsamples = full_uvh5.nsample_array[:, :, freqs1, :]
    partial_uvh5.write_uvh5_part(
        partial_testfile, data, flags, nsamples, freq_chans=freqs1
    )
    data = full_uvh5.data_array[:, :, freqs2, :]
    flags = full_uvh5.flag_array[:, :, freqs2, :]
    nsamples = full_uvh5.nsample_array[:, :, freqs2, :]
    partial_uvh5.write_uvh5_part(
        partial_testfile, data, flags, nsamples, freq_chans=freqs2
    )

    # read in the full file and make sure it matches
    partial_uvh5.read(partial_testfile)
    assert full_uvh5 == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_ints_blts(uv_uvh5):
    """
    Test writing an entire UVH5 file in pieces by blt using ints.
    """
    full_uvh5 = uv_uvh5

    # delete data arrays in partial file
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    partial_uvh5.initialize_uvh5_file(
        partial_testfile, clobber=True, data_write_dtype=_hera_corr_dtype
    )

    # only write certain blts
    Nblts = full_uvh5.Nblts
    Hblts = Nblts // 2
    blts1 = np.arange(Hblts)
    blts2 = np.arange(Hblts, Nblts)
    data = full_uvh5.data_array[blts1, :, :, :]
    flags = full_uvh5.flag_array[blts1, :, :, :]
    nsamples = full_uvh5.nsample_array[blts1, :, :, :]
    partial_uvh5.write_uvh5_part(
        partial_testfile, data, flags, nsamples, blt_inds=blts1
    )
    data = full_uvh5.data_array[blts2, :, :, :]
    flags = full_uvh5.flag_array[blts2, :, :, :]
    nsamples = full_uvh5.nsample_array[blts2, :, :, :]
    partial_uvh5.write_uvh5_part(
        partial_testfile, data, flags, nsamples, blt_inds=blts2
    )

    # read in the full file and make sure it matches
    partial_uvh5.read(partial_testfile)
    assert full_uvh5 == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_ints_pols(uv_uvh5):
    """
    Test writing an entire UVH5 file in pieces by polarization using ints.
    """
    full_uvh5 = uv_uvh5

    # delete data arrays in partial file
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    partial_uvh5.initialize_uvh5_file(
        partial_testfile, clobber=True, data_write_dtype=_hera_corr_dtype
    )

    # only write certain polarizations
    Npols = full_uvh5.Npols
    Hpols = Npols // 2
    pols1 = np.arange(Hpols)
    pols2 = np.arange(Hpols, Npols)
    data = full_uvh5.data_array[:, :, :, pols1]
    flags = full_uvh5.flag_array[:, :, :, pols1]
    nsamples = full_uvh5.nsample_array[:, :, :, pols1]
    partial_uvh5.write_uvh5_part(
        partial_testfile,
        data,
        flags,
        nsamples,
        polarizations=full_uvh5.polarization_array[:Hpols],
    )
    data = full_uvh5.data_array[:, :, :, pols2]
    flags = full_uvh5.flag_array[:, :, :, pols2]
    nsamples = full_uvh5.nsample_array[:, :, :, pols2]
    partial_uvh5.write_uvh5_part(
        partial_testfile,
        data,
        flags,
        nsamples,
        polarizations=full_uvh5.polarization_array[Hpols:],
    )

    # read in the full file and make sure it matches
    partial_uvh5.read(partial_testfile)
    assert full_uvh5 == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_read_complex_astype():
    # make a testfile with a test dataset
    test_file = os.path.join(DATA_PATH, 'test', 'test_file.h5')
    test_data_shape = (2, 3, 4, 5)
    test_data = np.zeros(test_data_shape, dtype=np.complex64)
    test_data.real = 1.
    test_data.imag = 2.
    with h5py.File(test_file, 'w') as h5f:
        dgrp = h5f.create_group('Data')
        dset = dgrp.create_dataset('testdata', test_data_shape,
                                   dtype=_hera_corr_dtype)
        with dset.astype(_hera_corr_dtype):
            dset[:, :, :, :, 'r'] = test_data.real
            dset[:, :, :, :, 'i'] = test_data.imag

    # test that reading the data back in works as expected
    indices = (np.s_[:], np.s_[:], np.s_[:], np.s_[:])
    with h5py.File(test_file, 'r') as h5f:
        dset = h5f['Data/testdata']
        file_data = uvh5._read_complex_astype(dset, indices, np.complex64)

    assert np.allclose(file_data, test_data)

    # clean up
    os.remove(test_file)

    return


def test_read_complex_astype_errors():
    # make a testfile with a test dataset
    test_file = os.path.join(DATA_PATH, 'test', 'test_file.h5')
    test_data_shape = (2, 3, 4, 5)
    test_data = np.zeros(test_data_shape, dtype=np.complex64)
    test_data.real = 1.
    test_data.imag = 2.
    with h5py.File(test_file, 'w') as h5f:
        dgrp = h5f.create_group('Data')
        dset = dgrp.create_dataset('testdata', test_data_shape,
                                   dtype=_hera_corr_dtype)
        with dset.astype(_hera_corr_dtype):
            dset[:, :, :, :, 'r'] = test_data.real
            dset[:, :, :, :, 'i'] = test_data.imag

    # test passing in a forbidden output datatype
    indices = (np.s_[:], np.s_[:], np.s_[:], np.s_[:])
    with h5py.File(test_file, 'r') as h5f:
        dset = h5f['Data/testdata']
        with pytest.raises(ValueError) as cm:
            uvh5._read_complex_astype(dset, indices, np.int32)
        assert str(cm.value).startswith("output datatype must be one of (complex")

    # clean up
    os.remove(test_file)

    return


def test_write_complex_astype():
    # make sure we can write data out
    test_file = os.path.join(DATA_PATH, 'test', 'test_file.h5')
    test_data_shape = (2, 3, 4, 5)
    test_data = np.zeros(test_data_shape, dtype=np.complex64)
    test_data.real = 1.
    test_data.imag = 2.
    with h5py.File(test_file, 'w') as h5f:
        dgrp = h5f.create_group('Data')
        dset = dgrp.create_dataset('testdata', test_data_shape,
                                   dtype=_hera_corr_dtype)
        inds = (np.s_[:], np.s_[:], np.s_[:], np.s_[:])
        uvh5._write_complex_astype(test_data, dset, inds)

    # read the data back in to confirm it's right
    with h5py.File(test_file, 'r') as h5f:
        dset = h5f['Data/testdata']
        file_data = np.zeros(test_data_shape, dtype=np.complex64)
        with dset.astype(_hera_corr_dtype):
            file_data.real = dset['r'][:, :, :, :]
            file_data.imag = dset['i'][:, :, :, :]

    assert np.allclose(file_data, test_data)

    return


def test_check_uvh5_dtype_errors():
    # test passing in something that's not a dtype
    with pytest.raises(ValueError) as cm:
        uvh5._check_uvh5_dtype('hi')
    assert str(cm.value).startswith("dtype in a uvh5 file must be a numpy dtype")

    # test using a dtype with bad field names
    dtype = np.dtype([('a', '<i4'), ('b', '<i4')])
    with pytest.raises(ValueError) as cm:
        uvh5._check_uvh5_dtype(dtype)
    assert str(cm.value).startswith("dtype must be a compound datatype")

    # test having different types for 'r' and 'i' fields
    dtype = np.dtype([('r', '<i4'), ('i', '<f4')])
    with pytest.raises(ValueError) as cm:
        uvh5._check_uvh5_dtype(dtype)
    assert str(cm.value).startswith("dtype must have the same kind")

    return


def test_uvh5_partial_write_ints_irregular_blt(uv_uvh5):
    """
    Test writing a uvh5 file using irregular interval for blt and integer dtype.
    """
    full_uvh5 = uv_uvh5
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # write a single blt to file
    blt_inds = np.arange(1)
    data = full_uvh5.data_array[blt_inds, :, :, :]
    flags = full_uvh5.flag_array[blt_inds, :, :, :]
    nsamples = full_uvh5.nsample_array[blt_inds, :, :, :]
    partial_uvh5.write_uvh5_part(partial_testfile, data, flags, nsamples, blt_inds=blt_inds)

    # also write the arrays to the partial object
    partial_uvh5.data_array[blt_inds, :, :, :] = data
    partial_uvh5.flag_array[blt_inds, :, :, :] = flags
    partial_uvh5.nsample_array[blt_inds, :, :, :] = nsamples

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_ints_irregular_freq(uv_uvh5):
    """
    Test writing a uvh5 file using irregular interval for freq and integer dtype.
    """
    full_uvh5 = uv_uvh5
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # write a single freq to file
    freq_inds = np.arange(1)
    data = full_uvh5.data_array[:, :, freq_inds, :]
    flags = full_uvh5.flag_array[:, :, freq_inds, :]
    nsamples = full_uvh5.nsample_array[:, :, freq_inds, :]
    partial_uvh5.write_uvh5_part(
        partial_testfile, data, flags, nsamples, freq_chans=freq_inds
    )

    # also write the arrays to the partial object
    partial_uvh5.data_array[:, :, freq_inds, :] = data
    partial_uvh5.flag_array[:, :, freq_inds, :] = flags
    partial_uvh5.nsample_array[:, :, freq_inds, :] = nsamples

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_ints_irregular_pol(uv_uvh5):
    """
    Test writing a uvh5 file using irregular interval for pol and integer dtype.
    """
    full_uvh5 = uv_uvh5
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # write a single pol to file
    pol_inds = np.arange(1)
    data = full_uvh5.data_array[:, :, :, pol_inds]
    flags = full_uvh5.flag_array[:, :, :, pol_inds]
    nsamples = full_uvh5.nsample_array[:, :, :, pol_inds]
    partial_uvh5.write_uvh5_part(
        partial_testfile,
        data,
        flags,
        nsamples,
        polarizations=partial_uvh5.polarization_array[pol_inds],
    )

    # also write the arrays to the partial object
    partial_uvh5.data_array[:, :, :, pol_inds] = data
    partial_uvh5.flag_array[:, :, :, pol_inds] = flags
    partial_uvh5.nsample_array[:, :, :, pol_inds] = nsamples

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_ints_irregular_multi1(uv_uvh5):
    """
    Test writing a uvh5 file using irregular interval for blt and freq and integer dtype.
    """
    full_uvh5 = uv_uvh5
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # define blts and freqs
    blt_inds = [0, 1, 2, 7]
    freq_inds = [0, 2, 3, 4]
    data_shape = (len(blt_inds), 1, len(freq_inds), full_uvh5.Npols)
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    for iblt, blt_idx in enumerate(blt_inds):
        for ifreq, freq_idx in enumerate(freq_inds):
            data[iblt, :, ifreq, :] = full_uvh5.data_array[blt_idx, :, freq_idx, :]
            flags[iblt, :, ifreq, :] = full_uvh5.flag_array[blt_idx, :, freq_idx, :]
            nsamples[iblt, :, ifreq, :] = full_uvh5.nsample_array[blt_idx, :, freq_idx, :]
    uvtest.checkWarnings(
        partial_uvh5.write_uvh5_part,
        [partial_testfile, data, flags, nsamples],
        {'blt_inds': blt_inds, 'freq_chans': freq_inds},
        message='Selected frequencies are not evenly spaced',
    )

    # also write the arrays to the partial object
    for iblt, blt_idx in enumerate(blt_inds):
        for ifreq, freq_idx in enumerate(freq_inds):
            partial_uvh5.data_array[blt_idx, :, freq_idx, :] = data[iblt, :, ifreq, :]
            partial_uvh5.flag_array[blt_idx, :, freq_idx, :] = flags[iblt, :, ifreq, :]
            partial_uvh5.nsample_array[blt_idx, :, freq_idx, :] = nsamples[iblt, :, ifreq, :]

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_ints_irregular_multi2(uv_uvh5):
    """
    Test writing a uvh5 file using irregular interval for freq and pol and integer dtype.
    """
    full_uvh5 = uv_uvh5
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # define freqs and pols
    freq_inds = [0, 1, 2, 7]
    pol_inds = [0, 1, 3]
    data_shape = (full_uvh5.Nblts, 1, len(freq_inds), len(pol_inds))
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    for ifreq, freq_idx in enumerate(freq_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            data[:, :, ifreq, ipol] = full_uvh5.data_array[:, :, freq_idx, pol_idx]
            flags[:, :, ifreq, ipol] = full_uvh5.flag_array[:, :, freq_idx, pol_idx]
            nsamples[:, :, ifreq, ipol] = full_uvh5.nsample_array[:, :, freq_idx, pol_idx]
    uvtest.checkWarnings(
        partial_uvh5.write_uvh5_part,
        [partial_testfile, data, flags, nsamples],
        {'freq_chans': freq_inds, 'polarizations': full_uvh5.polarization_array[pol_inds]},
        nwarnings=2,
        message=[
            'Selected frequencies are not evenly spaced',
            'Selected polarization values are not evenly spaced',
        ],
    )

    # also write the arrays to the partial object
    for ifreq, freq_idx in enumerate(freq_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            partial_uvh5.data_array[:, :, freq_idx, pol_idx] = data[:, :, ifreq, ipol]
            partial_uvh5.flag_array[:, :, freq_idx, pol_idx] = flags[:, :, ifreq, ipol]
            partial_uvh5.nsample_array[:, :, freq_idx, pol_idx] = nsamples[:, :, ifreq, ipol]

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_ints_irregular_multi3(uv_uvh5):
    """
    Test writing a uvh5 file using irregular interval for blt and pol and integer dtype.
    """
    full_uvh5 = uv_uvh5
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # define blts and pols
    blt_inds = [0, 1, 2, 7]
    pol_inds = [0, 1, 3]
    data_shape = (len(blt_inds), 1, full_uvh5.Nfreqs, len(pol_inds))
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    for iblt, blt_idx in enumerate(blt_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            data[iblt, :, :, ipol] = full_uvh5.data_array[blt_idx, :, :, pol_idx]
            flags[iblt, :, :, ipol] = full_uvh5.flag_array[blt_idx, :, :, pol_idx]
            nsamples[iblt, :, :, ipol] = full_uvh5.nsample_array[blt_idx, :, :, pol_idx]
    uvtest.checkWarnings(
        partial_uvh5.write_uvh5_part,
        [partial_testfile, data, flags, nsamples],
        {'blt_inds': blt_inds, 'polarizations': full_uvh5.polarization_array[pol_inds]},
        message='Selected polarization values are not evenly spaced',
    )

    # also write the arrays to the partial object
    for iblt, blt_idx in enumerate(blt_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            partial_uvh5.data_array[blt_idx, :, :, pol_idx] = data[iblt, :, :, ipol]
            partial_uvh5.flag_array[blt_idx, :, :, pol_idx] = flags[iblt, :, :, ipol]
            partial_uvh5.nsample_array[blt_idx, :, :, pol_idx] = nsamples[iblt, :, :, ipol]

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


def test_uvh5_partial_write_ints_irregular_multi4(uv_uvh5):
    """
    Test writing a uvh5 file using irregular interval for all axes and integer dtype.
    """
    full_uvh5 = uv_uvh5
    partial_uvh5 = full_uvh5.copy()
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # define blts and freqs
    blt_inds = [0, 1, 2, 7]
    freq_inds = [0, 2, 3, 4]
    pol_inds = [0, 1, 3]
    data_shape = (len(blt_inds), 1, len(freq_inds), len(pol_inds))
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    for iblt, blt_idx in enumerate(blt_inds):
        for ifreq, freq_idx in enumerate(freq_inds):
            for ipol, pol_idx in enumerate(pol_inds):
                data[iblt, :, ifreq, ipol] = full_uvh5.data_array[blt_idx, :, freq_idx, pol_idx]
                flags[iblt, :, ifreq, ipol] = full_uvh5.flag_array[blt_idx, :, freq_idx, pol_idx]
                nsamples[iblt, :, ifreq, ipol] = full_uvh5.nsample_array[blt_idx, :, freq_idx, pol_idx]
    uvtest.checkWarnings(
        partial_uvh5.write_uvh5_part,
        [partial_testfile, data, flags, nsamples],
        {
            'blt_inds': blt_inds,
            'freq_chans': freq_inds,
            'polarizations': full_uvh5.polarization_array[pol_inds],
        },
        nwarnings=2,
        message=[
            'Selected frequencies are not evenly spaced',
            'Selected polarization values are not evenly spaced',
        ],
    )

    # also write the arrays to the partial object
    for iblt, blt_idx in enumerate(blt_inds):
        for ifreq, freq_idx in enumerate(freq_inds):
            for ipol, pol_idx in enumerate(pol_inds):
                partial_uvh5.data_array[blt_idx, :, freq_idx, pol_idx] = data[iblt, :, ifreq, ipol]
                partial_uvh5.flag_array[blt_idx, :, freq_idx, pol_idx] = flags[iblt, :, ifreq, ipol]
                partial_uvh5.nsample_array[blt_idx, :, freq_idx, pol_idx] = nsamples[iblt, :, ifreq, ipol]

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


@pytest.mark.skipif(not six.PY3, reason="Skipping. This test is only relevant in python3.")
def test_antenna_names_not_list(uv_uvfits):
    """Test if antenna_names is cast to an array, dimensions are preserved in np.string_ call during uvh5 write."""
    uv_in = uv_uvfits
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits_ant_names.uvh5')

    # simulate a user defining antenna names as an array of unicode
    uv_in.antenna_names = np.array(uv_in.antenna_names, dtype='U')

    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)

    # recast as list since antenna names should be a list and will be cast as list on read
    uv_in.antenna_names = uv_in.antenna_names.tolist()
    assert uv_in == uv_out

    # clean up
    os.remove(testfile)

    return


def test_eq_coeffs_roundtrip(uv_uvfits):
    """Test reading and writing objects with eq_coeffs defined"""
    uv_in = uv_uvfits
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, "test", "outtest_eq_coeffs.uvh5")
    uv_in.eq_coeffs = np.ones((uv_in.Nants_telescope, uv_in.Nfreqs))
    uv_in.eq_coeffs_convention = "divide"
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)

    assert uv_in == uv_out

    # clean up
    os.remove(testfile)

    return

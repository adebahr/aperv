import os
import numpy as np
import subprocess

from configparser import ConfigParser

from astropy.table import Table
from astropy.io import fits as pyfits
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
from reproject import reproject_interp
from radio_beam import EllipticalGaussian2DKernel, Beam, Beams, commonbeam
from scipy.fft import ifft2, ifftshift
from astropy.convolution import convolve
import astropy.units as u


def load_config(config_object, file_=None):
    """
    Function to load the config file
    """
    config = ConfigParser()  # Initialise the config parser
    config.read_file(open(file_))
    for s in config.sections():
        for o in config.items(s):
            setattr(config_object, o[0], eval(o[1]))
    return config  # Save the loaded config file as defaults for later usage


def set_dirs(self):
    """
    Creates the directory names for the subdirectories to make scripting easier
    """
    # General data directory
    self.datadir = os.path.join(self.basedir, self.obsid)

    # Working directories for script
    self.polworkdir = os.path.join(self.datadir, self.mossubdir, self.mospoldir)
    self.polmosaicdir = os.path.join(self.polworkdir, 'mosaic')
    self.polanalysisdir = os.path.join(self.polworkdir, 'analysis')

    # Directories for the analysis of the leakage
    self.polanalysisleakagedir = os.path.join(self.polanalysisdir, 'leakage')
    self.polanalysisleakagedirV = os.path.join(self.polanalysisleakagedir, 'V')
    self.polanalysisleakagedirQ = os.path.join(self.polanalysisleakagedir, 'Q')

    # Working directory for subtracting the central sources and leakage contribution
    self.leakagesubtractdir = os.path.join(self.polworkdir, 'subtract')

    # Working directories for generating the mosaics
    self.contworkdir = os.path.join(self.datadir, self.mossubdir, self.moscontdir)
    self.contmosaicdir = os.path.join(self.contworkdir, 'mosaic')
    self.conttempdir = os.path.join(self.contworkdir, 'temp')

    self.leakageworkdir = os.path.join(self.datadir, self.mossubdir, self.mosleakdir)
    self.leakagemosaicdir = os.path.join(self.leakageworkdir, 'mosaic')
    self.leakagetempdir = os.path.join(self.leakageworkdir, 'temp')

    self.circpolworkdir = os.path.join(self.datadir, self.mossubdir, self.moscircdir)
    self.circpolmosaicdir = os.path.join(self.circpolworkdir, 'mosaic')
    self.circpoltempdir = os.path.join(self.circpolworkdir, 'temp')
    self.circpolanalysisdir = os.path.join(self.circpolworkdir, 'mosaic/analysis')



def gen_dirs(self):
    """
    Creates the necessary directories
    """
    if not os.path.isdir(self.leakagesubtractdir):
        os.makedirs(self.leakagesubtractdir)
    if not os.path.isdir(self.polanalysisleakagedir):
        os.makedirs(self.polanalysisleakagedir)
    if not os.path.isdir(self.polanalysisleakagedirV):
        os.makedirs(self.polanalysisleakagedirV)
    if not os.path.isdir(self.polanalysisleakagedirQ):
        os.makedirs(self.polanalysisleakagedirQ)
    if not os.path.isdir(self.contmosaicdir):
        os.makedirs(self.contmosaicdir)
    if not os.path.isdir(self.conttempdir):
        os.makedirs(self.conttempdir)
    if not os.path.isdir(self.leakageworkdir):
        os.makedirs(self.leakageworkdir)
    if not os.path.isdir(self.leakagemosaicdir):
        os.makedirs(self.leakagemosaicdir)
    if not os.path.isdir(self.leakagetempdir):
        os.makedirs(self.leakagetempdir)
    if not os.path.isdir(self.circpolmosaicdir):
        os.makedirs(self.circpolmosaicdir)
    if not os.path.isdir(self.circpoltempdir):
        os.makedirs(self.circpoltempdir)
    if not os.path.isdir(self.circpolanalysisdir):
        os.makedirs(self.circpolanalysisdir)


def load_catalogue(catalogue, meta=True):
    """
    Loads a catalogue file using astropy.table
    catalogue: Catalogue textfile created by pybdsf
    returns: catalogue as astropy table array
    """
    cat = Table.read(catalogue, format='ascii')
    if meta:
        keys = cat.meta['comments'][4].split(' ')
        cat = Table.read(catalogue, names=keys, format='ascii')
    else:
        pass
    return cat
#    f = open(catalogue)
#    lines = f.readlines()
#    header_line = lines[5].rstrip('\n').lstrip('# ')
#    col_names = header_line.split(' ')
#    t = Table.read(catalogue, format='ascii', names=col_names)
#    return t


def load_pointing_centres(pfile, nan=False):
    """
    Loads the pointing centre file and removes all nan values
    pfile: Pointing centre file from the mosaicking
    returns: Pointing centres with RA and DEC as two numpy arrays
    """
    parray = np.loadtxt(pfile)
    ra = parray[:,1]
    dec = parray[:,2]
    if nan:
        pass
    else:
        ra = ra[~np.isnan(ra)]
        dec = dec[~np.isnan(dec)]
    return ra, dec


def make_source_id(ra, dec, prefix):
    """
    Generate the name of the source using its ra and dec coordinates
    ra (float): RA in deg
    dec (float): DEC in deg
    return (string): source name
    """
    RA_str = str(np.around(ra, decimals=3))
    DEC_str = str(np.around(dec, decimals=3))
    RA_list = RA_str.split('.')
    DEC_list = DEC_str.split('.')
    len_RA_str = len(RA_list[-1])
    len_DEC_str = len(DEC_list[-1])
    last_string = '{:<03}'
    if len_RA_str < 3:
        RA = str(RA_list[0]) + '.' + last_string.format(str(RA_list[1]))
    else:
        RA = RA_str
    if len_DEC_str < 3:
        DEC = str(DEC_list[0]) + '.' + last_string.format(str(DEC_list[1]))
    else:
        DEC = DEC_str
    sourcestring = prefix + '_' + RA + '+' + DEC
    return sourcestring


def remove_dims(infile, outfile):
    with pyfits.open(infile) as infile_hdu:
        infile_data = infile_hdu[0].data
        infile_hdr = infile_hdu[0].header
        outfile_data = np.squeeze(infile_data)
        infile_hdr['NAXIS'] = 2
        for keyword in ['CRVAL3','CDELT3','CRPIX3','CUNIT3','CTYPE3','NAXIS3','CRVAL4','CDELT4','CRPIX4','CUNIT4','CTYPE4','NAXIS4']:
            try:
                del infile_hdr[keyword]
            except:
                pass
        pyfits.writeto(outfile, outfile_data, header=infile_hdr, overwrite=True)


def invert_values(infile, outfile):
    with pyfits.open(infile) as infile_hdu:
        infile_data = infile_hdu[0].data
        infile_hdr = infile_hdu[0].header
        outfile_data = -1.0 * infile_data
        pyfits.writeto(outfile, outfile_data, header=infile_hdr, overwrite=True)


def make_cutout(image, ra, dec, size):
    """
    Generates a cutout with an updated header for an image
    """
    with pyfits.open(image, memmap=0) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)
        skycoord = SkyCoord(ra, dec, unit="deg")
        cutout = Cutout2D(hdu.data, position=skycoord, size=size, wcs=wcs, mode='partial', fill_value=np.nan)
        hdu.data = cutout.data
        hdu.header.update(cutout.wcs.to_header())
        cutout_filename = image.rstrip('.fits') + '_cutout.fits'
        hdu.writeto(cutout_filename, overwrite=True)


def get_common_psf(veri):
    """
    Common psf for the list of fits files
    """
    bmajes = []
    bmines = []
    bpas = []
    for f in veri:
        ih = pyfits.getheader(f)
        bmajes.append(ih['BMAJ'])
        bmines.append(ih['BMIN'])
        bpas.append(ih['BPA'])
    bmajarr = np.array(bmajes)
    bminarr = np.array(bmines)
    bpaarr = np.array(bpas)
    for i in range(0, len(bmajes) - 1):
        ni = i + 1
        beams = Beams((bmajarr[[i, ni]]) * u.deg, (bminarr[[i, ni]]) * u.deg, bpaarr[[i, ni]] * u.deg)
        common = commonbeam.commonbeam(beams)
        bmajarr[ni] = common.major/u.deg
        bminarr[ni] = common.minor / u.deg
        bpaarr[ni] = common.pa / u.deg
        common = Beam.__new__(Beam, major=common.major * 1.01, minor=common.minor * 1.01, pa=common.pa)
        print('Increased final smallest common beam by 1 %')
        print('The final smallest common beam is ' + str(common))
    return common


def reproject_image(infile, template, outfile):
    with pyfits.open(infile) as infile_hdu:
        with pyfits.open(template) as template_hdu:
            array, footprint = reproject_interp(infile_hdu[0], template_hdu[0].header)
            pyfits.writeto(outfile, array, header=template_hdu[0].header, overwrite=True)


def get_beam(self):
    with pyfits.open(self.circpolanalysisdir + '/CP.fits') as hdu:
        hdu_header = hdu[0].header
        bmaj = hdu_header['BMAJ']
        bmin = hdu_header['BMIN']
    return bmaj, bmin


def get_beam_mosaic(image):
    with pyfits.open(image) as hdu:
        hdu_header = hdu[0].header
        bmaj = hdu_header['BMAJ']
        bmin = hdu_header['BMIN']
    return bmaj, bmin


def get_pixsize(image):
    with pyfits.open(image) as hdu:
        hdu_header = hdu[0].header
        pixsize = hdu_header['CDELT1']
    return pixsize


def get_wcs(image):
    with pyfits.open(image, memmap=0) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)
    return wcs


def get_data(image):
    with pyfits.open(image, memmap=0) as hdul:
        hdu = hdul[0]
        data = hdu.data
    return data


def make_tmp_copy(fname):
    base, ext = os.path.splitext(fname)
    tempname = fname.replace(ext, '_tmp{}'.format(ext))
    subprocess.call('cp {} {}'.format(fname, tempname), shell=True)
    return tempname
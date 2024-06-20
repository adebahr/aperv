import os
import glob
import shutil

import astropy.io.fits as pyfits
import numpy as np

from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd, find_optimal_celestial_wcs
from astropy.nddata import Cutout2D

import utilv
import fits_magic_v as fm

def generate_mosaics(self):
    """
    Generates the Stokes V (before and after leakage subtraction) and Stokes I mosaics
    """
    copy_images(self)
    copy_primary_beams(self)
    generate_continuum_mosaic(self)
    generate_circpol_mosaic(self)
#    generate_leakage_mosaic(self)

def copy_images(self):
    """
    Copy all the neccessary images to generate the mosaics to the temporary working directories
    """
    for beam in range(40):
        clist = glob.glob(os.path.join(self.datadir, str(beam).zfill(2), 'continuum') + '/image_mf_[0-9][0-9].fits')
        if len(clist) > 0 and os.path.isfile(os.path.join(self.basedir, self.obsid, str(beam).zfill(2), 'polarisation') + '/image_mf_V.fits'):
            bmaj, bmin = fm.get_beam(os.path.join(self.basedir, self.obsid, str(beam).zfill(2), 'polarisation') + '/image_mf_V.fits')
            rms = fm.get_rms(os.path.join(self.basedir, self.obsid, str(beam).zfill(2), 'polarisation') + '/image_mf_V.fits')
            if bmaj <= self.circ_bmaj and bmin <= self.circ_bmin and rms <= self.circ_rmsclip:
                tpimage = glob.glob(os.path.join(self.datadir, str(beam).zfill(2), 'continuum') + '/image_mf_[0-9][0-9].fits')
                shutil.copy(tpimage[0], self.conttempdir + '/I_' + str(beam).zfill(2) + '.fits')
                if os.path.isfile(os.path.join(self.datadir, 'param_' + str(beam).zfill(2) + '_preflag_3C286.npy')):
#                    shutil.copy(self.leakagesubtractdir + '/' + str(beam).zfill(2) + '_cssub.Vrstr.fits', self.circpoltempdir + '/V_' + str(beam).zfill(2) + '.fits')
                    shutil.copy(os.path.join(self.basedir, self.obsid, str(beam).zfill(2), 'polarisation') + '/image_mf_V.fits', self.circpoltempdir + '/V_' + str(beam).zfill(2) + '.fits')
                elif os.path.isfile(os.path.join(self.datadir, 'param_' + str(beam).zfill(2) + '_preflag_3C138.npy')):
#                    utilv.invert_values(self.leakagesubtractdir + '/' + str(beam).zfill(2) + '_cssub.Vrstr.fits', self.circpoltempdir + '/V_' + str(beam).zfill(2) + '.fits')
                    utilv.invert_values(os.path.join(self.basedir, self.obsid, str(beam).zfill(2), 'polarisation') + '/image_mf_V.fits', self.circpoltempdir + '/V_' + str(beam).zfill(2) + '.fits')
                else:
                    print('No parameter file for beam ' + str(beam).zfill(2) + ' available.')
#                shutil.copy(self.leakagesubtractdir + '/V_' + str(beam).zfill(2) + '_lmsub.Vrstr.fits', self.leakagetempdir + '/V_' + str(beam).zfill(2) + '.fits')
            else:
                print('Images for Beam ' + str(beam).zfill(2) + ' do not fulfill the set criteria!')
        else:
            print('No continuum image for beam ' + str(beam).zfill(2) + ' available.')


def copy_primary_beams(self):
    """
    Copies the primary beams from the Gaussian derivation method to the local working directories
    """
    for beam in range(40):
        if os.path.isfile(self.conttempdir + '/I_' + str(beam).zfill(2) + '.fits'):
            # Get the frequency from the image
            hdu_cont = pyfits.open(self.conttempdir + '/I_' + str(beam).zfill(2) + '.fits')
            freq = hdu_cont[0].header['CRVAL3']
            # Get the cellsize from the beam images and recalculate it based on the frequency of the image
            hdu_beam = pyfits.open(self.beamsrcdir + str(beam).zfill(2) + '_gp_avg_orig.fits')
            hdu_beam_hdr = hdu_beam[0].header
            hdu_beam_data = hdu_beam[0].data
            cs1 = hdu_beam_hdr['CDELT1']
            cs2 = hdu_beam_hdr['CDELT2']
            new_cs1 = cs1 * (1.36063551903e09 / freq)
            new_cs2 = cs2 * (1.36063551903e09 / freq)
            hdu_beam_hdr['CDELT1'] = new_cs1
            hdu_beam_hdr['CDELT2'] = new_cs2
            # Write the new not regridded beam to a temporary file
            pyfits.writeto(self.conttempdir + '/B_' + str(beam).zfill(2) + '.fits', data=hdu_beam_data, header=hdu_beam_hdr, overwrite=True)
            pyfits.writeto(self.circpoltempdir + '/B_' + str(beam).zfill(2) + '.fits', data=hdu_beam_data, header=hdu_beam_hdr, overwrite=True)
            pyfits.writeto(self.leakagetempdir + '/B_' + str(beam).zfill(2) + '.fits', data=hdu_beam_data, header=hdu_beam_hdr, overwrite=True)


def generate_continuum_mosaic(self):
    """
    Generates the total power continuum mosaic
    """
    images = sorted(glob.glob(self.conttempdir + '/I_[0-9][0-9].fits'))
    pbimages = sorted(glob.glob(self.conttempdir + '/B_[0-9][0-9].fits'))

    # Get the common psf
    common_psf = utilv.get_common_psf(images)
    print('Clipping primary beam response at the %f level', str(self.circ_pbclip))

    corrimages = []  # to mosaic
    uncorrimages = []
    pbweights = []  # of the pixels
    freqs = []
    # weight_images = []
    for img, pb in zip(images, pbimages):
        print('Doing primary beam correction for Beam ' + str(pb.split('/')[-1].replace('.fits', '').lstrip('B_')))
        # prepare the images (squeeze, transfer_coordinates, reproject, regrid pbeam, correct...)
        with pyfits.open(img) as f:
            imheader = f[0].header
            freqs.append(imheader['CRVAl3'])
            tg = imheader['OBJECT']
        # convolution with common psf
        reconvolved_image = img.replace('.fits', '_reconv_tmp.fits')
        reconvolved_image = fm.fits_reconvolve_psf(img, common_psf, out=reconvolved_image)
        # PB correction
        tmpimg = utilv.make_tmp_copy(reconvolved_image)
        tmppb = utilv.make_tmp_copy(pb)
        tmpimg = fm.fits_squeeze(tmpimg)  # remove extra dimensions
        tmppb = fm.fits_transfer_coordinates(tmpimg, tmppb)  # transfer_coordinates
        tmppb = fm.fits_squeeze(tmppb)  # remove extra dimensions
        with pyfits.open(tmpimg) as f:
            imheader = f[0].header
        with pyfits.open(tmppb) as f:
            pbhdu = f[0]
            pbheader = f[0].header
            pbarray = f[0].data
            if (imheader['CRVAL1'] != pbheader['CRVAL1']) or (imheader['CRVAL2'] != pbheader['CRVAL2']) or (
                    imheader['CDELT1'] != pbheader['CDELT1']) or (imheader['CDELT2'] != pbheader['CDELT2']):
                pbarray, reproj_footprint = reproject_interp(pbhdu, imheader)
            else:
                pass
        pbarray = np.float32(pbarray)
        pbarray[pbarray < self.circ_pbclip] = np.nan
        pb_regr_repr = tmppb.replace('_tmp.fits', '_repr_repr.fits')
        pyfits.writeto(pb_regr_repr, pbarray, imheader, overwrite=True)
        img_corr = reconvolved_image.replace('.fits', '_pbcorr.fits')
        img_uncorr = reconvolved_image.replace('.fits', '_uncorr.fits')

        img_corr = fm.fits_operation(tmpimg, pbarray, operation='/', out=img_corr)
        img_uncorr = fm.fits_operation(img_corr, pbarray, operation='*', out=img_uncorr)

        # cropping
        cropped_image = img.replace('.fits', '_mos.fits')
        cropped_image, cutout = fm.fits_crop(img_corr, out=cropped_image)

        uncorr_cropped_image = img.replace('.fits', '_uncorr.fits')
        uncorr_cropped_image, _ = fm.fits_crop(img_uncorr, out=uncorr_cropped_image)

        corrimages.append(cropped_image)
        uncorrimages.append(uncorr_cropped_image)

        # primary beam weights
        wg_arr = pbarray  #
        wg_arr[np.isnan(wg_arr)] = 0  # the NaNs weight 0
        wg_arr = wg_arr ** 2 / np.nanmax(wg_arr ** 2)  # normalize
        wcut = Cutout2D(wg_arr, cutout.input_position_original, cutout.shape)
        pbweights.append(wcut.data)

    # create the wcs and footprint for the output mosaic
    print('Generating primary beam corrected continuum mosaic.')
    wcs_out, shape_out = find_optimal_celestial_wcs(corrimages, auto_rotate=False, reference=None)

    array, footprint = reproject_and_coadd(corrimages, wcs_out, shape_out=shape_out, reproject_function=reproject_interp, input_weights=pbweights)
    array2, _ = reproject_and_coadd(uncorrimages, wcs_out, shape_out=shape_out,
                                           reproject_function=reproject_interp, input_weights=pbweights)

    array = np.float32(array)
    array2 = np.float32(array2)

    # insert common PSF into the header
    psf = common_psf.to_header_keywords()
    hdr = wcs_out.to_header()
    hdr.insert('RADESYS', ('FREQ', np.nanmean(freqs)))
    hdr.insert('RADESYS', ('BMAJ', psf['BMAJ']))
    hdr.insert('RADESYS', ('BMIN', psf['BMIN']))
    hdr.insert('RADESYS', ('BPA', psf['BPA']))

    # insert units to header:
    hdr.insert('RADESYS', ('BUNIT', 'JY/BEAM'))

    # Write out the mosaic
    pyfits.writeto(self.contmosaicdir + '/' + str(tg).upper() + '.fits', data=array, header=hdr, overwrite=True)
    pyfits.writeto(self.contmosaicdir + '/' + str(tg).upper() + '_uncorr.fits', data=array2, header=hdr, overwrite=True)


def generate_circpol_mosaic(self):
    """
    Generates the uncorrected Stokes V mosaic
    """
    images = sorted(glob.glob(self.circpoltempdir + '/V_[0-9][0-9].fits'))
    pbimages = sorted(glob.glob(self.circpoltempdir + '/B_[0-9][0-9].fits'))

    # Get the common psf
    common_psf = utilv.get_common_psf(images)
    print('Clipping primary beam response at the %f level', str(self.circ_pbclip))

    corrimages = []  # to mosaic
    pbweights = []  # of the pixels
    freqs = []
    # weight_images = []
    for img, pb in zip(images, pbimages):
        print('Doing primary beam correction for Beam ' + str(pb.split('/')[-1].replace('.fits', '').lstrip('B_')))
        # prepare the images (squeeze, transfer_coordinates, reproject, regrid pbeam, correct...)
        with pyfits.open(img) as f:
            imheader = f[0].header
            freqs.append(imheader['CRVAl3'])
            tg = imheader['OBJECT']
        # convolution with common psf
        reconvolved_image = img.replace('.fits', '_reconv_tmp.fits')
        reconvolved_image = fm.fits_reconvolve_psf(img, common_psf, out=reconvolved_image)
        # PB correction
        tmpimg = utilv.make_tmp_copy(reconvolved_image)
        tmppb = utilv.make_tmp_copy(pb)
        tmpimg = fm.fits_squeeze(tmpimg)  # remove extra dimensions
        tmppb = fm.fits_transfer_coordinates(tmpimg, tmppb)  # transfer_coordinates
        tmppb = fm.fits_squeeze(tmppb)  # remove extra dimensions
        with pyfits.open(tmpimg) as f:
            imheader = f[0].header
        with pyfits.open(tmppb) as f:
            pbhdu = f[0]
            pbheader = f[0].header
            pbarray = f[0].data
            if (imheader['CRVAL1'] != pbheader['CRVAL1']) or (imheader['CRVAL2'] != pbheader['CRVAL2']) or (
                    imheader['CDELT1'] != pbheader['CDELT1']) or (imheader['CDELT2'] != pbheader['CDELT2']):
                pbarray, reproj_footprint = reproject_interp(pbhdu, imheader)
            else:
                pass
        pbarray = np.float32(pbarray)
        pbarray[pbarray < self.circ_pbclip] = np.nan
        pb_regr_repr = tmppb.replace('_tmp.fits', '_repr_repr.fits')
        pyfits.writeto(pb_regr_repr, pbarray, imheader, overwrite=True)
        img_corr = reconvolved_image.replace('.fits', '_pbcorr.fits')
        img_corr = fm.fits_operation(tmpimg, pbarray, operation='/', out=img_corr)

        # cropping
        cropped_image = img.replace('.fits', '_mos.fits')
        cropped_image, cutout = fm.fits_crop(img_corr, out=cropped_image)

        corrimages.append(cropped_image)

        # primary beam weights
        wg_arr = pbarray  #
        wg_arr[np.isnan(wg_arr)] = 0  # the NaNs weight 0
        wg_arr = wg_arr ** 2 / np.nanmax(wg_arr ** 2)  # normalize
        wcut = Cutout2D(wg_arr, cutout.input_position_original, cutout.shape)
        pbweights.append(wcut.data)

    # create the wcs and footprint for output mosaic
    print('Generating primary beam corrected circular polarisation mosaic.')
    wcs_out, shape_out = find_optimal_celestial_wcs(corrimages, auto_rotate=False, reference=None)

    array, footprint = reproject_and_coadd(corrimages, wcs_out, shape_out=shape_out, reproject_function=reproject_interp, input_weights=pbweights)
    array = np.float32(array)

    # insert common PSF into the header
    psf = common_psf.to_header_keywords()
    hdr = wcs_out.to_header()
    hdr.insert('RADESYS', ('FREQ', np.nanmean(freqs)))
    hdr.insert('RADESYS', ('BMAJ', psf['BMAJ']))
    hdr.insert('RADESYS', ('BMIN', psf['BMIN']))
    hdr.insert('RADESYS', ('BPA', psf['BPA']))

    # insert units to header:
    hdr.insert('RADESYS', ('BUNIT', 'JY/BEAM'))

    # Write out the mosaic
    pyfits.writeto(self.circpolmosaicdir + '/' + str(tg).upper() + '.fits', data=array, header=hdr, overwrite=True)


def generate_leakage_mosaic(self):
    """
    Generates the leakage corrected Stokes V mosaic
    """
    images = sorted(glob.glob(self.leakagetempdir + '/V_[0-9][0-9].fits'))
    pbimages = sorted(glob.glob(self.leakagetempdir + '/B_[0-9][0-9].fits'))

    # Get the common psf
    common_psf = utilv.get_common_psf(images)
    print('Clipping primary beam response at the %f level', str(self.circ_pbclip))

    corrimages = []  # to mosaic
    pbweights = []  # of the pixels
    freqs = []
    # weight_images = []
    for img, pb in zip(images, pbimages):
        print('Doing primary beam correction for Beam ' + str(pb.split('/')[-1].replace('.fits', '').lstrip('B_')))
        # prepare the images (squeeze, transfer_coordinates, reproject, regrid pbeam, correct...)
        with pyfits.open(img) as f:
            imheader = f[0].header
            freqs.append(imheader['CRVAl3'])
            tg = imheader['OBJECT']
        # convolution with common psf
        reconvolved_image = img.replace('.fits', '_reconv_tmp.fits')
        reconvolved_image = fm.fits_reconvolve_psf(img, common_psf, out=reconvolved_image)
        # PB correction
        tmpimg = utilv.make_tmp_copy(reconvolved_image)
        tmppb = utilv.make_tmp_copy(pb)
        tmpimg = fm.fits_squeeze(tmpimg)  # remove extra dimensions
        tmppb = fm.fits_transfer_coordinates(tmpimg, tmppb)  # transfer_coordinates
        tmppb = fm.fits_squeeze(tmppb)  # remove extra dimensions
        with pyfits.open(tmpimg) as f:
            imheader = f[0].header
        with pyfits.open(tmppb) as f:
            pbhdu = f[0]
            pbheader = f[0].header
            pbarray = f[0].data
            if (imheader['CRVAL1'] != pbheader['CRVAL1']) or (imheader['CRVAL2'] != pbheader['CRVAL2']) or (
                    imheader['CDELT1'] != pbheader['CDELT1']) or (imheader['CDELT2'] != pbheader['CDELT2']):
                pbarray, reproj_footprint = reproject_interp(pbhdu, imheader)
            else:
                pass
        pbarray = np.float32(pbarray)
        pbarray[pbarray < self.circ_pbclip] = np.nan
        pb_regr_repr = tmppb.replace('_tmp.fits', '_repr_repr.fits')
        pyfits.writeto(pb_regr_repr, pbarray, imheader, overwrite=True)
        img_corr = reconvolved_image.replace('.fits', '_pbcorr.fits')
        img_corr = fm.fits_operation(tmpimg, pbarray, operation='/', out=img_corr)

        # cropping
        cropped_image = img.replace('.fits', '_mos.fits')
        cropped_image, cutout = fm.fits_crop(img_corr, out=cropped_image)

        corrimages.append(cropped_image)

        # primary beam weights
        wg_arr = pbarray  #
        wg_arr[np.isnan(wg_arr)] = 0  # the NaNs weight 0
        wg_arr = wg_arr ** 2 / np.nanmax(wg_arr ** 2)  # normalize
        wcut = Cutout2D(wg_arr, cutout.input_position_original, cutout.shape)
        pbweights.append(wcut.data)

    # create the wcs and footprint for output mosaic
    print('Generating primary beam and leakage corrected circular polarisation mosaic.')
    wcs_out, shape_out = find_optimal_celestial_wcs(corrimages, auto_rotate=False, reference=None)

    array, footprint = reproject_and_coadd(corrimages, wcs_out, shape_out=shape_out, reproject_function=reproject_interp, input_weights=pbweights)
    array = np.float32(array)

    # insert common PSF into the header
    psf = common_psf.to_header_keywords()
    hdr = wcs_out.to_header()
    hdr.insert('RADESYS', ('FREQ', np.nanmean(freqs)))
    hdr.insert('RADESYS', ('BMAJ', psf['BMAJ']))
    hdr.insert('RADESYS', ('BMIN', psf['BMIN']))
    hdr.insert('RADESYS', ('BPA', psf['BPA']))

    # insert units to header:
    hdr.insert('RADESYS', ('BUNIT', 'JY/BEAM'))

    # Write out the mosaic
    pyfits.writeto(self.leakagemosaicdir + '/' + str(tg).upper() + '.fits', data=array, header=hdr, overwrite=True)

# def generate_cfgfile(self):
#     """
#     Generates the cfg-file for the two mosaics
#     """
#     c0 = '[CONTINUUM_MOSAIC]\n'
#     c1 = 'basedir = "' + self.basedir + '"\n'
#     c2 = 'obsid = "' + self.obsid + '"\n'
#     c3 = 'mossubdir = "' + self.mossubdir + '"\n'
#     c4 = 'moscontdir = "' + self.moscontdir + '"\n'
#     c5 = 'mospoldir = "' + self.mosleakdir + '"\n'
#     c6 = 'moscircdir = "' + self.moscircdir + '"\n'
#     c7 = 'beamsrcdir = "' + self.beamsrcdir + '"\n'
#     c8 = 'lbeamsrcdir = "' + self.lbeamsrcdir + '"\n'
#     c9 = 'qadir = "' + self.qadir + '"\n'
#     c10 = 'cont_mode = "' + self.circ_mode + '"\n'
#     c11 = 'cont_mode_fixed_fwhm = "' + self.circ_mode_fixed_fwhm + '"\n'
#     c12 = 'cont_pbtype = "' + self.circ_pbtype + '"\n'
#     c13 = 'cont_pbclip = ' + str(self.circ_pbclip) + '\n'
#     c14 = 'cont_use00 = ' + str(self.circ_use00) + '\n'
#     c15 = 'cont_rmsclip = ' + str(self.circ_rmsclip) + '\n'
#     c16 = 'cont_bmaj = ' + str(self.circ_bmaj) + '\n'
#     c17 = 'cont_bmin = ' + str(self.circ_bmin) + '\n'
#     cp0 = '[CIRCULAR_POLARISATION_MOSAIC]\n'
#     cp1 = 'basedir = "' + self.basedir + '"\n'
#     cp2 = 'obsid = "' + self.obsid + '"\n'
#     cp3 = 'mossubdir = "' + self.mossubdir + '"\n'
#     cp4 = 'moscontdir = "' + self.moscontdir + '"\n'
#     cp5 = 'mospoldir = "' + self.mosleakdir + '"\n'
#     cp6 = 'moscircdir = "' + self.moscircdir + '"\n'
#     cp7 = 'beamsrcdir = "' + self.beamsrcdir + '"\n'
#     cp8 = 'lbeamsrcdir = "' + self.lbeamsrcdir + '"\n'
#     cp9 = 'qadir = "' + self.qadir + '"\n'
#     cp10 = 'circ_mode = "' + self.circ_mode + '"\n'
#     cp11 = 'circ_mode_fixed_fwhm = "' + self.circ_mode_fixed_fwhm + '"\n'
#     cp12 = 'circ_pbtype = "' + self.circ_pbtype + '"\n'
#     cp13 = 'circ_pbclip = ' + str(self.circ_pbclip) + '\n'
#     cp14 = 'circ_use00 = ' + str(self.circ_use00) + '\n'
#     cp15 = 'circ_rmsclip = ' + str(self.circ_rmsclip) + '\n'
#     cp16 = 'circ_bmaj = ' + str(self.circ_bmaj) + '\n'
#     cp17 = 'circ_bmin = ' + str(self.circ_bmin) + '\n'
#     cfgstr = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15 + c16 + c17 + '\n' + cp0 + cp1 + cp2 + cp3 + cp4 + cp5 + cp6 + cp7 + cp8 + cp9 + cp10 + cp11 + cp12 + cp13 + cp14 + cp15 + cp16 + cp17
#     with open(os.path.join(self.basedir, self.obsid, self.mossubdir, self.mosleakdir) + self.obsid + ".cfg", "w") as cfg_file:
#         cfg_file.write(cfgstr)
#
#
# def generate_Stokes_V(self):
#     """
#     Generates the Stokes V mosaic
#     """
#     cfgfile = os.path.join(self.basedir, self.obsid, self.mossubdir, self.mosleakdir) + self.obsid + ".cfg"
#     circmosaic = circ_pol_mosaic.circ_pol_mosaic(cfgfile)
#     circmosaic.go()
#
#
# def generate_Stokes_I(self):
#     """
#     Generates the Stokes I mosaic
#     """
#     cfgfile = os.path.join(self.basedir, self.obsid, self.mossubdir, self.mosleakdir) + self.obsid + ".cfg"
#     contmosaic = continuum_mosaic.continuum_mosaic(cfgfile)
#     contmosaic.go()

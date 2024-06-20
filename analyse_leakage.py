import os
import glob
import shutil

import numpy as np
import astropy.io.fits as pyfits
from astropy.table import Table
import bdsf

from astropy.coordinates import Angle, SkyCoord
from regions import PixCoord, EllipsePixelRegion

import utilv
import fits_magic_v as fm


def analyse_leakage(self):
    copy_imagesV(self)
    correct_imagesV(self)
    write_pointing_centres(self)
    generate_catalogue_I(self)
    remove_pc(self)
    calc_leakage_catalogue_I(self)

def copy_imagesV(self):
    """
    Copy the images from the main data directory into the working directory
    """
    for beam in range(40):
    #for beam in range(2):
        clist = glob.glob(os.path.join(self.datadir, str(beam).zfill(2), 'continuum') + '/image_mf_[0-9][0-9].fits')
        if len(clist) > 0 and os.path.isfile(os.path.join(self.basedir, self.obsid, str(beam).zfill(2), 'polarisation') + '/image_mf_V.fits'):
            # Check the quality of the Stokes V image
            bmaj, bmin = fm.get_beam(os.path.join(self.basedir, self.obsid, str(beam).zfill(2), 'polarisation') + '/image_mf_V.fits')
            rms = fm.get_rms(os.path.join(self.basedir, self.obsid, str(beam).zfill(2), 'polarisation') + '/image_mf_V.fits')
            if bmaj <= self.circ_bmaj and bmin <= self.circ_bmin and rms <= self.circ_rmsclip:
                # Copy the total power images
                tpimage = glob.glob(os.path.join(self.datadir, str(beam).zfill(2), 'continuum') + '/image_mf_[0-9][0-9].fits')
                shutil.copy(tpimage[0], self.polanalysisleakagedirV + '/I_' + str(beam).zfill(2) + '.fits')
                # Copy the Stokes V images
                shutil.copy(os.path.join(self.datadir, str(beam).zfill(2), 'polarisation') + '/image_mf_V.fits', self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits')
    #            # Save a copy of the Stokes V image with all positive values for the source finder
    #            with pyfits.open(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits') as hdu:
    #                hdu_header = hdu[0].header
    #                hdu_data = np.fabs(hdu[0].data)
    #                pyfits.writeto(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_pos.fits', hdu_data, header=hdu_header, overwrite=True)
            else:
                print('Stokes V image for beam ' + str(beam).zfill(2) + ' does not fulfill the given requirements. Not using it.')
        else:
            print('Not all information available for beam ' + str(beam).zfill(2) + '. Discarding beam.')


def correct_imagesV(self):
    """
    Correct for the discrepancy of the sign in Stokes V depending on the observed calibrator
    3C286 shows the right sign checked via cross-matching with VLOTSS
    The sign of the observations using 3C138 is flipped
    """
    for beam in range(40):
    #for beam in range(2):
        if os.path.isfile(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits'):
            if os.path.isfile(os.path.join(self.datadir, 'param_' + str(beam).zfill(2) + '_preflag_3C286.npy')):
                pass
            elif os.path.isfile(os.path.join(self.datadir, 'param_' + str(beam).zfill(2) + '_preflag_3C138.npy')):
                utilv.invert_values(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits', self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits')
            else:
                print('No parameter file for beam ' + str(beam).zfill(2) + ' available.')
        else:
            print('No Stokes V image for beam ' + str(beam).zfill(2) + ' available.')


def write_pointing_centres(self):
    """
    Write a file with the central coordinates of each pointing used
    """
    coord_arr = np.full((40, 3), np.nan)
    coord_arr[:, 0] = np.arange(0, 40, 1)
    for beam in range(40):
    #for beam in range(2):
        if os.path.isfile(os.path.join(self.datadir, str(beam).zfill(2), 'polarisation/image_mf_V.fits')):
            vim = pyfits.open(os.path.join(self.datadir, str(beam).zfill(2), 'polarisation/image_mf_V.fits'))[0]
            vim_hdr = vim.header
            coord_arr[beam, 1] = vim_hdr['CRVAL1']
            coord_arr[beam, 2] = vim_hdr['CRVAL2']
    np.savetxt(self.polanalysisleakagedirV + '/pointings.txt', coord_arr, fmt=['%2s', '%1.13e', '%1.13e'], delimiter='\t')


def generate_catalogue_I(self):
    """
    Generate the catalogue for the individual beam Stokes I images using pybdsf
    """
    for beam in range(40):
    #for beam in range(2):
        if os.path.isfile(self.polanalysisleakagedirV + '/I_' + str(beam).zfill(2) + '.fits'):
            try:
                source_list_i = bdsf.process_image(self.polanalysisleakagedirV + '/I_' + str(beam).zfill(2) + '.fits', adaptive_rms_box=True, thresh_isl=7.0, thresh_pix=5.0, quiet=True)
                source_list_i.write_catalog(outfile=self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt', format='ascii', clobber=True)
                cat_i = utilv.load_catalogue(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt')
                islands = np.unique(cat_i['Isl_id'])
                RA_arr = np.full(len(islands), np.nan)
                E_RA_arr = np.full(len(islands), np.nan)
                DEC_arr = np.full(len(islands), np.nan)
                E_DEC_arr = np.full(len(islands), np.nan)
                Total_flux_arr = np.full(len(islands), np.nan)
                E_Total_flux_arr = np.full(len(islands), np.nan)
                Peak_flux_arr = np.full(len(islands), np.nan)
                E_Peak_flux_arr = np.full(len(islands), np.nan)
                Xposn_arr = np.full(len(islands), np.nan)
                E_Xposn_arr = np.full(len(islands), np.nan)
                Yposn_arr = np.full(len(islands), np.nan)
                E_Yposn_arr = np.full(len(islands), np.nan)
                Maj_arr = np.full(len(islands), np.nan)
                E_Maj_arr = np.full(len(islands), np.nan)
                Min_arr = np.full(len(islands), np.nan)
                E_Min_arr = np.full(len(islands), np.nan)
                PA_arr = np.full(len(islands), np.nan)
                E_PA_arr = np.full(len(islands), np.nan)
                Isl_rms_arr = np.full(len(islands), np.nan)
                # Remove the sources with multiple components
                for n, isl in enumerate(islands):
                    islidxs = np.where(isl == cat_i['Isl_id'])[0]
                    RA_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['RA']
                    E_RA_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['E_RA']
                    DEC_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['DEC']
                    E_DEC_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['E_DEC']
                    Total_flux_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['Total_flux']
                    E_Total_flux_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['E_Total_flux']
                    Peak_flux_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['Peak_flux']
                    E_Peak_flux_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['E_Peak_flux']
                    Xposn_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['Xposn']
                    E_Xposn_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['E_Xposn']
                    Yposn_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['Yposn']
                    E_Yposn_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['E_Yposn']
                    Maj_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['Maj']
                    E_Maj_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['E_Maj']
                    Min_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['Min']
                    E_Min_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['E_Min']
                    PA_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['PA']
                    E_PA_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['E_PA']
                    Isl_rms_arr[n] = cat_i[islidxs][np.argmax(cat_i['Peak_flux'][islidxs])]['Isl_rms']
                t = Table()
                t['RA'] = RA_arr
                t['E_RA'] = E_RA_arr
                t['DEC'] = DEC_arr
                t['E_DEC'] = E_DEC_arr
                t['Total_flux'] = Total_flux_arr
                t['E_Total_flux'] = E_Total_flux_arr
                t['Peak_flux'] = Peak_flux_arr
                t['E_Peak_flux'] = E_Peak_flux_arr
                t['Xposn'] = Xposn_arr
                t['E_Xposn'] = E_Xposn_arr
                t['Yposn'] = Yposn_arr
                t['E_Yposn'] = E_Yposn_arr
                t['Maj'] = Maj_arr
                t['E_Maj'] = E_Maj_arr
                t['Min'] = Min_arr
                t['E_Min'] = E_Min_arr
                t['PA'] = PA_arr
                t['E_PA'] = E_PA_arr
                t['Isl_rms'] = Isl_rms_arr
                t.write(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt', format='ascii', overwrite=True, comment=False)
            except:
                print('Beam ' + str(beam).zfill(2) + ' for ID ' + str(self.obsid) + ' was not successfully processed!')
        else:
            print('No total power image for Beam ' + str(beam).zfill(2) + ' for ID ' + str(self.obsid) + ' available. No catalogue will be created.')


def remove_pc(self):
    """
    Remove the sources at the pointing centres from the catalogue
    """
    for beam in range(40):
    #for beam in range(2):
        if os.path.isfile(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt'):
            cat = utilv.load_catalogue(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt', meta=False)
            ra, dec = utilv.load_pointing_centres(self.polanalysisleakagedirV + '/pointings.txt')
            # Initialise a table for removing sources
            remove_list = []
            PI_hdu = pyfits.open(self.polanalysisleakagedirV + '/I_' + str(beam).zfill(2) + '.fits')
            PI_hdr = PI_hdu[0].header
            bmaj = PI_hdr['BMAJ']
            cat_ra = cat['RA']
            cat_dec = cat['DEC']
            for s, source in enumerate(cat_ra):
                dist_arr = np.sqrt(np.square(ra - cat_ra[s]) + np.square(dec - cat_dec[s]))
                if np.any(dist_arr < (bmaj / 2.0)):
                    remove_list.append(s)
            cat.remove_rows([remove_list])
            cat.write(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt', format='ascii', overwrite=True, comment=False)


def calc_leakage_catalogue_I(self):
    """
    Use the central pixel of each source in the Stokes I catalogue to determine its value in the Stokes V image
    """
    for beam in range(40):
    #for beam in range(2):
        if os.path.isfile(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt'):
            cat_i = Table.read(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt', format='ascii')
            # Get the pixel size of the original total power image
            cdelt = utilv.get_pixsize(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits')
            V_pix_flux_arr = np.full(len(cat_i), np.nan)
            for c, comp in enumerate(cat_i):
                # Get the component size
                size = max(comp['Maj'], comp['Min'])
                utilv.remove_dims(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits', self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_nd.fits')
                utilv.make_cutout(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_nd.fits', comp['RA'], comp['DEC'], np.fabs((size/cdelt)*2.0))
                # Find the position of the component in the cutout
                wcs_cutout = utilv.get_wcs(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_nd_cutout.fits')
                data_cutout = utilv.get_data(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_nd_cutout.fits')
                skycoord = SkyCoord(comp['RA'], comp['DEC'], unit="deg")
                pixcoord = PixCoord.from_sky(skycoord, wcs_cutout)
                # Generate the elliptical mask
                ellipse_reg = EllipsePixelRegion(pixcoord, width=np.fabs(comp['Maj']/cdelt), height=np.fabs(comp['Min']/cdelt), angle=Angle(comp['PA'],'deg'))
                ellipse_mask = ellipse_reg.to_mask()
                data_cutout_masked = ellipse_mask.multiply(data_cutout, fill_value=np.nan)
#                pyfits.writeto(self.polanalysisleakagedirV + '/I_' + str(beam).zfill(2) + '_nd_cutout_mask.fits', data=data_cutout_masked)
                data_cutout_min = np.nanmin(data_cutout_masked)
                data_cutout_max = np.nanmax(data_cutout_masked)
                if np.fabs(data_cutout_min) > data_cutout_max:
                    V_pix_flux_arr[c] = data_cutout_min
                else:
                    V_pix_flux_arr[c] = data_cutout_max
            cat_i['V_pix_flux'] = V_pix_flux_arr
            cat_i.write(self.polanalysisleakagedirV + '/IV_cat_B' + str(beam).zfill(2) + '.txt', format='ascii', overwrite=True, comment=False)
        else:
            print('No catalogue for Beam ' + str(beam).zfill(2) + '!')
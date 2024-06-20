import os
import numpy as np
from astropy.io import fits as pyfits
from astropy.table import Table, vstack
import glob
import shutil

import bdsf

import utilv


def catalogue_v(self):
    copy_images(self)
    source_finding_v(self)
    gen_catalogue_v(self)
    source_finding_i(self)
    gen_catalogue_i(self)
    remove_pointing_centres(self)


def copy_images(self):
    cp_image = sorted(glob.glob(self.circpolmosaicdir + '/*.fits'))[0]
    cont_image = sorted(glob.glob(self.contmosaicdir + '/*.fits'))[0]
    cont_uncorr_image = sorted(glob.glob(self.contmosaicdir + '/*.fits'))[1]
    shutil.copyfile(cp_image, self.circpolanalysisdir + '/CP.fits')
    shutil.copyfile(cont_image, self.circpolanalysisdir + '/TP.fits')
    shutil.copyfile(cont_uncorr_image, self.circpolanalysisdir + '/TP_uncorr.fits')


def source_finding_v(self):
    """
    Run pybdsf on the Stokes V mosaic and an inverted version of it to also find the negative sources
    """
    cp_image = self.circpolanalysisdir + '/CP.fits'
    bmaj, bmin = utilv.get_beam_mosaic(cp_image)
    rmsbox = ((bmaj*3600.0)/3.0)*30.0
    rmsboxshift = ((bmaj*3600.0)/3.0)*(30.0/8.0)
    brightbox = ((bmaj*3600.0)/3.0)*12.0
    brightboxshift = ((bmaj*3600.0)/3.0)*(12.0/8.0)
    source_list_cp = bdsf.process_image(cp_image, adaptive_rms_box=True, adaptive_thresh=10.0, ncores=8, advanced_opts=True, mean_map='zero', rms_box=(rmsbox,rmsboxshift), rms_box_bright=(brightbox,brightboxshift), thresh='fdr', thresh_isl=4.0, thresh_pix=5.0, quiet=True)
    cp_image_inv = cp_image.rstrip('.fits') + '_inv.fits'
    cp_image_hdu = pyfits.open(cp_image)[0]
    cp_image_data = cp_image_hdu.data
    cp_image_hdr = cp_image_hdu.header
    cp_image_data_inv = -1.0*cp_image_data
    pyfits.writeto(cp_image_inv, data=cp_image_data_inv, header=cp_image_hdr, overwrite=True)
    source_list_cp_inv = bdsf.process_image(cp_image_inv, adaptive_rms_box=True, adaptive_thresh=10.0, ncores=8, advanced_opts=True, mean_map='zero', rms_box=(rmsbox,rmsboxshift), rms_box_bright=(brightbox,brightboxshift), thresh='fdr', thresh_isl=4.0, thresh_pix=5.0, quiet=True)
    source_list_cp.write_catalog(outfile = self.circpolanalysisdir + '/CP_orig_cat.txt', format='ascii', clobber=True)
    source_list_cp_inv.write_catalog(outfile = self.circpolanalysisdir + '/CP_inv_cat.txt', format='ascii', clobber=True)


def gen_catalogue_v(self):
    """
    Write out and combine the two catalogues into one
    """
    # Load the catalogues
    catcp_orig = utilv.load_catalogue(self.circpolanalysisdir + '/CP_orig_cat.txt')
    catcp_inv = utilv.load_catalogue(self.circpolanalysisdir + '/CP_inv_cat.txt')
    # Correct the inverted catalogue for the negative sign for the fluxes
    catcp_inv['Total_flux'] = catcp_inv['Total_flux'] * -1.0
    catcp_inv['Peak_flux'] = catcp_inv['Peak_flux'] * -1.0
    catcp_inv['Isl_Total_flux'] = catcp_inv['Isl_Total_flux'] * -1.0
    max_id = np.max(catcp_orig['Isl_id'])
    # Add 1 to inverted catalogue since Isl_ids are the same
    catcp_inv['Isl_id'] = catcp_inv['Isl_id'] + max_id + 1
    # Combine the two catalogues
    catcp = vstack([catcp_orig, catcp_inv])
    catcp_filtered = catcp['Isl_id','RA','E_RA','DEC','E_DEC','Total_flux','E_Total_flux','Peak_flux','E_Peak_flux','S_Code','Isl_rms','DC_Maj','DC_Min','DC_PA']
    catcp_filtered.rename_column('RA', 'CP_RA_Comp')
    catcp_filtered.rename_column('E_RA', 'CP_RA_Comp_err')
    catcp_filtered.rename_column('DEC', 'CP_DEC_Comp')
    catcp_filtered.rename_column('E_DEC', 'CP_DEC_Comp_err')
    catcp_filtered.rename_column('Total_flux', 'CP_Flux_Comp')
    catcp_filtered.rename_column('E_Total_flux', 'CP_Flux_Comp_err')
    catcp_filtered.rename_column('Peak_flux', 'CP_Flux_Comp_Peak')
    catcp_filtered.rename_column('E_Peak_flux', 'CP_Flux_Comp_Peak_err')
    catcp_filtered.rename_column('S_Code', 'CP_S_Code')
    catcp_filtered.rename_column('Isl_rms', 'CP_rms')
    catcp_filtered.rename_column('DC_Maj', 'CP_DC_maj')
    catcp_filtered.rename_column('DC_Min', 'CP_DC_min')
    catcp_filtered.rename_column('DC_PA', 'CP_DC_pa')
    # Generate the source names and the correct RA, DEC, fluxes etc. for the sources by combining components
    source_ids = np.full(len(catcp_filtered), '', dtype='S20')
    obsids = np.full(len(catcp_filtered), self.obsid, dtype='S9')
    fieldname = glob.glob(self.circpolmosaicdir + '/*.fits')[0].split('/')[-1].rstrip('.fits')
    fieldnames = np.full(len(catcp_filtered), fieldname, dtype='S10')
    RA_arr = np.full(len(catcp_filtered), np.nan)
    RAerr_arr = np.full(len(catcp_filtered), np.nan)
    DEC_arr = np.full(len(catcp_filtered), np.nan)
    DECerr_arr = np.full(len(catcp_filtered), np.nan)
    CP_arr = np.full(len(catcp_filtered), np.nan)
    CPerr_arr = np.full(len(catcp_filtered), np.nan)
    CPrms_arr = np.full(len(catcp_filtered), np.nan)
    islands = np.unique(catcp_filtered['Isl_id'])
    for isl in islands:
        islidxs = np.where(isl == catcp_filtered['Isl_id'])[0]
        if len(islidxs) == 1:
            RA_arr[islidxs] = catcp_filtered['CP_RA_Comp'][islidxs]
            RAerr_arr[islidxs] = catcp_filtered['CP_RA_Comp_err'][islidxs]
            DEC_arr[islidxs] = catcp_filtered['CP_DEC_Comp'][islidxs]
            DECerr_arr[islidxs] = catcp_filtered['CP_DEC_Comp_err'][islidxs]
            CP_arr[islidxs] = catcp_filtered['CP_Flux_Comp'][islidxs]
            CPerr_arr[islidxs] = catcp_filtered['CP_Flux_Comp_err'][islidxs]
            CPrms_arr[islidxs] = catcp_filtered['CP_rms'][islidxs]
            # Generate the source name
            src_name = utilv.make_source_id(RA_arr[islidxs][0], DEC_arr[islidxs][0], 'CP')
            source_ids[islidxs] = src_name
        else:
            RA_arr[islidxs] = np.mean(catcp_filtered['CP_RA_Comp'][islidxs])
            RAerr_arr[islidxs] = list(np.full(len(islidxs), np.sqrt(np.sum(np.square(catcp_filtered['CP_RA_Comp_err'][islidxs])))))
            DEC_arr[islidxs] = np.mean(catcp_filtered['CP_DEC_Comp'][islidxs])
            DECerr_arr[islidxs] = list(np.full(len(islidxs), np.sqrt(np.sum(np.square(catcp_filtered['CP_DEC_Comp_err'][islidxs])))))
            CP_arr[islidxs] = np.full(len(islidxs), np.sum(catcp_filtered['CP_Flux_Comp'][islidxs]))
            CPerr_arr[islidxs] = list(np.full(len(islidxs), np.sqrt(np.sum(np.square(catcp_filtered['CP_Flux_Comp_err'][islidxs])))))
            CPrms_arr[islidxs] = np.mean(catcp_filtered['CP_rms'][islidxs])
            # Generate the source name
            src_name = utilv.make_source_id(RA_arr[islidxs][0], DEC_arr[islidxs][0], 'CP')
            source_ids[islidxs] = np.full(len(islidxs), src_name)
    for s in catcp_filtered:
        if (s['CP_DC_maj'] > 0.0) or (s['CP_DC_min'] > 0.0) or (s['CP_DC_pa'] > 0.0) or s['CP_S_Code'] != 'S':
            s['CP_S_Code'] = 'E'
        else:
            s['CP_S_Code'] = 'S'
    catcp_filtered['ID'] = source_ids
    catcp_filtered['OBSID'] = obsids
    catcp_filtered['Field'] = fieldnames
    catcp_filtered['CP_RA'] = RA_arr
    catcp_filtered['CP_RA_err'] = RAerr_arr
    catcp_filtered['CP_DEC'] = DEC_arr
    catcp_filtered['CP_DEC_err'] = DECerr_arr
    catcp_filtered['CP_Flux'] = CP_arr
    catcp_filtered['CP_Flux_err'] = CPerr_arr
    catcp_filtered['CP_rms'] = CPrms_arr
    catcp_filtered.remove_column('Isl_id')
    catcp_filtered.remove_column('CP_DC_maj')
    catcp_filtered.remove_column('CP_DC_min')
    catcp_filtered.remove_column('CP_DC_pa')
    catcp_filtered.write(self.circpolanalysisdir + '/CP_cat.txt', format='ascii', overwrite=True)


def source_finding_i(self):
    """
    Run pybdsf on the Stokes I mosaic
    """
    tp_image = self.circpolanalysisdir + '/TP.fits'
    tp_image_uncorr = self.circpolanalysisdir + '/TP_uncorr.fits'
    bmaj, bmin = utilv.get_beam_mosaic(tp_image)
    rmsbox = ((bmaj*3600.0)/3.0)*30.0
    rmsboxshift = ((bmaj*3600.0)/3.0)*(30.0/8.0)
    brightbox = ((bmaj*3600.0)/3.0)*12.0
    brightboxshift = ((bmaj*3600.0)/3.0)*(12.0/8.0)
    source_list_tp = bdsf.process_image(tp_image, detection_image=tp_image_uncorr, adaptive_rms_box=True, ncores=8, thresh_isl=4.0, thresh_pix=5.0, quiet=True, rms_box=(rmsbox,rmsboxshift), rms_box_bright=(brightbox,brightboxshift), rms_map=True, group_by_isl=False, group_tol=10.0, advanced_opts=True, blank_limit=None, mean_map='zero', spline_rank=1)
    source_list_tp.export_image(outfile=self.circpolanalysisdir + '/TP_rms.fits', clobber=True, img_type='rms')
    source_list_tp.export_image(outfile=self.circpolanalysisdir + '/TP_ch0.fits', clobber=True, img_type='ch0')
    source_list_tp.export_image(outfile=self.circpolanalysisdir + '/TP_mean.fits', clobber=True, img_type='mean')
    rms_hdu = pyfits.open(self.circpolanalysisdir + '/TP_rms.fits')[0]
    ch0_hdu = pyfits.open(self.circpolanalysisdir + '/TP_ch0.fits')[0]
    mean_hdu = pyfits.open(self.circpolanalysisdir + '/TP_mean.fits')[0]
    rms_data = rms_hdu.data
    ch0_data = ch0_hdu.data
    mean_data = mean_hdu.data
    norm_data = (ch0_data - mean_data) / rms_data
    pyfits.writeto(self.circpolanalysisdir + '/TP_norm.fits', data=norm_data, header=mean_hdu.header, overwrite=True)
    source_list_tp.write_catalog(outfile=self.circpolanalysisdir + '/TP_cat.txt', format='ascii', clobber=True)


def gen_catalogue_i(self):
    """
    Save and filter the catalogue, combine source components into single sources, generate IDs
    """
    TP_cat = utilv.load_catalogue(self.circpolanalysisdir + '/TP_cat.txt')
    TP_cat_filtered = TP_cat['Isl_id','RA','E_RA','DEC','E_DEC','Total_flux','E_Total_flux','Peak_flux','E_Peak_flux','S_Code']
    TP_cat_filtered.rename_column('RA', 'TP_RA_Comp')
    TP_cat_filtered.rename_column('E_RA', 'TP_RA_Comp_err')
    TP_cat_filtered.rename_column('DEC', 'TP_DEC_Comp')
    TP_cat_filtered.rename_column('E_DEC', 'TP_DEC_Comp_err')
    TP_cat_filtered.rename_column('Total_flux', 'TP_Flux_Comp')
    TP_cat_filtered.rename_column('E_Total_flux', 'TP_Flux_Comp_err')
    TP_cat_filtered.rename_column('Peak_flux', 'TP_Flux_Comp_Peak')
    TP_cat_filtered.rename_column('E_Peak_flux', 'TP_Flux_Comp_Peak_err')
    # Correct RA, DEC, fluxes etc. for the sources by combining components
    RA_arr = np.full(len(TP_cat_filtered), np.nan)
    RAerr_arr = np.full(len(TP_cat_filtered), np.nan)
    DEC_arr = np.full(len(TP_cat_filtered), np.nan)
    DECerr_arr = np.full(len(TP_cat_filtered), np.nan)
    TP_arr = np.full(len(TP_cat_filtered), np.nan)
    TPerr_arr = np.full(len(TP_cat_filtered), np.nan)
    islands = np.unique(TP_cat_filtered['Isl_id'])
    for isl in islands:
        islidxs = np.where(isl == TP_cat_filtered['Isl_id'])[0]
        if len(islidxs) == 1:
            RA_arr[islidxs] = TP_cat_filtered['TP_RA_Comp'][islidxs]
            RAerr_arr[islidxs] = TP_cat_filtered['TP_RA_Comp_err'][islidxs]
            DEC_arr[islidxs] = TP_cat_filtered['TP_DEC_Comp'][islidxs]
            DECerr_arr[islidxs] = TP_cat_filtered['TP_DEC_Comp_err'][islidxs]
            TP_arr[islidxs] = TP_cat_filtered['TP_Flux_Comp'][islidxs]
            TPerr_arr[islidxs] = TP_cat_filtered['TP_Flux_Comp_err'][islidxs]
        else:
            RA_arr[islidxs] = np.mean(TP_cat_filtered['TP_RA_Comp'][islidxs])
            RAerr_arr[islidxs] = list(np.full(len(islidxs), np.sqrt(np.sum(np.square(TP_cat_filtered['TP_RA_Comp_err'][islidxs])))))
            DEC_arr[islidxs] = np.mean(TP_cat_filtered['TP_DEC_Comp'][islidxs])
            DECerr_arr[islidxs] = list(np.full(len(islidxs), np.sqrt(np.sum(np.square(TP_cat_filtered['TP_DEC_Comp_err'][islidxs])))))
            TP_arr[islidxs] = np.full(len(islidxs), np.sum(TP_cat_filtered['TP_Flux_Comp'][islidxs]))
            TPerr_arr[islidxs] = list(np.full(len(islidxs), np.sqrt(np.sum(np.square(TP_cat_filtered['TP_Flux_Comp_err'][islidxs])))))
    TP_cat_filtered['TP_RA'] = RA_arr
    TP_cat_filtered['TP_RA_err'] = RAerr_arr
    TP_cat_filtered['TP_DEC'] = DEC_arr
    TP_cat_filtered['TP_DEC_err'] = DECerr_arr
    TP_cat_filtered['TP_Flux'] = TP_arr
    TP_cat_filtered['TP_Flux_err'] = TPerr_arr
    # Generate source IDs for total power
    TP_ID_arr = np.full(len(TP_cat_filtered), '', dtype='S20')
    for s, source in enumerate(TP_cat_filtered):
        TP_ID_arr[s] = utilv.make_source_id(TP_cat_filtered['TP_RA'][s], TP_cat_filtered['TP_DEC'][s], 'TP')
        TP_cat_filtered['TP_ID'] = TP_ID_arr
    TP_cat_filtered.remove_column('Isl_id')
    new_order = ['TP_ID','TP_RA','TP_RA_err','TP_DEC','TP_DEC_err','TP_Flux','TP_Flux_err','S_Code','TP_RA_Comp','TP_RA_Comp_err','TP_DEC_Comp','TP_DEC_Comp_err','TP_Flux_Comp','TP_Flux_Comp_err','TP_Flux_Comp_Peak','TP_Flux_Comp_Peak_err']
    tp_new_order = TP_cat_filtered[new_order]
    tp_new_order.write(self.circpolanalysisdir + '/TP_cat.txt', format='ascii', overwrite=True)


def remove_pointing_centres(self):
    print('Removing sources at the pointing centres from circular polarisation and total power catalogue')
    # Write a file with the central coordinates of each pointing used
    coord_arr = np.full((40, 3), np.nan)
    coord_arr[:, 0] = np.arange(0, 40, 1)
    for b in range(40):
        if os.path.isfile(os.path.join(self.basedir, self.obsid, str(b).zfill(2), 'polarisation') + '/image_mf_V.fits'):
            vfile = pyfits.open(os.path.join(self.basedir, self.obsid, str(b).zfill(2), 'polarisation') + '/image_mf_V.fits')[0]
            vfile_hdr = vfile.header
            if float(vfile_hdr['CRVAL1']) < 0.0:
                coord_arr[b, 1] = 360.0 + float(vfile_hdr['CRVAL1'])
            else:
                coord_arr[b, 1] = vfile_hdr['CRVAL1']
            coord_arr[b, 2] = vfile_hdr['CRVAL2']
    np.savetxt(self.circpolanalysisdir + '/pointings.txt', coord_arr, fmt=['%2s', '%1.13e', '%1.13e'], delimiter='\t')
    ra, dec = utilv.load_pointing_centres(self.circpolanalysisdir + '/pointings.txt')
    cat_i = Table.read(self.circpolanalysisdir + '/TP_cat.txt', format='ascii')
    cat_v = Table.read(self.circpolanalysisdir + '/CP_cat.txt', format='ascii')
    cat_clean_i = match_and_remove(self, cat_i, ra, dec, 'TP_', '_Comp')
    cat_clean_v = match_and_remove(self, cat_v, ra, dec, 'CP_', '')
    cat_clean_i.write(self.circpolanalysisdir + '/TP_cat.txt', format='ascii', overwrite=True)
    cat_clean_i.write(self.circpolanalysisdir + '/TP_cat.fits', format='fits', overwrite=True)
    cat_clean_v.write(self.circpolanalysisdir + '/CP_cat.txt', format='ascii', overwrite=True)
    cat_clean_v.write(self.circpolanalysisdir + '/CP_cat.fits', format='fits', overwrite=True)


def match_and_remove(self, cat, ra, dec, prefix, suffix):
    # Initialise a table for removing sources
    remove_list = []
    # Get the major beam size of the mosaic
    PI_hdu = pyfits.open(self.circpolanalysisdir + '/CP.fits')
    PI_hdr = PI_hdu[0].header
    bmaj = PI_hdr['BMAJ']
    cat_ra = cat[prefix + 'RA' + suffix]
    cat_dec = cat[prefix + 'DEC' + suffix]
    for s, source in enumerate(cat_ra):
        dist_arr = np.sqrt(np.square(ra - cat_ra[s]) + np.square(dec - cat_dec[s]))
        if np.any(dist_arr<(bmaj/2.0)):
            remove_list.append(s)
    cat.remove_rows([remove_list])
    return cat
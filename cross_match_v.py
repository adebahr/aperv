import numpy as np

from psrqpy import QueryATNF
from astropy.table import Table
from astropy import units as u
import astropy.coordinates as coord
from astroquery.irsa import Irsa
from astroquery.sdss import SDSS
from astroquery.gaia import Gaia

import utilv


def cross_match_v(self):
    """
    Cross-match the circular polarisation catalogue with the TP-catalogue, AllWISE, SDSS, NASA exoplanet database and the ATNF pulsar catalogue
    """
    CP_cat, TP_cat = load_catalogues(self)
    CP_cat = cross_match_TP(self, CP_cat, TP_cat)
    CP_cat = cross_match_vlotss(self, CP_cat)
    CP_cat = cross_match_allwise(self, CP_cat)
    CP_cat = cross_match_sdss(self, CP_cat)
    CP_cat = cross_match_gaia(self, CP_cat)
    CP_cat = cross_match_exoplanet(self, CP_cat)
    CP_cat = cross_match_PSR(self, CP_cat)
    CP_cat = cross_match_UC(self, CP_cat)
    CP_cat = cross_match_TDE(self, CP_cat)
    CP_cat = cross_match_flares(self, CP_cat)
    write_CP_cat(self, CP_cat)


def load_catalogues(self):
    CP_cat = Table.read(self.circpolanalysisdir + '/CP_cat.txt', format='ascii')
    TP_cat = Table.read(self.circpolanalysisdir + '/TP_cat.txt', format='ascii')
    return CP_cat, TP_cat


def cross_match_TP(self, CP_cat, TP_cat):
    """
    Cross-match the circular polarised sources with their TP counterparts
    """
    # Generate the columns for the TP information
    TP_crossmatch_arr = np.full(len(CP_cat), False, dtype=bool)
    TP_id_arr = np.full(len(CP_cat), '--', dtype='S20')
    TP_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    TP_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    TP_ra_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    TP_dec_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    TP_ra_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    TP_dec_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    FP_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    FP_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    for s, source in enumerate(CP_cat):
        ra_source = source['CP_RA']
        dec_source = source['CP_DEC']
        dist_arr = np.sqrt(np.square(TP_cat['TP_RA'] - ra_source) + np.square(TP_cat['TP_DEC'] - dec_source))
        min_idx = np.argmin(np.abs(dist_arr))
        # Calculate maximum distance for a match
        bmaj, bmin = utilv.get_beam(self)
        dist = (np.max([bmaj, bmin]) * 1.0)
        if dist_arr[min_idx] <= dist:
            TP_crossmatch_arr[s] = True
            TP_id_arr[s] = TP_cat[min_idx]['TP_ID']
            TP_arr[s] = TP_cat[min_idx]['TP_Flux']
            TP_err_arr[s] = TP_cat[min_idx]['TP_Flux_err']
            TP_ra_arr[s] = TP_cat[min_idx]['TP_RA']
            TP_ra_err_arr[s] = TP_cat[min_idx]['TP_RA_err']
            TP_dec_arr[s] = TP_cat[min_idx]['TP_DEC']
            TP_dec_err_arr[s] = TP_cat[min_idx]['TP_DEC_err']
            FP_arr[s] = source['CP_Flux'] / TP_cat[min_idx]['TP_Flux']
            FP_err_arr[s] = np.sqrt((1.0 / TP_cat[min_idx]['TP_Flux']) ** 2.0 * source['CP_Flux_err'] ** 2.0 + (source['CP_Flux'] / (TP_cat[min_idx]['TP_Flux'] ** 2.0)) * TP_cat[min_idx]['TP_Flux_err'] ** 2.0)
        else:
            pass
    CP_cat['TP_cross'] = TP_crossmatch_arr
    CP_cat['TP_ID'] = TP_id_arr
    CP_cat['TP_Flux'] = TP_arr
    CP_cat['TP_Flux_err'] = TP_err_arr
    CP_cat['TP_RA'] = TP_ra_arr
    CP_cat['TP_RA_err'] = TP_ra_err_arr
    CP_cat['TP_DEC'] = TP_dec_arr
    CP_cat['TP_DEC_err'] = TP_dec_err_arr
    CP_cat['FP'] = FP_arr
    CP_cat['FP_err'] = FP_err_arr
    # Combine different polarised sources, which have the same total power counterpart
    for tp_source_ra in np.unique(TP_ra_arr):
        nsources = np.where(TP_ra_arr == tp_source_ra)
        if len(nsources[0]) < 2:
            pass
        else:
            if len(np.unique(CP_cat['ID'][nsources])) > 1:
                for pis in nsources:
                    CP_cat['CP_RA'][pis] = np.mean(CP_cat['CP_RA'][nsources])
                    CP_cat['CP_RA_err'][pis] = np.sqrt(np.sum(np.square(CP_cat['CP_RA_err'][pis])))
                    CP_cat['CP_DEC'][pis] = np.mean(CP_cat['CP_DEC'][nsources])
                    CP_cat['CP_DEC_err'][pis] = np.sqrt(np.sum(np.square(CP_cat['CP_DEC_err'][pis])))
                    CP_cat['ID'][pis] = utilv.make_source_id(CP_cat['CP_RA'][pis][0], CP_cat['CP_DEC'][pis][0], 'CP')
                    CP_cat['CP_Flux'][pis] = np.sum(CP_cat['CP_Flux'][nsources])
                    CP_cat['CP_Flux_err'][pis] = np.sqrt(np.sum(np.square(CP_cat['CP_Flux_err'][pis])))
                    CP_cat['CP_S_Code'][pis] = 'E'
                    CP_cat['FP'][pis] = CP_cat['CP_Flux'][pis] / CP_cat['TP_Flux'][pis]
                    CP_cat['FP_err'][pis] = np.sqrt((1.0 / CP_cat['TP_Flux'][nsources]) ** 2.0 * CP_cat['CP_Flux_err'][nsources] ** 2.0 + (CP_cat['CP_Flux'][nsources] / (CP_cat['TP_Flux'][nsources] ** 2.0)) * CP_cat['TP_Flux_err'][nsources] ** 2.0)
            else:
                pass
    return CP_cat


def cross_match_vlotss(self, CP_cat):
    """
    Cross-match with the vlotss catalogue
    """
    vlotss_table = Table.read(self.cataloguedir + 'vlotss_apertif.fits', format='fits')
    vlotss_crossmatch_arr = np.full(len(CP_cat), False, dtype=bool)
    vlotss_name_arr = np.full(len(CP_cat), '--', dtype='S23')
    vlotss_ra_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    vlotss_dec_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    vlotss_TP_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    vlotss_CP_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    vlotss_FP_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    for s, source in enumerate(CP_cat):
        ra_source = source['CP_RA']
        dec_source = source['CP_DEC']
        dist_arr = np.sqrt(np.square(vlotss_table['RAdeg'] - ra_source) + np.square(vlotss_table['DEdeg'] - dec_source))
        min_idx = np.argmin(np.abs(dist_arr))
        # Calculate maximum distance for a match
        bmaj, bmin = utilv.get_beam(self)
        dist = (np.max([bmaj, bmin]) * 1.0)
        if dist_arr[min_idx] <= dist:
            vlotss_crossmatch_arr[s] = True
            vlotss_name_arr[s] = str(vlotss_table[min_idx]['Name'])
            vlotss_ra_arr[s] = vlotss_table[min_idx]['RAdeg']
            vlotss_dec_arr[s] = vlotss_table[min_idx]['DEdeg']
            vlotss_TP_arr[s] = vlotss_table[min_idx]['FtotI']
            vlotss_CP_arr[s] = vlotss_table[min_idx]['FtotV']
            vlotss_FP_arr[s] = vlotss_table[min_idx]['vI']
        else:
            pass
    CP_cat['VLOTSS_cross'] = vlotss_crossmatch_arr
    CP_cat['VLOTSS_ID'] = vlotss_name_arr
    CP_cat['VLOTSS_RA'] = vlotss_ra_arr
    CP_cat['VLOTSS_DEC'] = vlotss_dec_arr
    CP_cat['VLOTSS_TP'] = vlotss_TP_arr
    CP_cat['VLOTSS_CP'] = vlotss_CP_arr
    CP_cat['VLOTSS_FP'] = vlotss_FP_arr
    return CP_cat


def cross_match_allwise(self, CP_cat):
    """
    Query the AllWISE database for cross-matches
    """
    # Calculate maximum distance for a match
    bmaj, bmin = utilv.get_beam(self)
    dist = (np.max([bmaj, bmin]) / 2.0) * 3600.0
    # Generate the list for the additional information
    crossmatch_arr = np.full(len(CP_cat), False, dtype=bool)
    dist_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    designation_arr = np.full(len(CP_cat), '--', dtype='S20')
    wise_ra_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    wise_ra_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    wise_dec_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    wise_dec_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    w1mpro_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    w1mpro_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    w1snr_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    w2mpro_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    w2mpro_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    w2snr_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    w3mpro_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    w3mpro_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    w3snr_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    w4mpro_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    w4mpro_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    w4snr_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    # Cross-match WISE sources with Apertif source catalogue
    for s, source in enumerate(CP_cat):
        # First match with the total power source if available otherwise with the PI source
        if np.isnan(source['TP_RA']):
            match = []
        else:
            try:
                match = Irsa.query_region(coord.SkyCoord(source['TP_RA'], source['TP_DEC'], unit=(u.deg, u.deg), frame='icrs'), catalog='allwise_p3as_psd', radius=dist * u.arcsec)
            except:
                match = []
        if len(match) == 0:
            pass
        else:
            crossmatch_arr[s] = True
            dist_arr[s] = match['dist'][0]
            designation_arr[s] = str(match['designation'][0])
            wise_ra_arr[s] = match['ra'][0]
            wise_ra_err_arr[s] = match['sigra'][0]
            wise_dec_arr[s] = match['dec'][0]
            wise_dec_err_arr[s] = match['sigdec'][0]
            w1mpro_arr[s] = match['w1mpro'][0]
            w1mpro_err_arr[s] = match['w1sigmpro'][0]
            w1snr_arr[s] = match['w1snr'][0]
            w2mpro_arr[s] = match['w2mpro'][0]
            w2mpro_err_arr[s] = match['w2sigmpro'][0]
            w2snr_arr[s] = match['w2snr'][0]
            w3mpro_arr[s] = match['w3mpro'][0]
            w3mpro_err_arr[s] = match['w3sigmpro'][0]
            w3snr_arr[s] = match['w3snr'][0]
            w4mpro_arr[s] = match['w4mpro'][0]
            w4mpro_err_arr[s] = match['w4sigmpro'][0]
            w4snr_arr[s] = match['w4snr'][0]
    CP_cat['WISE_cross'] = crossmatch_arr
    CP_cat['WISE_ID'] = designation_arr
    CP_cat['WISE_RA'] = wise_ra_arr
    CP_cat['WISE_RA_err'] = wise_ra_err_arr
    CP_cat['WISE_DEC'] = wise_dec_arr
    CP_cat['WISE_DEC_err'] = wise_dec_err_arr
    CP_cat['WISE_Flux_3.4'] = w1mpro_arr
    CP_cat['WISE_Flux_3.4_err'] = w1mpro_err_arr
    CP_cat['WISE_SNR_3.4'] = w1snr_arr
    CP_cat['WISE_Flux_4.6'] = w2mpro_arr
    CP_cat['WISE_Flux_4.6_err'] = w2mpro_err_arr
    CP_cat['WISE_SNR_4.6'] = w2snr_arr
    CP_cat['WISE_Flux_12'] = w3mpro_arr
    CP_cat['WISE_Flux_12_err'] = w3mpro_err_arr
    CP_cat['WISE_SNR_12'] = w3snr_arr
    CP_cat['WISE_Flux_22'] = w4mpro_arr
    CP_cat['WISE_Flux_22_err'] = w4mpro_err_arr
    CP_cat['WISE_SNR_22'] = w4snr_arr
    return CP_cat


def cross_match_sdss(self, CP_cat):
    """
    Query the SDSS database for cross-matches
    """
    # Calculate maximum distance for a match
    bmaj, bmin = utilv.get_beam(self)
    dist = (np.max([bmaj, bmin])/2.0)*3600.0
    # Generate the list for the additional information
    crossmatch_arr = np.full(len(CP_cat), False, dtype=bool)
    objid_arr = np.full(len(CP_cat), '--', dtype='S18')
    type_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_ra_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_dec_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_u_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_uerr_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_g_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_gerr_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_r_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_rerr_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_i_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_ierr_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_z_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_zerr_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_rs_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    sdss_rserr_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    # Cross-match SDSS sources with WISE coordinates of Apertif polarised source catalogue
    for s, source in enumerate(CP_cat):
        if np.isnan(source['WISE_RA']):
            pass
        else:
            match = SDSS.query_region(coord.SkyCoord(source['WISE_RA'], source['WISE_DEC'], unit=(u.deg, u.deg), frame='icrs'), photoobj_fields=['objid','ra','dec','u','err_u','g','err_g','r','err_r','i','err_i','z','err_z','type'], spectro=False, radius=3.2 * u.arcsec, data_release=16)
            if match is not None and len(match) != 0:
                dist = np.sqrt(np.square(match['ra']-source['WISE_RA']) + np.square(match['dec']-source['WISE_DEC']))
                minidx = np.argmin(dist)
                sdss_src = match[minidx]
                crossmatch_arr[s] = True
                objid_arr[s] = str(sdss_src['objid'])
                sdss_ra_arr[s] = sdss_src['ra']
                sdss_dec_arr[s] = sdss_src['dec']
                type_arr[s] = sdss_src['type']
                if sdss_src['u'] < 0.0:
                    sdss_u_arr[s] = np.nan
                else:
                    sdss_u_arr[s] = sdss_src['u']
                if sdss_src['err_u'] < 0.0:
                    sdss_uerr_arr[s] = np.nan
                else:
                    sdss_uerr_arr[s] = sdss_src['err_u']
                if sdss_src['g'] < 0.0:
                    sdss_g_arr[s] = np.nan
                else:
                    sdss_g_arr[s] = sdss_src['g']
                if sdss_src['err_g'] < 0.0:
                    sdss_gerr_arr[s] = np.nan
                else:
                    sdss_gerr_arr[s] = sdss_src['err_g']
                if sdss_src['r'] < 0.0:
                    sdss_r_arr[s] = np.nan
                else:
                    sdss_r_arr[s] = sdss_src['r']
                if sdss_src['err_r'] < 0.0:
                    sdss_rerr_arr[s] = np.nan
                else:
                    sdss_rerr_arr[s] = sdss_src['err_r']
                if sdss_src['i'] < 0.0:
                    sdss_i_arr[s] = np.nan
                else:
                    sdss_i_arr[s] = sdss_src['i']
                if sdss_src['err_i'] < 0.0:
                    sdss_ierr_arr[s] = np.nan
                else:
                    sdss_ierr_arr[s] = sdss_src['err_i']
                if sdss_src['z'] < 0.0:
                    sdss_z_arr[s] = np.nan
                else:
                    sdss_z_arr[s] = sdss_src['z']
                if sdss_src['err_z'] < 0.0:
                    sdss_zerr_arr[s] = np.nan
                else:
                    sdss_zerr_arr[s] = sdss_src['err_z']
                match_spec = SDSS.query_region(coord.SkyCoord(source['WISE_RA'], source['WISE_DEC'], unit=(u.deg, u.deg), frame='icrs'), photoobj_fields=['objid'], specobj_fields=['ra','dec','z','zerr'], spectro=True, radius=3.2 * u.arcsec, data_release=16)
                if match_spec is not None:
                    match_spec_crossid = match_spec[np.where(match_spec['objid'] == sdss_src['objid'])]
                    if len(match_spec_crossid) == 0:
                        sdss_rs_arr[s] = np.nan
                        sdss_rserr_arr[s] = np.nan
                    elif len(match_spec_crossid) == 1:
                        sdss_rs_arr[s] = match_spec_crossid['z']
                        sdss_rserr_arr[s] = match_spec_crossid['zerr']
                    else:
                        sdss_rs_arr[s] = match_spec_crossid['z'][0]
                        sdss_rserr_arr[s] = match_spec_crossid['zerr'][0]
                else:
                    sdss_rs_arr[s] = np.nan
                    sdss_rserr_arr[s] = np.nan
            else:
                objid_arr[s] = np.nan
                type_arr[s] = np.nan
                sdss_ra_arr[s] = np.nan
                sdss_dec_arr[s] = np.nan
                sdss_u_arr[s] = np.nan
                sdss_uerr_arr[s] = np.nan
                sdss_g_arr[s] = np.nan
                sdss_gerr_arr[s] = np.nan
                sdss_r_arr[s] = np.nan
                sdss_rerr_arr[s] = np.nan
                sdss_i_arr[s] = np.nan
                sdss_ierr_arr[s] = np.nan
                sdss_z_arr[s] = np.nan
                sdss_zerr_arr[s] = np.nan
                sdss_rs_arr[s] = np.nan
                sdss_rserr_arr[s] = np.nan
                sdss_rs_arr[s] = np.nan
                sdss_rserr_arr[s] = np.nan
    CP_cat['SDSS_cross'] = crossmatch_arr
    CP_cat['SDSS_ID'] = objid_arr
    CP_cat['SDSS_RA'] = sdss_ra_arr
    CP_cat['SDSS_DEC'] = sdss_dec_arr
    CP_cat['SDSS_Type'] = type_arr
    CP_cat['SDSS_Flux_U'] = sdss_u_arr
    CP_cat['SDSS_Flux_U_err'] = sdss_uerr_arr
    CP_cat['SDSS_Flux_G'] = sdss_g_arr
    CP_cat['SDSS_Flux_G_err'] = sdss_gerr_arr
    CP_cat['SDSS_Flux_R'] = sdss_r_arr
    CP_cat['SDSS_Flux_R_err'] = sdss_rerr_arr
    CP_cat['SDSS_Flux_I'] = sdss_i_arr
    CP_cat['SDSS_Flux_I_err'] = sdss_ierr_arr
    CP_cat['SDSS_Flux_Z'] = sdss_z_arr
    CP_cat['SDSS_Flux_Z_err'] = sdss_zerr_arr
    CP_cat['SDSS_z'] = sdss_rs_arr
    CP_cat['SDSS_z_err'] = sdss_rserr_arr
    return CP_cat


def cross_match_gaia(self, CP_cat):
    """
    Query the GAIA database for cross-matches
    """
    # Calculate maximum distance for a match
    bmaj, bmin = utilv.get_beam(self)
    dist = (np.max([bmaj, bmin]) / 2.0) * 3600.0
    # Generate the list for the additional information
    Gaia_crossmatch_arr = np.full(len(CP_cat), False, dtype=bool)
    Gaia_dist_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Gaia_id_arr = np.full(len(CP_cat), '--', dtype='S19')
    Gaia_ra_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Gaia_ra_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Gaia_dec_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Gaia_dec_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Gaia_parallax_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Gaia_parallax_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    # Cross-match Gaia sources with Apertif CP source catalogue
    for s, source in enumerate(CP_cat):
        sc = coord.SkyCoord(source['CP_RA'], source['CP_DEC'], unit=(u.deg, u.deg), frame='icrs')
        j = Gaia.cone_search(sc, radius=u.Quantity(dist, u.arcsec))
        match = j.get_results()
        if len(match) == 0:
            pass
        else:
            Gaia_crossmatch_arr[s] = True
            Gaia_dist_arr[s] = match['dist'][0]
            Gaia_id_arr[s] = str(match['SOURCE_ID'][0])
            Gaia_ra_arr[s] = match['ra'][0]
            Gaia_ra_err_arr[s] = match['ra_error'][0]
            Gaia_dec_arr[s] = match['dec'][0]
            Gaia_dec_err_arr[s] = match['dec_error'][0]
            Gaia_parallax_arr[s] = match['parallax'][0]
            Gaia_parallax_err_arr[s] = match['parallax_error'][0]
    CP_cat['Gaia_cross'] = Gaia_crossmatch_arr
    CP_cat['Gaia_ID'] = Gaia_id_arr
    CP_cat['Gaia_dist'] = Gaia_dist_arr
    CP_cat['Gaia_RA'] = Gaia_ra_arr
    CP_cat['Gaia_RA_err'] = Gaia_ra_err_arr
    CP_cat['Gaia_DEC'] = Gaia_dec_arr
    CP_cat['Gaia_DEC_err'] = Gaia_dec_err_arr
    CP_cat['Gaia_parallax'] = Gaia_parallax_arr
    CP_cat['Gaia_parallax_err'] = Gaia_parallax_err_arr
    return CP_cat


def cross_match_exoplanet(self, CP_cat):
    """
    Query the Open Exoplanet Database for cross-matches
    """
#    nexa = pyasl.NasaExoplanetArchive()
#    alldata = nexa.getAllData()
    exo_table = Table.read(self.cataloguedir + 'Exoplanet.votable')
    Exo_crossmatch_arr = np.full(len(CP_cat), False, dtype=bool)
    Exo_name_arr = np.full(len(CP_cat), '--', dtype='S20')
    Exo_ra_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Exo_dec_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    for s, source in enumerate(CP_cat):
        ra_source = source['CP_RA']
        dec_source = source['CP_DEC']
        dist_arr = np.sqrt(np.square(exo_table['ra'] - ra_source) + np.square(exo_table['dec'] - dec_source))
        min_idx = np.argmin(np.abs(dist_arr))
        # Calculate maximum distance for a match
        bmaj, bmin = utilv.get_beam(self)
        dist = (np.max([bmaj, bmin]) * 1.0)
        if dist_arr[min_idx] <= dist:
            Exo_crossmatch_arr[s] = True
            Exo_name_arr[s] = str(exo_table[min_idx]['hostname'])
            Exo_ra_arr[s] = exo_table[min_idx]['ra']
            Exo_dec_arr[s] = exo_table[min_idx]['dec']
        else:
            pass
    CP_cat['Exo_cross'] = Exo_crossmatch_arr
    CP_cat['Exo_name'] = Exo_name_arr
    CP_cat['Exo_RA'] = Exo_ra_arr
    CP_cat['Exo_DEC'] = Exo_dec_arr
    return CP_cat


def cross_match_PSR(self, CP_cat):
    """
    Query the ATNF pulsar database for cross-matches
    """
    query = QueryATNF()
    pulsar_table = Table.read(self.cataloguedir + 'ATNF_pulsars.fits', format='fits')
    Pulsar_crossmatch_arr = np.full(len(CP_cat), False, dtype=bool)
    Pulsar_name_arr = np.full(len(CP_cat), '--', dtype='S12')
    Pulsar_ra_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_ra_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_dec_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_dec_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_P0_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_P0_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_P1_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_P1_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_DM_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_DM_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_RM_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_RM_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_W50_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_W10_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S40_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S40_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S50_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S50_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S60_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S60_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S80_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S80_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S100_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S100_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S150_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S150_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S200_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S200_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S300_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S300_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S350_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S350_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S400_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S400_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S600_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S600_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S700_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S700_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S800_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S800_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S900_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S900_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S1400_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S1400_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S1600_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S1600_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S2000_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S2000_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S3000_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S3000_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S4000_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S4000_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S5000_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S5000_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S6000_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S6000_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S8000_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S8000_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S9000_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_S9000_err_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_dist_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_age_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    Pulsar_bsurf_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    for s, source in enumerate(CP_cat):
        ra_source = source['CP_RA']
        dec_source = source['CP_DEC']
        dist_arr = np.sqrt(np.square(pulsar_table['RAJD'] - ra_source) + np.square(pulsar_table['DECJD'] - dec_source))
        min_idx = np.argmin(np.abs(dist_arr))
        # Calculate maximum distance for a match
        bmaj, bmin = utilv.get_beam(self)
        dist = (np.max([bmaj, bmin]) * 1.0)
        if dist_arr[min_idx] <= dist:
            Pulsar_crossmatch_arr[s] = True
            spec_pulsar_table = query[pulsar_table[min_idx]['PSRJ']]
            Pulsar_name_arr[s] = str(spec_pulsar_table['PSRJ'][0])
            Pulsar_ra_arr[s] = spec_pulsar_table['RAJD'][0]
            Pulsar_ra_err_arr[s] = spec_pulsar_table['RAJD_ERR'][0]
            Pulsar_dec_arr[s] = spec_pulsar_table['DECJD'][0]
            Pulsar_dec_err_arr[s] = spec_pulsar_table['DECJD_ERR'][0]
            Pulsar_P0_arr[s] = spec_pulsar_table['P0'][0]
            Pulsar_P0_err_arr[s] = spec_pulsar_table['P0_ERR'][0]
            Pulsar_P1_arr[s] = spec_pulsar_table['P1'][0]
            Pulsar_P1_err_arr[s] = spec_pulsar_table['P1_ERR'][0]
            Pulsar_DM_arr[s] = spec_pulsar_table['DM'][0]
            Pulsar_DM_err_arr[s] = spec_pulsar_table['DM_ERR'][0]
            Pulsar_RM_arr[s] = spec_pulsar_table['RM'][0]
            Pulsar_RM_err_arr[s] = spec_pulsar_table['RM_ERR'][0]
            Pulsar_W50_arr[s] = spec_pulsar_table['W50'][0]
            Pulsar_W10_arr[s] = spec_pulsar_table['W10'][0]
            Pulsar_S40_arr[s] = spec_pulsar_table['S40'][0]
            Pulsar_S40_err_arr[s] = spec_pulsar_table['S40_ERR'][0]
            Pulsar_S50_arr[s] = spec_pulsar_table['S50'][0]
            Pulsar_S50_err_arr[s] = spec_pulsar_table['S50_ERR'][0]
            Pulsar_S60_arr[s] = spec_pulsar_table['S60'][0]
            Pulsar_S60_err_arr[s] = spec_pulsar_table['S60_ERR'][0]
            Pulsar_S80_arr[s] = spec_pulsar_table['S80'][0]
            Pulsar_S80_err_arr[s] = spec_pulsar_table['S80_ERR'][0]
            Pulsar_S100_arr[s] = spec_pulsar_table['S100'][0]
            Pulsar_S100_err_arr[s] = spec_pulsar_table['S100_ERR'][0]
            Pulsar_S150_arr[s] = spec_pulsar_table['S150'][0]
            Pulsar_S150_err_arr[s] = spec_pulsar_table['S150_ERR'][0]
            Pulsar_S200_arr[s] = spec_pulsar_table['S200'][0]
            Pulsar_S200_err_arr[s] = spec_pulsar_table['S200_ERR'][0]
            Pulsar_S300_arr[s] = spec_pulsar_table['S300'][0]
            Pulsar_S300_err_arr[s] = spec_pulsar_table['S300_ERR'][0]
            Pulsar_S350_arr[s] = spec_pulsar_table['S350'][0]
            Pulsar_S350_err_arr[s] = spec_pulsar_table['S350_ERR'][0]
            Pulsar_S400_arr[s] = spec_pulsar_table['S400'][0]
            Pulsar_S400_err_arr[s] = spec_pulsar_table['S400_ERR'][0]
            Pulsar_S600_arr[s] = spec_pulsar_table['S600'][0]
            Pulsar_S600_err_arr[s] = spec_pulsar_table['S600_ERR'][0]
            Pulsar_S700_arr[s] = spec_pulsar_table['S700'][0]
            Pulsar_S700_err_arr[s] = spec_pulsar_table['S700_ERR'][0]
            Pulsar_S800_arr[s] = spec_pulsar_table['S800'][0]
            Pulsar_S800_err_arr[s] = spec_pulsar_table['S800_ERR'][0]
            Pulsar_S900_arr[s] = spec_pulsar_table['S900'][0]
            Pulsar_S900_err_arr[s] = spec_pulsar_table['S900_ERR'][0]
            Pulsar_S1400_arr[s] = spec_pulsar_table['S1400'][0]
            Pulsar_S1400_err_arr[s] = spec_pulsar_table['S1400_ERR'][0]
            Pulsar_S1600_arr[s] = spec_pulsar_table['S1600'][0]
            Pulsar_S1600_err_arr[s] = spec_pulsar_table['S1600_ERR'][0]
            Pulsar_S2000_arr[s] = spec_pulsar_table['S2000'][0]
            Pulsar_S2000_err_arr[s] = spec_pulsar_table['S2000_ERR'][0]
            Pulsar_S3000_arr[s] = spec_pulsar_table['S3000'][0]
            Pulsar_S3000_err_arr[s] = spec_pulsar_table['S3000_ERR'][0]
            Pulsar_S4000_arr[s] = spec_pulsar_table['S4000'][0]
            Pulsar_S4000_err_arr[s] = spec_pulsar_table['S4000_ERR'][0]
            Pulsar_S5000_arr[s] = spec_pulsar_table['S5000'][0]
            Pulsar_S5000_err_arr[s] = spec_pulsar_table['S5000_ERR'][0]
            Pulsar_S6000_arr[s] = spec_pulsar_table['S6000'][0]
            Pulsar_S6000_err_arr[s] = spec_pulsar_table['S6000_ERR'][0]
            Pulsar_S8000_arr[s] = spec_pulsar_table['S8000'][0]
            Pulsar_S8000_err_arr[s] = spec_pulsar_table['S8000_ERR'][0]
            Pulsar_S9000_arr[s] = spec_pulsar_table['S9000'][0]
            Pulsar_S9000_err_arr[s] = spec_pulsar_table['S9000_ERR'][0]
            Pulsar_dist_arr[s] = spec_pulsar_table['DIST'][0]
            Pulsar_age_arr[s] = spec_pulsar_table['AGE'][0]
            Pulsar_bsurf_arr[s] = spec_pulsar_table['BSURF'][0]
        else:
            pass
    CP_cat['PSR_cross'] = Pulsar_crossmatch_arr
    CP_cat['PSR_ID'] = Pulsar_name_arr
    CP_cat['PSR_RA'] = Pulsar_ra_arr
    CP_cat['PSR_RA_err'] = Pulsar_ra_err_arr
    CP_cat['PSR_DEC'] = Pulsar_dec_arr
    CP_cat['PSR_DEC_err'] = Pulsar_dec_err_arr
    CP_cat['PSR_P0'] = Pulsar_P0_arr
    CP_cat['PSR_P0_err'] = Pulsar_P0_err_arr
    CP_cat['PSR_P1'] = Pulsar_P1_arr
    CP_cat['PSR_P1_err'] = Pulsar_P1_err_arr
    CP_cat['PSR_DM'] = Pulsar_DM_arr
    CP_cat['PSR_DM_err'] = Pulsar_DM_err_arr
    CP_cat['PSR_RM'] = Pulsar_RM_arr
    CP_cat['PSR_RM_err'] = Pulsar_RM_err_arr
    CP_cat['PSR_W50'] = Pulsar_W50_arr
    CP_cat['PSR_W10'] = Pulsar_W10_arr
    CP_cat['PSR_S40'] = Pulsar_S40_arr
    CP_cat['PSR_S40_err'] = Pulsar_S40_err_arr
    CP_cat['PSR_S50'] = Pulsar_S50_arr
    CP_cat['PSR_S50_err'] = Pulsar_S50_err_arr
    CP_cat['PSR_S60'] = Pulsar_S60_arr
    CP_cat['PSR_S60_err'] = Pulsar_S60_err_arr
    CP_cat['PSR_S80'] = Pulsar_S80_arr
    CP_cat['PSR_S80_err'] = Pulsar_S80_err_arr
    CP_cat['PSR_S100'] = Pulsar_S100_arr
    CP_cat['PSR_S100_err'] = Pulsar_S100_err_arr
    CP_cat['PSR_S150'] = Pulsar_S150_arr
    CP_cat['PSR_S150_err'] = Pulsar_S150_err_arr
    CP_cat['PSR_S200'] = Pulsar_S200_arr
    CP_cat['PSR_S200_err'] = Pulsar_S200_err_arr
    CP_cat['PSR_S300'] = Pulsar_S300_arr
    CP_cat['PSR_S300_err'] = Pulsar_S300_err_arr
    CP_cat['PSR_S350'] = Pulsar_S350_arr
    CP_cat['PSR_S350_err'] = Pulsar_S350_err_arr
    CP_cat['PSR_S400'] = Pulsar_S400_arr
    CP_cat['PSR_S400_err'] = Pulsar_S400_err_arr
    CP_cat['PSR_S600'] = Pulsar_S600_arr
    CP_cat['PSR_S600_err'] = Pulsar_S600_err_arr
    CP_cat['PSR_S700'] = Pulsar_S700_arr
    CP_cat['PSR_S700_err'] = Pulsar_S700_err_arr
    CP_cat['PSR_S800'] = Pulsar_S800_arr
    CP_cat['PSR_S800_err'] = Pulsar_S800_err_arr
    CP_cat['PSR_S900'] = Pulsar_S900_arr
    CP_cat['PSR_S900_err'] = Pulsar_S900_err_arr
    CP_cat['PSR_S1400'] = Pulsar_S1400_arr
    CP_cat['PSR_S1400_err'] = Pulsar_S1400_err_arr
    CP_cat['PSR_S1600'] = Pulsar_S1600_arr
    CP_cat['PSR_S1600_err'] = Pulsar_S1600_err_arr
    CP_cat['PSR_S2000'] = Pulsar_S2000_arr
    CP_cat['PSR_S2000_err'] = Pulsar_S2000_err_arr
    CP_cat['PSR_S3000'] = Pulsar_S3000_arr
    CP_cat['PSR_S3000_err'] = Pulsar_S3000_err_arr
    CP_cat['PSR_S4000'] = Pulsar_S4000_arr
    CP_cat['PSR_S4000_err'] = Pulsar_S4000_err_arr
    CP_cat['PSR_S5000'] = Pulsar_S5000_arr
    CP_cat['PSR_S5000_err'] = Pulsar_S5000_err_arr
    CP_cat['PSR_S6000'] = Pulsar_S6000_arr
    CP_cat['PSR_S6000_err'] = Pulsar_S6000_err_arr
    CP_cat['PSR_S8000'] = Pulsar_S8000_arr
    CP_cat['PSR_S8000_err'] = Pulsar_S8000_err_arr
    CP_cat['PSR_S9000'] = Pulsar_S9000_arr
    CP_cat['PSR_S9000_err'] = Pulsar_S9000_err_arr
    CP_cat['PSR_DIST'] = Pulsar_dist_arr
    CP_cat['PSR_AGE'] = Pulsar_age_arr
    CP_cat['PSR_BSURF'] = Pulsar_bsurf_arr
    return CP_cat


def cross_match_UC(self, CP_cat):
    """
    Cross-match the ultracool dwarf catalogue with the sources
    """
    UC_table = Table.read(self.cataloguedir + 'uctable.dat', format='ascii')
    UC_crossmatch_arr = np.full(len(CP_cat), False, dtype=bool)
    UC_name_arr = np.full(len(CP_cat), '--', dtype='S28')
    UC_ra_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    UC_dec_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    for s, source in enumerate(CP_cat):
        ra_source = source['CP_RA']
        dec_source = source['CP_DEC']
        dist_arr = np.sqrt(np.square(UC_table['ra_j2000_formula'] - ra_source) + np.square(UC_table['dec_j2000_formula'] - dec_source))
        min_idx = np.argmin(np.abs(dist_arr))
        # Calculate maximum distance for a match
        bmaj, bmin = utilv.get_beam(self)
        dist = (np.max([bmaj, bmin]) * 1.0)
        if dist_arr[min_idx] <= dist:
            UC_crossmatch_arr[s] = True
            UC_name_arr[s] = str(UC_table[min_idx]['name'])
            UC_ra_arr[s] = UC_table[min_idx]['ra_j2000_formula']
            UC_dec_arr[s] = UC_table[min_idx]['dec_j2000_formula']
        else:
            pass
    CP_cat['UC_cross'] = UC_crossmatch_arr
    CP_cat['UC_ID'] = UC_name_arr
    CP_cat['UC_RA'] = UC_ra_arr
    CP_cat['UC_DEC'] = UC_dec_arr
    return CP_cat


def cross_match_TDE(self, CP_cat):
    """
    Cross-match the TDE catalogue with the sources
    """
    TDE_table = Table.read(self.cataloguedir + 'TDE.fits', format='fits')
    TDE_crossmatch_arr = np.full(len(CP_cat), False, dtype=bool)
    TDE_name_arr = np.full(len(CP_cat), '--', dtype='S20')
    TDE_ra_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    TDE_dec_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    for s, source in enumerate(CP_cat):
        ra_source = source['CP_RA']
        dec_source = source['CP_DEC']
        dist_arr = np.sqrt(np.square(TDE_table['RA'] - ra_source) + np.square(TDE_table['DEC'] - dec_source))
        min_idx = np.argmin(np.abs(dist_arr))
        # Calculate maximum distance for a match
        bmaj, bmin = utilv.get_beam(self)
        dist = (np.max([bmaj, bmin]) * 1.0)
        if dist_arr[min_idx] <= dist:
            TDE_crossmatch_arr[s] = True
            TDE_name_arr[s] = str(TDE_table[min_idx]['NAME'])
            TDE_ra_arr[s] = TDE_table[min_idx]['RA']
            TDE_dec_arr[s] = TDE_table[min_idx]['DEC']
        else:
            pass
    CP_cat['TDE_cross'] = TDE_crossmatch_arr
    CP_cat['TDE_ID'] = TDE_name_arr
    CP_cat['TDE_RA'] = TDE_ra_arr
    CP_cat['TDE_DEC'] = TDE_dec_arr
    return CP_cat


def cross_match_flares(self, CP_cat):
    """
    Cross-match the flares catalogue with the sources
    """
    flares_table = Table.read(self.cataloguedir + 'air_flares_export.txt', format='ascii.cds', readme=self.cataloguedir + 'air_flares_export_ReadMe.txt')
    flares_crossmatch_arr = np.full(len(CP_cat), False, dtype=bool)
    flares_name_arr = np.full(len(CP_cat), '--', dtype='S25')
    flares_ra_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    flares_dec_arr = np.full(len(CP_cat), np.nan, dtype=np.float64)
    for s, source in enumerate(CP_cat):
        ra_source = source['CP_RA']
        dec_source = source['CP_DEC']
        dist_arr = np.sqrt(np.square(flares_table['RAdeg'] - ra_source) + np.square(flares_table['DEdeg'] - dec_source))
        min_idx = np.argmin(np.abs(dist_arr))
        # Calculate maximum distance for a match
        bmaj, bmin = utilv.get_beam(self)
        dist = (np.max([bmaj, bmin]) * 1.0)
        if dist_arr[min_idx] <= dist:
            flares_crossmatch_arr[s] = True
            flares_name_arr[s] = str(flares_table[min_idx]['Name'])
            flares_ra_arr[s] = flares_table[min_idx]['RAdeg']
            flares_dec_arr[s] = flares_table[min_idx]['DEdeg']
        else:
            pass
    CP_cat['Flares_cross'] = flares_crossmatch_arr
    CP_cat['Flares_ID'] = flares_name_arr
    CP_cat['Flares_RA'] = flares_ra_arr
    CP_cat['Flares_DEC'] = flares_dec_arr
    return CP_cat


def write_CP_cat(self, cpcat):
    cpcat.write(self.circpolanalysisdir + '/CP_cat_final.txt', format='ascii', comment=False, overwrite=True)
import os
import glob
import shutil

import numpy as np
import astropy.io.fits as pyfits

import utilv
import fits_magic_v as fm

def remove_leakage(self):
    """
    Executes the whole subtraction of the polarisation leakage using the leakage beam models and Stokes I CLEAN-models
    """
    copy_images(self)
    remove_centres(self)
    subtract_leakage(self)
#    delete_obsfiles(self)


def copy_images(self):
    """
    Copy the relevant images and datasets to the directory for mosaicking
    """
    for b in range(40): # Iterate over all beams
        beamselfcaldir = os.path.join(self.basedir, self.obsid, str(b).zfill(2), 'selfcal')
        beamcontdir = os.path.join(self.basedir, self.obsid, str(b).zfill(2), 'continuum')
        # Get the names and directories of all the needed files
        cdatasetlist = glob.glob(beamselfcaldir + '/*.mir')
        cimagelist = glob.glob(beamcontdir + '/image_mf_*.fits')
        cmodellist = glob.glob(beamcontdir + '/model*')
        cmasklist = glob.glob(beamcontdir + '/mask_mf_*')
        # Check if all the files for the specific beam are available
        if len(cdatasetlist) == 1 and len(cimagelist) == 1 and len(cmodellist) == 1 and len(cmasklist) == 1:
            bmaj, bmin = fm.get_beam(os.path.join(self.basedir, self.obsid, str(b).zfill(2), 'polarisation') + '/image_mf_V.fits')
            rms = fm.get_rms(os.path.join(self.basedir, self.obsid, str(b).zfill(2), 'polarisation') + '/image_mf_V.fits')
            if bmaj <= self.circ_bmaj and bmin <= self.circ_bmin and rms <= self.circ_rmsclip:
                # Move to the target directory to shorten file location strings and copy files to the local directory
                os.chdir(self.leakagesubtractdir)
                shutil.copytree(cdatasetlist[0], self.leakagesubtractdir + '/' + str(b).zfill(2) + '.mir', dirs_exist_ok=True)
                os.system('uvaver vis=' + str(b).zfill(2) + '.mir' + ' stokes=V out=' + str(b).zfill(2) + '_uvaver.mir')
                shutil.copy(cimagelist[0], self.leakagesubtractdir + '/' + str(b).zfill(2) + '.Irstr.fits')
                shutil.copytree(cmodellist[0], self.leakagesubtractdir + '/' + str(b).zfill(2) + '.Imfcl', dirs_exist_ok=True)
                shutil.copytree(cmasklist[0], self.leakagesubtractdir + '/' + str(b).zfill(2) + '.Imask', dirs_exist_ok=True)
            else:
                print('Stokes V image for beam ' + str(b).zfill(2) + ' not used. It did not fulfill the requirements!')
        else:
            print('Some data does not seem to be available for beam ' + str(b).zfill(2) + '! Skipping beam!')


def remove_centres(self):
    """
    Function to remove the central artificial source caused by the correlator issue
    """
    for b in range(40): # Iterate over all beams
        os.chdir(self.leakagesubtractdir)
        # Generate a new image and model in Stokes V
        if os.path.isdir(str(b).zfill(2) + '_uvaver.mir') and os.path.isdir(str(b).zfill(2) + '.Imask'):
            os.system('invert vis=' + str(b).zfill(2) + '_uvaver.mir map=' + str(b).zfill(2) + '.Vmap beam=' + str(b).zfill(2) + '.Vbeam imsize=3073 cell=4 options=mfs,sdb,double stokes=v slop=1 robust=-2')
            os.system('fits op=xyout in=' + str(b).zfill(2) + '.Vmap out=' + str(b).zfill(2) + '.Vmap.fits')
            image_py = pyfits.open(str(b).zfill(2) + '.Vmap.fits')
            image_data_py = image_py[0].data
            imagestd_py = np.nanstd(image_data_py[0, 0, :, :])
            os.system('mfclean map=' + str(b).zfill(2) + '.Vmap beam=' + str(b).zfill(2) + '.Vbeam out=' + str(b).zfill(2) + '.Vmfcl niters=25000 region="mask(' + str(b).zfill(2) + '.Imask)" cutoff=' + str(imagestd_py * 0.5))
            os.system('restor map=' + str(b).zfill(2) + '.Vmap beam=' + str(b).zfill(2) + '.Vbeam model=' + str(b).zfill(2) + '.Vmfcl out=' + str(b).zfill(2) + '.Vrstr')
        else:
            print('Dataset and/or mask for beam ' + str(b).zfill(2) + ' not available!')
        # Isolate the central source and subtract it from the data
        if os.path.isdir(str(b).zfill(2) + '_uvaver.mir') and os.path.isdir(str(b).zfill(2) + '.Vmfcl') and os.path.isdir(str(b).zfill(2) + '.Vmap') and os.path.isdir(str(b).zfill(2) + '.mir') and os.path.isdir(str(b).zfill(2) + '.Imask'):
            # Extract the central source from the CLEAN model
            os.system('regrid in=' + str(b).zfill(2) + '.Vmfcl out=' + str(b).zfill(2) + '.Vmfcl_rg tin=' + str(b).zfill(2) + '.Vmap axes=1,2')
            os.system('imsub in=' + str(b).zfill(2) + '.Vmfcl_rg out=' + str(b).zfill(2) + '.Vmfcl_cs region="box(1534,1534,1539,1541)"')
            # Subtract the source from the data
            os.system('uvmodel vis=' + str(b).zfill(2) + '_uvaver.mir model=' + str(b).zfill(2) + '.Vmfcl_cs options=subtract,mfs out=' + str(b).zfill(2) + '_uvaver_cssub.mir')
            # Reimage the data
            os.system('invert vis=' + str(b).zfill(2) + '_uvaver_cssub.mir map=' + str(b).zfill(2) + '_cssub.Vmap beam=' + str(b).zfill(2) + '_cssub.Vbeam imsize=3073 cell=4 options=mfs,sdb,double stokes=v slop=1 robust=-2')
            os.system('fits op=xyout in=' + str(b).zfill(2) + '_cssub.Vmap out=' + str(b).zfill(2) + '_cssub.Vmap.fits')
            image_py = pyfits.open(str(b).zfill(2) + '_cssub.Vmap.fits')
            image_data_py = image_py[0].data
            imagestd_py = np.nanstd(image_data_py[0, 0, :, :])
            os.system('mfclean map=' + str(b).zfill(2) + '_cssub.Vmap beam=' + str(b).zfill(2) + '_cssub.Vbeam out=' + str(b).zfill(2) + '_cssub.Vmfcl niters=25000 region="mask(' + str(b).zfill(2) + '.Imask)" cutoff=' + str(imagestd_py * 0.5))
            os.system('restor map=' + str(b).zfill(2) + '_cssub.Vmap beam=' + str(b).zfill(2) + '_cssub.Vbeam model=' + str(b).zfill(2) + '_cssub.Vmfcl out=' + str(b).zfill(2) + '_cssub.Vrstr')
            # Write out the final image as a FITS-file compatible with amosaic
            os.system('fits op=xyout in=' + str(b).zfill(2) + '_cssub.Vrstr out=' + str(b).zfill(2) + '_cssub.Vrstr.fits')
        else:
            print('Removal of central artificial source for beam ' + str(b).zfill(2) + ' failed!')


def subtract_leakage(self):
    """
    Subtract the leakage using MIRIAD
    """
    for b in range(40):  # Iterate over all beams
        os.chdir(self.leakagesubtractdir)
        if os.path.isdir(str(b).zfill(2) + '_cssub.Vmfcl'):
            if self.leakmode == 'all':
                shutil.copy('/aux/bade/Apertif/aperv/data/allweeks/lbeams/' + str(b).zfill(2) + '/leakage_B' + str(b).zfill(2) + '.fits', self.leakagesubtractdir + '/' + str(b).zfill(2) + '_lpb.fits')
            elif self.leakmode == 'obsrun':
                shutil.copy(self.datadir + '/lbeams/' + str(b).zfill(2) + '/leakage_B' + str(b).zfill(2) + '.fits', self.leakagesubtractdir + '/' + str(b).zfill(2) + '_lpb.fits')
            else:
                print('Mode not supported!')
            # Convert the leakage beam model to MIRIAD format
            os.system('fits op=xyin in=' + str(b).zfill(2) + '_lpb.fits out=' + str(b).zfill(2) + '_lpb')
            # Regrid the continuum model back to the grid of the images
            os.system('regrid in=' + str(b).zfill(2) + '.Imfcl out=' + str(b).zfill(2) + '_rg.Imfcl tin=' + str(b).zfill(2) + '_cssub.Vrstr region="images(1)"')
            # Calculate the leakage model blanking the area of the central source
            os.system('maths exp="<' + str(b).zfill(2) + '_rg.Imfcl>*<' + str(b).zfill(2) + '_lpb>" out=' + str(b).zfill(2) + '_lmfcl region="box(1,1,3073,1533),box(1542,3073,1,3073),box(1,1,1533,3073),box(1,1542,3073,3073)"')
            # Set the Stokes parameter to V in model
            os.system('puthd in=' + str(b).zfill(2) + '_lmfcl/crval4 value=4')
            # Set the NaNs to zero in the leakage model
            os.system('fits op=xyout in='+ str(b).zfill(2) + '_lmfcl out='+ str(b).zfill(2) + '_lmfcl.fits')
            hdul = pyfits.open(str(b).zfill(2) + '_lmfcl.fits')
            hdul_data = hdul[0].data
            hdul_data_zeros = np.nan_to_num(hdul_data)
            pyfits.writeto(str(b).zfill(2) + '_zeros_lpb.fits', data=hdul_data_zeros, header=hdul[0].header, overwrite=True)
            os.system('fits op=xyin in='+ str(b).zfill(2) + '_zeros_lpb.fits out='+ str(b).zfill(2) + '_zeros_lpb')
            # Subtract the leakage
            os.system('uvmodel vis=' + str(b).zfill(2) + '_uvaver_cssub.mir model='+ str(b).zfill(2) + '_zeros_lpb options=subtract,mfs out=' + str(b).zfill(2) + '_uvaver_cssub_lmsub.mir')
            # Generate a new image
            os.system('invert vis=' + str(b).zfill(2) + '_uvaver_cssub_lmsub.mir map=' + str(b).zfill(2) + '_lmsub.Vmap beam=' + str(b).zfill(2) + '_lmsub.Vbeam imsize=3073 cell=4 options=mfs,sdb,double stokes=v slop=1 robust=-2')
            # Get the noise of the new image
            os.system('fits op=xyout in=' + str(b).zfill(2) + '_lmsub.Vmap out=' + str(b).zfill(2) + '_lmsub.Vmap.fits')
            image_py = pyfits.open(str(b).zfill(2) + '_lmsub.Vmap.fits')
            image_data_py = image_py[0].data
            imagestd_py = np.nanstd(image_data_py[0,0,:,:])
            # Clean the image using the continuum mask
            os.system('mfclean map=' + str(b).zfill(2) + '_lmsub.Vmap beam=' + str(b).zfill(2) + '_lmsub.Vbeam out=' + str(b).zfill(2) + '_lmsub.Vmfcl niters=25000 region="mask(' + str(b).zfill(2) + '.Imask)" cutoff=' + str(imagestd_py*0.5))
            # Restore the image
            os.system('restor map=' + str(b).zfill(2) + '_lmsub.Vmap beam=' + str(b).zfill(2) + '_lmsub.Vbeam model=' + str(b).zfill(2) + '_lmsub.Vmfcl out=' + str(b).zfill(2) + '_lmsub.Vrstr')
            # Convert to fits for mosaicking
            os.system('fits op=xyout in=' + str(b).zfill(2) + '_lmsub.Vrstr out=' + str(b).zfill(2) + '_lmsub.Vrstr.fits')
            # Swap sign in image in case observations were calibrated using 3C138
            if os.path.isfile(os.path.join(self.datadir, 'param_' + str(b).zfill(2) + '_preflag_3C286.npy')):
                pass
            elif os.path.isfile(os.path.join(self.datadir, 'param_' + str(b).zfill(2) + '_preflag_3C138.npy')):
                utilv.invert_values(self.leakagesubtractdir + '/' + str(b).zfill(2) + '_lmsub.Vrstr.fits', self.leakagesubtractdir + '/' + str(b).zfill(2) + '_lmsub.Vrstr.fits')
            else:
                print('No parameter file for beam ' + str(b).zfill(2) + ' available.')
        else:
            print('Some data does not seem to be available for beam ' + str(b).zfill(2) + '! Skipping beam!')
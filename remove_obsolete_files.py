import os


def remove_obsolete_files(self):
    remove_leakage_files(self)


def remove_analysis_files(self):
    """
    Remove the obsolete files
    """
    os.system('rm -rf ' + self.polanalysisleakagedirV + '/I_*.fits')
    os.system('rm -rf ' + self.polanalysisleakagedirV + '/I_*.fits.pybdsf.log')
    os.system('rm -rf ' + self.polanalysisleakagedirV + '/I_cat_B*.txt')
    os.system('rm -rf ' + self.polanalysisleakagedirV + '/V_*.fits')
    os.system('rm -rf ' + self.polanalysisleakagedirV + '/V_*_inv.fits')
    os.system('rm -rf ' + self.polanalysisleakagedirV + '/V_*_nd.fits')
    os.system('rm -rf ' + self.polanalysisleakagedirV + '/V_*_nd_cutout.fits')


def remove_leakage_files(self):
    """
    Delete the obsolete files from the previous steps
    """
    for b in range(40):
        beamleakdir = os.path.join(self.basedir, self.obsid, str(b).zfill(2), 'leakage')
        os.chdir(beamleakdir)
        os.system('rm -rf beam_c beam_c_centre *.mir I* image_mf_0[0-9].fits L* LM* M* map* mask* mcl* model* rstr*')
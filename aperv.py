import utilv
import fits_magic_v

import analyse_leakage
import remove_leakage
import generate_mosaics
import catalogue_v
import cross_match_v
import remove_obsolete_files


class aperv:
    """
    Class to remove leakage from Stokes images and analyse the final mosaics
    """
    module_name = 'APERV'

    def __init__(self, file_=None, **kwargs):
        self.default = utilv.load_config(self, file_)
        utilv.set_dirs(self)
        self.config_file_name = file_


    def go(self):
        """
        Function to remove the leakage from the (u,v)-data, reimage, source find and cross-match
        """
        utilv.gen_dirs(self)

        if self.gen_mosaic:
            generate_mosaics.generate_mosaics(self)
        if self.ana_leakage:
            analyse_leakage.analyse_leakage(self)
        if self.cat_v:
            catalogue_v.catalogue_v(self)
        if self.cr_match:
            cross_match_v.cross_match_v(self)
#        remove_leakage.remove_leakage(self)
#        cross_match_v.cross_match_v(self)
#       remove_obsolete_files(self)
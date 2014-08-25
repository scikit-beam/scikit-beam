
#TODO: Need to sort out tests for each function and operation as a whole.

#Reference am files:
f_path = '/home/giltis/dev/my_src/test_data/file_io/am_files/'

# Avizo v.6.x file format test files
# ----------------------------------
v6_am_binary_data = 'gScale_test_av6_binary.am' #Grayscale volume: float dtype
v6_am_ascii_data = 'gScale_test_av6_ascii.am'
v6_am_zip_data = 'gScale_test_av6_zip.am'

# Avizo v.7.x file format test files
# ----------------------------------
v7_am_binary_data = 'gScale_test_av7_binary.am'
v7_am_ascii_data = 'gScale_test_av7_ascii.am'
v7_am_zip_data = 'gScale_test_av7_zip.am'

# Avizo v.8.x file format test files
# ----------------------------------
# v8_am_binary_data = XXXXXXXXX

# Avizo data type test files, sourced from Avizo v.7,x
# ----------------------------------------------------
#fname_short = 'C2_dType_Short.am' #Grayscale volume: short dtype
#fname_test = 'APS_2C_Raw_Abv_CROP_tester.am' #Grayscale volume: float dtype
#fname_dbasin = 'C2_dBasin.am' #labelfield: ushort dtype
#fname_label = 'C2_LabelField.am' #labelfield: ushort dtype
#fname_label2 = 'Rad1_blw_GlsBd-Label.surf' #surface file. not sure we can read yet
#fname_binary = 'Shew_C8_bio_blw_GlsBd-Bnry.am' #binary data set: byte dtype
#fname_list = [fname_flt, fname_short, fname__test, fname_dbasin, fname_label, fname_label2, fname_binary]
#head_list = [head_flt, head_short, head_test, head_dbasin, head_label, head_label2, head_binary]
#data_list =
"""
FunTest data sets for Avizo 6.x
Types of data that is currently able to be loaded:
    Grayscale data
    Binary data (highlighting a particular phase, or material)
    Labeled data (e.g. after segmentation, and prior to surface generation)

"""
def test_read_amira():
    pass

def test_cnvrt_amira_data_2numpy():
    pass

def test_sort_amira_header():
    pass

def test_create_md_dict():
    pass

def test_load_amiramesh_as_np():
    pass

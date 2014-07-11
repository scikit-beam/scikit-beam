def read_amira_header {}{
    """
    Standard Header Format: Avizo Binary file
    -----------------------------------------
    Line #      Contents
    ------      --------
    0           # Avizo BINARY-LITTLE-ENDIAN 2.1
    1           '\n',
    2           '\n', 
    3            'define Lattice 426 426 121\n', 
    4           '\n', 
    5           'Parameters {\n',    
    6                        'Units {\n',
    7                               'Coordinates "m"\n',
    8                               '}\n',
    9                        'Colormap "labels.am",\n',
    10                       'Content "426x426x121 ushort, uniform coordinates",\n',
    11                       'BoundingBox 1417.5 5880 1407 5869.5 5649 6909,\n',
    12                       'CoordType "uniform"\n',
    13                       '}\n',
    14          '\n', 
    15          'Lattice { ushort Labels } @1(HxByteRLE,44262998)\n',
    16          '\n', 
    17          '# Data section follows\n'
    """
    }

#Reference am files:
fname_orig = '/home/giltis/Dropbox/BNL_Docs/Alt_File_Formats/am_FILES/raw_cnvrted/Control_column_1_DI_A_Above-Edge.am'
def read_amira (src_file):
    f = open(src_file, 'r')
    while True:
        line = f.readline()
        try:
            am_header == []
        except:
            am_header = []
        am_header.append(line)
        if (line == '# Data section follows\n'):
            f.readline()
            break
    try:
        am_data == []
    except:
        am_data = []
    while True:
        am_data.append(f.readline())
    return am_header, am_data

    file_header = 
    file_breakdown = f.readlines()
    header_end_line_num = file_breakdown.index('# Data section follows\n')

def create_md_dict (header_list):
    for _ in range(len(header_list)):
        head_list[_] = header_list[_].strip
        head_list[_] = header_list[_].split(" ")
    md_dict = {
        'software_src' : head_list[0][1],
        'data_format' : head_list[0][2],
        'data_format_version' : head_list[0][3]
        'array_dims' : {'x_dim' : int(head_list[3][2]),
                        'y_dim' : int(head_list[3][3]),
                        'z_dim' : int(head_list[3][4])
                        },
        'data_type' : head_list[6][2],
        'bounding_box' : {'x_min' : head_list[7][1],
                          'x_max' : head_list[7][2],
                          'y_min' : head_list[7][3],
                          'y_max' : head_list[7][4],
                          'z_min' : head_list[7][5],
                          'z_max' : head_list[7][6]
                          },
        'coordinate_type' : 'x_min' : head_list[8][1],
                              '
        

    
    

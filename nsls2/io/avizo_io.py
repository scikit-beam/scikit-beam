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
fname_orig = '/home/giltis/Dropbox/BNL_Docs/Alt_File_Formats/am_cnvrt_compare/Shew_C5_bio_abv.am'
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
    am_data = f.read()
    f.close()
    return am_header, am_data

def cnvrt_amira_data_2numpy (data, header):
    Zdim = *from_header*
    Ydim = *from_header*
    Xdim = *from_header*
    #Strip out null characters from the string of binary values
    data_strip = am_data.strip('\n')
    flt_values = np.fromsstring(data_strip, '<f4')
    #Resize the 1D array to the correct ndarray dimensions
    flt_values.resize(Zdim, Ydim, Xdim)
    return am_header, flt_values

def sort_amira_header (header_list):
    for row in range(len(header_list)):
        header_list[row] = header_list[row].strip('\n')
        header_list[row] = header_list[row].split(" ")
        for column in range(len(header_list[row])):
            header_list[row] = filter(None, header_list[row])
    header_list = filter(None, header_list)
    return header_list

def create_md_dict (header_list):
    md_dict = {'software_src' : header_list[0][1],
               'data_format' : header_list[0][2],
               'data_format_version' : header_list[0][3]
                }
    for row in range(len(header_list)):
        try:
            md_dict['array_dims'] = {'x_dim' : int(header_list[row][header_list[row].index('define') + 2]),
                                     'y_dim' : int(header_list[row][header_list[row].index('define') + 3]),
                                     'z_dim' : int(header_list[row][header_list[row].index('define') + 4])
                                     }
        except:
        #    continue
            try:
                md_dict['data_type'] = header_list[row][header_list[row].index('Content') + 2]
            except: 
            #    continue
                try:
                    md_dict['coord_type'] = header_list[row][header_list[row].index('CoordType') + 1]
                except:
                    try:
                        md_dict['bounding_box'] = {'x_min' : float(header_list[row][header_list[row].index('BoundingBox') + 1]),
                                                   'x_max' : float(header_list[row][header_list[row].index('BoundingBox') + 2]),
                                                   'y_min' : float(header_list[row][header_list[row].index('BoundingBox') + 3]),
                                                   'y_max' : float(header_list[row][header_list[row].index('BoundingBox') + 4]),
                                                   'z_min' : float(header_list[row][header_list[row].index('BoundingBox') + 5]),
                                                   'z_max' : float(header_list[row][header_list[row].index('BoundingBox') + 6])
                                                   }
                    except:
                        continue
    return md_dict

                              '
        

    
    

from collections import (Mapping, OrderedDict)
#from collections import OrderedDict
import six

import xraylib


line_name = ['Ka1', 'Ka2', 'Kb1', 'Kb2', 'La1', 'La2', 'Lb1', 'Lb2', 'Lb3', 'Lb4', 'Lb5',
             'Lg1', 'Lg2', 'Lg3', 'Lg4', 'Ll', 'Ln', 'Ma1', 'Ma2', 'Mb', 'Mg']
line_list = [xraylib.KA1_LINE, xraylib.KA2_LINE, xraylib.KB1_LINE, xraylib.KB2_LINE,
             xraylib.LA1_LINE, xraylib.LA2_LINE,
             xraylib.LB1_LINE, xraylib.LB2_LINE, xraylib.LB3_LINE, xraylib.LB4_LINE, xraylib.LB5_LINE,
             xraylib.LG1_LINE, xraylib.LG2_LINE, xraylib.LG3_LINE, xraylib.LG4_LINE,
             xraylib.LL_LINE, xraylib.LE_LINE,
             xraylib.MA1_LINE, xraylib.MA2_LINE, xraylib.MB_LINE, xraylib.MG_LINE]
line_dict = dict(zip(line_name, line_list))


bindingE = ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5',
            'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2', 'O3', 'O4', 'O5', 'P1', 'P2', 'P3']
shell_list = [xraylib.K_SHELL, xraylib.L1_SHELL, xraylib.L2_SHELL, xraylib.L3_SHELL,
              xraylib.M1_SHELL, xraylib.M2_SHELL, xraylib.M3_SHELL, xraylib.M4_SHELL, xraylib.M5_SHELL,
              xraylib.N1_SHELL, xraylib.N2_SHELL, xraylib.N3_SHELL, xraylib.N4_SHELL,
              xraylib.N5_SHELL, xraylib.N6_SHELL, xraylib.N7_SHELL,
              xraylib.O1_SHELL, xraylib.O2_SHELL, xraylib.O3_SHELL, xraylib.O4_SHELL, xraylib.O5_SHELL,
              xraylib.P1_SHELL, xraylib.P2_SHELL, xraylib.P3_SHELL]
shell_dict = dict(zip(bindingE, shell_list))


_XRAYLIB_MAP = {'lines': (line_dict, xraylib.LineEnergy),
                'cs': (line_dict, xraylib.CS_FluorLine),
                'binding_e': (shell_dict, xraylib.EdgeEnergy),
                'jump': (shell_dict, xraylib.JumpFactor),
                'yield': (shell_dict, xraylib.FluorYield),
                }


_OTHER_VALUES = {'H': {'Z': 1, 'rho': .524, },
                 'He': {'Z': 2, }}


class Element(object):

    def __init__(self, element, energy):
        self._element = element
        self._energy = energy
        self.emission_line = _XrayLibWrap('lines', element)
        self.cs = _XrayLibWrap('cs', element, energy)
        self.bind_energy = _XrayLibWrap('binding_e', element)
        self.jump_factor = _XrayLibWrap('jump', element)
        self.f_yield = _XrayLibWrap('yield', element)

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, in_val):
        self._energy = in_val
        #self.xrf.energy = in_val
        #self.levels.energy = in_val


class _XrayLibWrap(Mapping):

    def __init__(self, info_type,
                 element, energy=None):

        self.info_type = info_type
        self._map, self._func = _XRAYLIB_MAP[info_type]
        self._keys = sorted(list(six.iterkeys(self._map)))
        self._element = element
        self._energy = energy

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, in_val):
        # optional sanity checks?
        self._energy = in_val

    def __getitem__(self, key):
        if self.info_type == 'cs':
            return self._func(self._element,
                              self._map[key],
                              self.energy)
        else:
            return self._func(self._element,
                              self._map[key])

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)


e = Element(30, 10)
print e.emission_line['Ka1']
print e.cs['Ka1']
print e.f_yield['K']
#e.energy = new_energy

print line_dict
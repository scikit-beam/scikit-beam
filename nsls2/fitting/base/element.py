from collections import Mapping
import six

from element_data import XRAYLIB_MAP, OTHER_VAL


class Element(object):

    def __init__(self, element, energy):

        if isinstance(element, str):
            item_val = OTHER_VAL[OTHER_VAL[:, 0] == element][0]
        elif isinstance(element, int):
            item_val = OTHER_VAL[element-1]
        else:
            raise TypeError('Please define element by '
                            'atomic number z or element name')
        self.name = item_val[0]
        self.z = item_val[1]['Z']
        self.mass = item_val[1]['mass']
        self.density = item_val[1]['rho']
        self._element = self.z


        if not isinstance(energy, float and int):
            raise TypeError('Expected a number for energy')
            self._energy = energy

        self.emission_line = _XrayLibWrap('lines', self._element)
        self.cs = _XrayLibWrap('cs', self._element, energy)
        self.bind_energy = _XrayLibWrap('binding_e', self._element)
        self.jump_factor = _XrayLibWrap('jump', self._element)
        self.f_yield = _XrayLibWrap('yield', self._element)


    @property
    def element(self):
        return self._element

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, val):
        if not isinstance(val, float and int):
            raise TypeError('Expected a number for energy')
        self._energy = val
        self.cs.energy = val


class _XrayLibWrap(Mapping):
    """
    This is an interface to wrap xraylib to obtain calculations related
    to xray fluorescence.

    Attributes
    ----------
    info_type : string
        defines which physics quantity to calculate
    element : int
        atomic number
    energy : float, optional
        incident energy for fluorescence
    """
    def __init__(self, info_type,
                 element, energy=None):
        self.info_type = info_type
        self._map, self._func = XRAYLIB_MAP[info_type]
        self._keys = sorted(list(six.iterkeys(self._map)))
        self._element = element
        self._energy = energy

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, val):
        """
        Parameters
        ----------
        val : float
            new energy value
        """
        self._energy = val

    def __getitem__(self, key):
        """
        call xraylib function to calculate physics quantity

        Parameters
        ----------
        key : string
            defines which physics quantity to calculate
        """
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


e = Element('Zn', 10)
#e.energy='A'
print e.emission_line['Ka1']
print e.cs['Ka1']
print e.f_yield['K']
#e.energy = new_energy

print line_dict
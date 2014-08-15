from collections import Mapping
import six


_XRAYLIB_MAP_MAP = {'edges': ({'Ka1': xraylib.KALPHA,..}, xraylib.CS_FlourLine),
                    'levels': ({'K': xraylib.K, }),
                    }


_OTHER_VALUES = {'H': {'Z': 1, 'rho': .524, },
                 'He': {'Z': 2, }}


class Element(object):
    def __init__(self, elmement, energy):
        self._element = element
        self._energy = energy
        self.xrf = XrayLib_wrap('edges', element, energy)
        self.levels = XrayLib_wrap('levels', element, energy)


    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, in_val):
        self._energy = in_val
        self.xrf.energy = in_val
        self.levels.energy = in_val


class _XrayLib_wrap(Mapping):
    def __init__(self, info_type, element, energy):
        self._map, self._func = _XRAYLIB_MAP_MAP[info_type]
        self._keys = six.keys(self._map)
        self._element = element
        self._enegry = energy

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, in_val):
        # optional sanity checks?
        self._energy = in_val

    def __getitem__(self, key):
        return self._func(self._element,
                          self._map[key],
                          self.energy)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)


e = Element('H')
e.xrf['Ka1']
e.levels['K']
e.energy = new_energy
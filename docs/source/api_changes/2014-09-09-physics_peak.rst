=========
 Changes
=========
- changed function arguments, including add center positions, and put independent variable x as first argument to function

from ::

  def gauss_peak(area, sigma, dx)

to::

  def gauss_peak(x, area, center, sigma)

from ::

  gauss_step(area, sigma, dx, peak_e)

to ::

  gauss_step(x, area, center, sigma, peak_e)

from ::

  def gauss_tail(area, sigma, dx, gamma)

to ::

  gauss_tail(x, area, center, sigma, gamma)

from ::

  elastic_peak(coherent_sct_energy,fwhm_offset, fwhm_fanoprime, area, ev, epsilon=2.96)

to ::

  elastic_peak(x, coherent_sct_energy, fwhm_offset, fwhm_fanoprime, area, epsilon=2.96)

from ::

  def compton_peak(coherent_sct_energy, fwhm_offset, fwhm_fanoprime,
                   compton_angle, compton_fwhm_corr, compton_amplitude,
                   compton_f_step, compton_f_tail, compton_gamma,
                   compton_hi_f_tail, compton_hi_gamma,
                   area, ev, epsilon=2.96, matrix=False)

to ::

  def compton_peak(x, coherent_sct_energy, fwhm_offset, fwhm_fanoprime,
                   compton_angle, compton_fwhm_corr, compton_amplitude,
                   compton_f_step, compton_f_tail, compton_gamma,
                   compton_hi_f_tail, compton_hi_gamma,
                   area, epsilon=2.96, matrix=False)

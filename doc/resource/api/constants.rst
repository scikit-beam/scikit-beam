========================
 :mod:`constants` Module
========================

.. currentmodule:: nsls2.constants

Elemental data
--------------
Tools for working with elemental constants.

.. autosummary::
   :toctree: generated/

   emission_line_search
   Element




X-ray fluorescence data
~~~~~~~~~~~~~~~~~~~~~~~

The c library `xraylib <https://github.com/tschoonj/xraylib/wiki>`__
provides a programmatic interface to x-ray fluorescence constants.  We
provide dictionary-like wrapping classes to provide easy access to the
data.  For the most part users should access this data through the
`Element` object which manages the creation and access to these objects.

.. autosummary::
   :toctree: generated/

   XrayLibWrap
   XrayLibWrap_Energy

Powder Diffraction Standards
----------------------------


.. autosummary::
   :toctree: generated/

   calibration_standards
   PowderStandard
   Reflection
   HKL







Converting :math:`2\theta` between wavelengths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Given that

    .. math ::

        \frac{\lambda_c}{2 a} \sqrt{h^2 + k^2 + l^2} = \sin\left(\frac{2\theta_c}{2}\right)

    If we multiply both sides by
    :math:`\frac{\lambda_n}{\lambda_c}` then we have

    .. math ::

        \frac{\lambda_n}{2 a} \sqrt{h^2 + k^2 + l^2} = \frac{\lambda_n}{\lambda_c} \sin\left(\frac{2\theta_c}{2}\right)

        \sin\left(\frac{2\theta_n}{2}\right) = \frac{\lambda_n}{\lambda_c} \sin\left(\frac{2\theta_c}{2}\right)

    which solving for :math:`2\theta_n` gives us

    .. math ::

       2\theta_n = 2 \arcsin\left(\frac{\lambda_n}{\lambda_c} \sin\left(\frac{2\theta_c}{2}\right)\right)

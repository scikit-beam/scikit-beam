========================
 :mod:`constants` Module
========================

Some diffraction math
---------------------

Given that

    .. math ::

        \\frac{\\lambda_c}{2 a} \\sqrt{h^2 + k^2 + l^2} = \\sin\\left(\\frac{2\\theta_c}{2}\\right)

    If we multiply both sides by
    :math:`\\frac{\\lambda_n}{\\lambda_c}` then we have

    .. math ::

        \\frac{\lambda_n}{2 a} \\sqrt{h^2 + k^2 + l^2} = \\frac{\\lambda_n}{\\lambda_c} \\sin\\left(\\frac{2\theta_c}{2}\\right)

        \\sin\\left(\\frac{2\\theta_n}{2}\\right) = \\frac{\\lambda_n}{\\lambda_c} \\sin\\left(\\frac{2\\theta_c}{2}\\right)

    which solving for :math:`2\\theta_n` gives us

    .. math ::

       2\\theta_n = 2 \\arcsin\\left(\\frac{\\lambda_n}{\\lambda_c} \\sin\\left(\\frac{2\\theta_c}{2}\\right)\\right)


.. automodule:: nsls2.constants
   :members:
   :show-inheritance:
   :undoc-members:

Install HiClass
===============

To install HiClass from the Python Package Index (PyPI) simply run:

.. code-block:: bash

    pip install hiclass

Additionally, it is also possible to install optional packages along. To install optional packages run:

.. code-block:: bash

    pip install hiclass"[<extra_name>]"

:literal:`<extra_name>` can have one of the following options:

- ray: Installs the ray package, which is required for parallel processing support.

It is also possible to install HiClass using :literal:`conda`, as follows:

.. code-block:: bash

    conda install -c conda-forge hiclass --yes

.. note::

    We recommend using :literal:`pip` at this point to eliminate any potential dependency issues.

.. toctree::
    :hidden:

    verify

conda
=====

Install :literal:`conda` on your computer, following the `official guide <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

Create a new virtual environment called :literal:`hiclass-environment` using :literal:`conda`:

.. code-block:: bash

    conda create --name hiclass-environment python=3.8 --yes


This will create an isolated Python 3.8 environment. To activate it:

.. code-block:: bash

    conda activate hiclass-environment


To exit :literal:`hiclass-environment`:

.. code-block:: bash

    conda deactivate


.. note::

    The :literal:`conda` virtual environment is not dependent on your current working directory and can be activated from any folder.

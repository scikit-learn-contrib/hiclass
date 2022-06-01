venv
====

If you are using Python 3, you should already have the :literal:`venv` module installed with the standard library. Create a directory for HiClass within your virtual environment:

.. code-block:: bash

    mkdir hiclass-environment && cd hiclass-environment


This will create a folder called :literal:`hiclass-environment` in your current working directory. Then you should create a new virtual environment in this directory by running:

.. tabs::

    .. code-tab:: bash
        :caption: GNU/Linux or macOS

        python -m venv env/hiclass-environment

    .. code-tab:: bash
        :caption: Windows

        python -m venv env\hiclass-environment

Activate this virtual environment:

.. tabs::

    .. code-tab:: bash
        :caption: GNU/Linux or macOS

        source env/hiclass-environment/bin/activate

    .. code-tab:: bash
        :caption: Windows

        .\env\hiclass-environment\Scripts\activate

To exit the environment:

.. code-block:: bash

    deactivate

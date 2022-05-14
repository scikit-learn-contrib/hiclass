venv
====

If you are using Python 3, you should already have the :literal:`venv` module installed with the standard library. Create a directory for working with HiClass within your virtual environment:

.. code-block:: bash

    mkdir hiclass-environment && cd hiclass-environment


This will create a folder called :literal:`hiclass-environment` in your current working directory. Then you should create a new virtual environment in this directory by running:

.. code-block:: bash

    python -m venv env/hiclass-environment  # macOS / Linux
    python -m venv env\hiclass-environment  # Windows


Activate this virtual environment:

.. code-block:: bash

    source env/hiclass-environment/bin/activate # macOS / Linux
    .\env\hiclass-environment\Scripts\activate  # Windows


To exit the environment:

.. code-block:: bash

    deactivate

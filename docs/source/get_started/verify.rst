Verify a successful installation
================================

To check that HiClass is installed, start the Python interpreter by running ``python`` on the terminal, then try to import HiClass:

.. code-block:: python

    Python 3.8.13 (default, Mar 28 2022, 11:38:47)
    [GCC 7.5.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import hiclass
    >>>

If everything goes smoothly, it means your installation works. However, you should see an error if HiClass was not installed successfully. For example:

.. code-block:: python

    Python 3.9.7 (default, Sep 16 2021, 13:09:58)
    [GCC 7.5.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import hiclass
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ModuleNotFoundError: No module named 'hiclass'

If you have any problems with your installation, please open an issue describing it at `https://github.com/mirand863/hiclass/issues <https://github.com/mirand863/hiclass/issues>`_.

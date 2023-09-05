from setuptools import setup, find_packages

__package_name__ = "haca3"


def get_version_and_cmdclass(pkg_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(pkg_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.get_cmdclass(pkg_path)


__version__, cmdclass = get_version_and_cmdclass(__package_name__)


# noinspection PyTypeChecker
setup(
    name=__package_name__,
    version=__version__,
    description="HACA3: A unified approach for multi-site MR image harmonization",
    long_description="HACA3: A unified approach for multi-site MR image harmonization",
    author="Lianrui Zuo",
    author_email="lr_zuo@jhu.edu",
    url="https://gitlab.com/iacl/haca3",
    license="Apache License, 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    keywords="mri harmonization",
    entry_points={
        "console_scripts": [
            "haca3-train=haca3.train:main",
            "haca3-test=haca3.test:main",
            "run-haca3=haca3.main:main",
        ]
    },
    install_requires=[
        "nibabel",
        "numpy",
        "scipy",
        "torch",
        "torchvision",
        "tqdm",
        "torchio",
        "scikit-image"
    ],
    cmdclass=cmdclass,
)

from setuptools import setup, find_packages

__package_name__ = "smore"


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
    description="SMORE: Synthetic Multi-Orientation Resolution Enhancement",
    long_description="SMORE: Synthetic Multi-Orientation Resolution Enhancement",
    author="Samuel Remedios",
    author_email="sremedi1@jhu.edu",
    url="https://gitlab.com/iacl/smore",
    license="Apache License, 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    keywords="mri ct super-resolution",
    entry_points={
        "console_scripts": [
            "smore-train=smore.train:main",
            "smore-test=smore.test:main",
            "run-smore=smore.main:main",
        ]
    },
    install_requires=[
        "nibabel",
        "numpy",
        "scipy",
        "torch>=2.0",
        "tqdm",
        "resize @ git+https://gitlab.com/shan-utils/resize@0.1.3",
        "degrade @ git+https://gitlab.com/iacl/degrade@v0.2.0",
        "transforms3d",
    ],
    cmdclass=cmdclass,
)

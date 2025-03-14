from setuptools import setup, find_packages

setup(
    name="fast_einsum",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "fast_einsum": ["*.so"],
    },
    install_requires=[
        "numpy",
    ],
    extras_require={
        "test": ["pytest"]
    },
    python_requires=">=3.13",
    author="Sonja Weitzing and Erik Henicke",
    author_email="sonja.weitzing@uni-jena.de, erik.henicke@uni-jena.de",
    description="Fast einsum with batch matrix multiplication",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="GNU General Public License v3.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
    ],
)

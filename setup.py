from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

required_packages = [
    "lightningflower", "lightningdata-modules", "torchmetrics"
]

setup(
    name="proto_fs",
    version="0.1.0",
    description="Using prototypical networks and Few-Shot Learning in a Federated Learning setup",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ManuelRoeder/proto_fs",
    author="Manuel Roeder",
    author_email="manuel.roeder@web.de",
    license="MIT",
    #packages=["lightningflower"],
    install_requires=required_packages,
    python_requires='>=3.8.12',
    package_data={"": ["README.md", "LICENSE"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)
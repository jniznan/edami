from setuptools import setup

setup(
    name='Edami',
    version='0.0.1',
    author='Juraj Niznan',
    author_email='jurajniznan@gmail.com',
    packages=['edami'],
    scripts=[],
    description='Educational Data Mining Module',
    install_requires=[
        "pandas >= 0.12.0",
    ],
)

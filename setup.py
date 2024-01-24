from setuptools import setup, find_packages

setup(
    name='abstrakTS',
    version='0.0.5',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23.5',
        'pandas>=1.5.3',
        'matplotlib>=3.7.1',
        'plotly>=5.15.0',
        'scikit-learn>=1.2.2',
        'tensorflow>=2.10.1'
        ],
    url='https://github.com/rizalpurnawan23/AbstrakTS'
)

from setuptools import setup, find_packages

setup(
    name='abstrakTS',
    version='0.0.5',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23.5',
        'pandas>=2.0.3',
        'matplotlib>=3.7.4',
        'plotly>=5.16.1',
        'scikit-learn>=1.2.2',
        'tensorflow>=2.10.1'
        ],
    url='https://github.com/rizalpurnawan23/AbstrakTS'
)

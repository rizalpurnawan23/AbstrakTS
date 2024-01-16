from setuptools import setup, find_packages

setup(
    name='abstrakTS',
    version='0.0.3',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23.5',
        'pandas>=2.1.1',
        'matplotlib>=3.8.0',
        'plotly>=5.18.0',
        'scikit-learn>=1.3.0',
        'tensorflow>=2.10.1'
        ],
    url='https://github.com/rizalpurnawan23/AbstrakTS'
)

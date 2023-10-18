from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


PACKAGE_NAME = 'pymops'
VERSION = '1.1.0'
DESCRIPTION = 'Multi-agent reinforcement learning simulation environment for multi-objective optimization in power scheduling'
LONG_DESCRIPTION = long_description  
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown' 
KEYWORDS = ['Economic Dispach', 'Power Scheduling', 'Reinforcement Learning','Unit Commitment'] 
AUTHOR = 'Awol Seid Ebrie and Young Jin Kim'
EMAIL = 'es.awol@gmail.com'
LICENSE = 'MIT'
INSTALL_REQUIRES = ['numpy', 'pandas', 'scipy', 'torch', 'tqdm']

setup(
	name = PACKAGE_NAME,
	version = VERSION,
	description = DESCRIPTION,
	long_description = LONG_DESCRIPTION,
	long_description_content_type = LONG_DESCRIPTION_CONTENT_TYPE,
	keywords = KEYWORDS,
	author = AUTHOR,
	author_email = EMAIL,
	license = LICENSE,
	packages = find_packages(),
	install_requires = INSTALL_REQUIRES,
	classifiers = [
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
	],
	url = 'https://github.com/awolseid/pymops',
    project_urls = {
        'Bug Tracker': 'https://github.com/awolseid/pymops'
    },
	python_requires = '>=3.9',
    include_package_data = True,
    package_data = {'pymops': ['data/*.csv']},
)

# -*- coding: utf-8 -*-
# @Time     : 2022/11/15 21:14
# @File     : setup.py
# @Author   : Zhou Hang
# @Email    : zhouhang@idataway.com
# @Software : Python 3.7
# @About    :
from distutils.core import setup

from setuptools import find_packages, setup

setup(
  name="transformers_expand",
  version="0.0.1.dev1",
  long_description=open("README.md", "r", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  license="Apache",
  package_dir={"": "src"},
  packages=find_packages("src"),
  description = 'transformers 拓展',
  author = 'ZhouHang',
  author_email = '841765793@qq.com',
  url = 'https://github.com/casuallyName/transformers_expand/',
  download_url = 'https://github.com/casuallyName/transformers_expand/archive/master.zip',
  keywords = ['transformers_expand', 'transformers'],   # Keywords that define your package best
  install_requires=['transformers'],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    "License :: OSI Approved :: Apache Software License",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
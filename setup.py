#!/usr/bin/env python

from distutils.core import setup
import os.path as osp

def _read_reqs(relpath):
    fullpath = osp.join(osp.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]

REQUIREMENTS = _read_reqs("requirement.txt")

setup(
    name='kn_util',
    version='1.0',
    description='Swift Development Utilities',
    author='NickyMouse',
    author_email='jing005@e.ntu.edu.sg',
    packages=['kn_util'],
    install_requires=REQUIREMENTS
)

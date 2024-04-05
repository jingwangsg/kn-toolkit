#!/usr/bin/env python

from distutils.core import setup
import os.path as osp
import sys

REQUIREMENT_FELE = "requirement.txt"
if sys.platform == "darwin":
    REQUIREMENT_FELE = "requirement-mac.txt"


def _read_reqs(relpath):
    fullpath = osp.join(osp.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [
            s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))
        ]


REQUIREMENTS = _read_reqs(REQUIREMENT_FELE)

setup(
    name="kn_util",
    version="1.0",
    description="Swift Development Utilities",
    author="NickyMouse",
    author_email="jing005@e.ntu.edu.sg",
    packages=["kn_util"],
    entry_points={
        "console_scripts": [
            "klfs = kn_util.tools.lfs:main",
            "krsync = kn_util.tools.rsync:main",
            "kget = kn_util.tools.download:main",
            "kbrew = kn_util.tools.brew:main",
        ]
    },
    install_requires=REQUIREMENTS,
)

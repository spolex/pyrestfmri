#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 23:07:46 2017

@author: spolex
"""
from os import path as op
from os import makedirs as om
from itertools import chain, imap

def flatmap(f, items):
  return chain.from_iterable(imap(f, items))

def create_dir(directory):
  if not op.exists(directory):
    om(directory)
  return directory
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:24:20 2017

DataGrabber and DataSink: tipicamente los estudios de neuroimagen requieren
ejecutar un pipeline en múltiples usuarios o con diferentes parametrizaciones 
de los algoritmos. Para este propósito haciendo uso de la clase _DataGrabber_
obtenemos los datos almacenados en el FS mientras que haciendo uso de la clase
_DataSink_ se almacenan los datos creados en una estructura jerárquica de di-
rectorios.

@author: spolex
"""

import os

# In[]
## DataGrabber 
# es la interfaz para obtener los datos a partir de los archivos en disco.
import nipype.interfaces.io as nio
datasource1 = nio.DataGrabber()
datasource1.inputs.base_directory = os.getcwd()
datasource1.inputs.template = 'data/subject001/func_data.nii.gz'
datasource1.inputs.sort_filelist = True
results = datasource1.run()

# In[]
# También es posible obtener archivos a partir de un patrón 

datasource2 = nio.DataGrabber()
datasource2.inputs.base_directory = '/media/spolex/data_nw/Dropbox_old/Dropbox/TFM-Elekin/TFM' #indica el directorio raíz en el que realizar la búsqueda
datasource2.inputs.template = 'datos/T003/*' #indica el patrón a machear
datasource2.inputs.sort_filelist = True
datasource2.run()

# In[]
datasource3 = nio.DataGrabber()
datasource3 = nio.DataGrabber(infields=['run'])
datasource3.inputs.base_directory = '/media/spolex/data_nw/Dropbox_old/Dropbox/TFM-Elekin/TFM'
datasource3.inputs.template = 'datos/T%/*' #indica el patrón a machear
datasource3.inputs.sort_filelist = True
datasource3.inputs.run = [013, 014] # para completar las entradas % en el patrón
# Esto buscará dentro de los directorios T013 y T014 los archivos que cumplan el patrón

# In[]
# Un caso de uso más realista supone obtener diferentes archivos de un sujeto dado
# y almacenarlo con un significado semántico.

datasource = nio.DataGrabber(infields=['subject_id'], outfields=['func', 'struct'])
datasource.inputs.base_directory = '/media/spolex/data_nw/Dropbox_old/Dropbox/TFM-Elekin/TFM'
datasource.inputs.template = '*'
datasource.inputs.sort_filelist = True
datasource.inputs.field_template = dict(func='datos/%s/f1.nii.gz',
                                        struct='datos/%s/mprage.nii.gz')
datasource.inputs.subject_id = ['T013', 'T014']
datasource.run()

# In[]
datasink = nio.DataSink()
datasink.inputs.base_directory = '/media/spolex/data_nw/Master/TFM/datos/sujetos/outputs'
"""
Helper functions.
"""

import datetime

def namespace(infile, experiment_num):
    now = datetime.datetime.now()
    return '{}_exp{}_d{}_{}h-{}m_'.format(infile, experiment_num, now.date(), now.hour, now.minute)

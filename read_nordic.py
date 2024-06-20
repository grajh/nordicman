# -*- coding: utf-8 -*-

#**********************************************************************
# Script name: read_nordic.py
# Version: 2024.06.20
# Description: Python tool for reading and manipulating seismological
# Nordic event (earthquake) data format. Useful methods for earthquake
# and phase (pick) filtering, writing output files, calculting
# parameters and statistics, adding noise and several more are
# available.
#
# Author: Gregor Rajh
# Year: 2024
# Python version: 3
# Dependencies:
#    * Numpy
# github: http://github.com/grajh/nordicman
# e-mail: rajhgregor@gmail.com
#
# Changes:
#   2024.06.20
#    fetch_trexcerpt
#    * implemented station parameter defined by actual picks
#    * changed the default database and added "db_in" as a parameter
#    * removed deprecated archive path "/mnt/szm/{ev_year}/snrs/snrs_{ev_year}"
#**********************************************************************


import copy
import math
import os
import re
import subprocess as sp
import time
from collections import Counter
from datetime import datetime, timedelta
from functools import reduce

import numpy as np

if __name__ == "__main__":
    import deglen as dl
else:
    import src.deglen as dl


class Event_Nordic(object):
    """Event class."""

    # Line type 1, hypocenter line, also written without line-type
    # number, if "" and first line.
    # Skip line 2, which is a line containing macroseismic information,
    # it is indicated.
    # Line type 3, comment line, it is indicated.
    # Line type 7, after this line are the phase lines for a particular
    # earthquake, index this line, then write phase lines.
    # Type 4 lines are the phase lines, type for these lines is usually
    # left blank, if 4 or "" and not first.

    # Templates to copy.
    # int ('', , , 'int')
    # float ('', , , 'float')
    # string ('', , , 'str')
    # string no spacing ('', , , 'str_nsp')

     # Define first line structure (first line of type 1).
    line_first_params = [('YEAR', 1, 5, 'int'), ('MONTH', 6, 8, 'int'),
        ('DAY', 8, 10, 'int'), ('HOUR', 11, 13, 'int'),
        ('MIN', 13, 15, 'int'), ('SEC', 16, 20, 'float'),
        ('Locm_ind', 20, 21, 'str'), ('Dist_ind', 21, 22, 'str'),
        ('Event_type', 22, 23, 'str'), ('LAT', 23, 30, 'float'),
        ('LON', 30, 38, 'float'), ('DEPTH', 38, 43, 'float'),
        ('Depth_ind', 43, 44, 'str'), ('Loc_ind', 44, 45, 'str'),
        ('Hyp_rep_agn', 45, 48, 'str'), ('NST', 48, 51, 'int'),
        ('RMS_t_ress', 51, 55, 'float'), ('MAG1', 55, 59, 'float'),
        ('MAG1_type', 59, 60, 'str'),('MAG1_rep_agn', 60, 63, 'str'),
        ('MAG2', 63, 67, 'float'), ('MAG2_type', 67, 68, 'str'),
        ('MAG2_rep_agn', 68, 71, 'str'), ('MAG3', 71, 75, 'float'),
        ('MAG3_type', 75, 76, 'str'), ('MAG3_rep_agn', 76, 79, 'str')]

    # Define line type 1 structure (other lines of type 1).
    line_one_params = [('YEAR_alt', 1, 5, 'int'), ('MONTH_alt', 6, 8, 'int'),
        ('DAY_alt', 8, 10, 'int'), ('HOUR_alt', 11, 13, 'int'),
        ('MIN_alt', 13, 15, 'int'), ('SEC_alt', 16, 20, 'float'),
        ('Locm_ind_alt', 20, 21, 'str'), ('Dist_ind_alt', 21, 22, 'str'),
        ('Event_type_alt', 22, 23, 'str'), ('LAT_alt', 23, 30, 'float'),
        ('LON_alt', 30, 38, 'float'), ('DEPTH_alt', 38, 43, 'float'),
        ('Depth_ind_alt', 43, 44, 'str'), ('Loc_ind_alt', 44, 45, 'str'),
        ('Hyp_rep_agn_alt', 45, 48, 'str'), ('NST_alt', 48, 51, 'int'),
        ('RMS_t_ress_alt', 51, 55, 'float'), ('MAG1_alt', 55, 59, 'float'),
        ('MAG1_type_alt', 59, 60, 'str'), ('MAG1_rep_agn_alt', 60, 63, 'str'),
        ('MAG2_alt', 63, 67, 'float'), ('MAG2_type_alt', 67, 68, 'str'),
        ('MAG2_rep_agn_alt', 68, 71, 'str'), ('MAG3_alt', 71, 75, 'float'),
        ('MAG3_type_alt', 75, 76, 'str'), ('MAG3_rep_agn_alt', 76, 79, 'str')]

    # Define line type E structure. Hypocenter error estimates.
    line_e_params = [('GAP', 5, 8, 'int'), ('Orig_t_err', 14, 20, 'float'),
        ('Lat_err', 24, 30, 'float'), ('Lon_err', 32, 38, 'float'),
        ('Depth_err', 38, 43,'float'), ('Cov_xy', 43, 55,'float'),
        ('Cov_xz', 55, 67,'float'), ('Cov_yz', 67, 79,'float')]

    # Define line type H structure. High accuracy hypocenter line.
    line_h_params = [('YEAR', 1, 5, 'int'), ('MONTH', 6, 8, 'int'),
        ('DAY', 8, 10, 'int'), ('HOUR', 11, 13, 'int'),
        ('MIN', 13, 15, 'int'), ('SEC', 16, 22, 'float'),
        ('LAT', 23, 32, 'float'), ('LON', 33, 43, 'float'),
        ('DEPTH', 44, 52, 'float'), ('RMS_t_ress', 53, 59, 'float')]

    # Define line type I structure. Action, operator, status and ID.
    line_i_params = [('Last_act', 8, 11, 'str'),
        ('Last_act_dt', 12, 26, 'str'), ('Operator', 30, 34, 'str'),
        ('Status', 42, 56,'str'), ('oID', 60, 74,'str'),
        ('New_file_ID', 74, 75,'str'), ('ID_lock', 75, 76,'str')]

    # Define phase line structure (line type 4).
    line_phase_params = [('Stat_name', 1, 6, 'str_nsp'),
        ('Inst_type', 6, 7, 'str_nsp'), ('Component', 7, 8, 'str_nsp'),
        ('Q_ind', 9, 10, 'str_nsp'), ('Phase_ID', 10, 14, 'str_nsp'),
        ('Uncert_ind', 14, 15, 'int'), ('Pick_type', 15, 16, 'str_nsp'),
        ('First_mot', 16, 17, 'str_nsp'), ('Pick_hour', 18, 20, 'int'),
        ('Pick_minutes', 20, 22, 'int'), ('Pick_seconds', 22, 28, 'float'),
        ('CODA_dur', 29, 33, 'str_nsp'), ('Amplitude', 33, 40, 'float'),
        ('Period', 41, 45, 'float'), ('Azimuth_stat', 46, 51, 'float'),
        ('Phase_veloc', 52, 56, 'float'), ('Angle_incid', 56, 60, 'float'),
        ('Azimuth_res', 60, 63, 'float'), ('Travel_t_res', 63, 68, 'float'),
        ('Weight', 68, 70, 'int'), ('Epic_dist', 70, 75, 'float'),
        ('Azimuth_src', 76, 79, 'float')]

    # Define phase line structure (line type 4). Long phase name.
    line_phase_params_alt = [('Stat_name', 1, 6, 'str_nsp'),
        ('Inst_type', 6, 7, 'str_nsp'), ('Component', 7, 8, 'str_nsp'),
        ('Uncert_ind', 8, 9, 'int'), ('Q_ind', 9, 10, 'str_nsp'),
        ('Phase_ID', 10, 18, 'str_nsp'), ('Pick_hour', 18, 20, 'int'),
        ('Pick_minutes', 20, 22, 'int'), ('Pick_seconds', 22, 28, 'float'),
        ('CODA_dur', 29, 33, 'str_nsp'), ('Amplitude', 33, 40, 'float'),
        ('Period', 41, 45, 'float'), ('Azimuth_stat', 46, 51, 'float'),
        ('Phase_veloc', 52, 56, 'float'), ('Angle_incid', 56, 60, 'float'),
        ('Azimuth_res', 60, 63, 'float'), ('Travel_t_res', 63, 68, 'float'),
        ('Weight', 68, 70, 'int'), ('Epic_dist', 70, 75, 'float'),
        ('Azimuth_src', 76, 79, 'float')]

    def __init__(self, event_block, catalog_name=None):
        self.ID = None
        self.event_block = event_block
        self.Nordic = True
        self.catalog_name = catalog_name
        self.first_line = {}
        self.one_lines = {}
        self.e_line = {}
        self.h_line = {}
        self.i_line = {}
        self.phase_lines = {}
        # Backward indices lists of removed phase lines. In order of
        # filtering.
        self.filt_bi_phl = []

        for param in self.line_first_params:
            param_name = param[0]
            self.first_line[param_name] = None

        for param in self.line_one_params:
            param_name = param[0]
            self.one_lines[param_name] = []

        for param in self.line_e_params:
            param_name = param[0]
            self.e_line[param_name] = None

        for param in self.line_h_params:
            param_name = param[0]
            self.h_line[param_name] = None

        for param in self.line_i_params:
            param_name = param[0]
            self.i_line[param_name] = None

        for param in self.line_phase_params:
            param_name = param[0]
            self.phase_lines[param_name] = []

    # Define some methods common to all line types.
    # Formatting functions. Return None if empty string.
    def _return_int(self, input_string):
        try:
            return int(input_string.replace(' ', ''))
        except ValueError:
            if input_string.replace(' ', ''):
                print('\nFormatting or value error in event block.')
                print('Input string: {}.'.format(input_string))
                print('Corresponding first line.')
                print(self.event_block[0])
                return None
            else:
                return None

    def _return_float(self, input_string):
        try:
            return float(input_string.replace(' ', ''))
        except ValueError:
            if input_string.replace(' ', ''):
                print('\nFormatting or value error in event block.')
                print('Input string: {}.'.format(input_string))
                print('Corresponding first line.')
                print(self.event_block[0])
                return None
            else:
                return None

    def _return_str(self, input_string):
        if input_string.replace(' ', '') == '':
            return None
        else:
            return input_string

    def _return_str_nsp(self, input_string):
        if input_string.replace(' ', '') == '':
            return None
        else:
            return input_string.replace(' ', '')
    
    def _parse_to_dict(self, pars_line, params_list, params_dict):
        """Parameter parsing function."""

        for param in params_list:
            param_name = param[0]
            str_start = param[1]
            str_end = param[2]
            str_type = param[3]

            if str_type == "int":
                params_dict[param_name] = \
                self._return_int(pars_line[str_start:str_end])
            elif str_type == "float":
                params_dict[param_name] = \
                self._return_float(pars_line[str_start:str_end])
            elif str_type == "str":
                params_dict[param_name] = \
                self._return_str(pars_line[str_start:str_end])
            elif str_type == "str_nsp":
                params_dict[param_name] = \
                self._return_str_nsp(pars_line[str_start:str_end])
            else:
                continue
    
    def _parse_to_dictl(self, pars_line, params_list, params_dict):
        """Parameter parsing function for multiple lines of the same
        type."""

        for param in params_list:
            param_name = param[0]
            str_start = param[1]
            str_end = param[2]
            str_type = param[3]

            if str_type == "int":
                try:
                    params_dict[param_name].append(
                        self._return_int(pars_line[str_start:str_end])
                        )
                except ValueError:
                    params_dict[param_name].append(
                        self._return_str(pars_line[str_start:str_end])
                        )
            elif str_type == "float":
                params_dict[param_name].append(
                    self._return_float(pars_line[str_start:str_end])
                    )
            elif str_type == "str":
                params_dict[param_name].append(
                    self._return_str(pars_line[str_start:str_end])
                    )
            elif str_type == "str_nsp":
                params_dict[param_name].append(
                    self._return_str_nsp(pars_line[str_start:str_end])
                    )
            else:
                continue

    def _parse_params(self):
        """Extract and format line based on its type."""

        hl = " STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO AIN AR TRES W  DIS CAZ"
        # remi = []
        hli = 999999

        for li, line in enumerate(self.event_block):
            # li = self.event_block.index(line)

            if (line[-1] == " " or line[-1] == "1") and li == 0:
                self._parse_to_dict(line, self.line_first_params,
                    self.first_line)
            elif line[-1] == "1" and li != 0:
                self._parse_to_dictl(line, self.line_one_params,
                    self.one_lines)
            elif line[-1] =="E":
                self._parse_to_dict(line, self.line_e_params, self.e_line)
            elif line[-1] =="H":
                self._parse_to_dict(line, self.line_h_params, self.h_line)
            elif line[-1] =="I":
                self._parse_to_dict(line, self.line_i_params, self.i_line)
            elif (line[-1] == " " and li != 0) or line[-1] == "4":
                if hli == 999999:
                    hli = li
                else:
                    pass

                if line[8].strip() and line[10:14].strip():
                    self._parse_to_dictl(line, self.line_phase_params_alt,
                        self.phase_lines)
                    self.phase_lines['Pick_type'].append(None)
                    self.phase_lines['First_mot'].append(None)
                else:
                    self._parse_to_dictl(line, self.line_phase_params,
                        self.phase_lines)
            elif line[:-1] == hl:
                hli = li
            else:
                # Line not recognised action.
                if li > hli and hli != 999999:
                    line = line[:-1] + " "
                    self.event_block[li] = line

                    if line[8].strip() and line[10:14].strip():
                        self._parse_to_dictl(line, self.line_phase_params_alt,
                            self.phase_lines)
                        self.phase_lines['Pick_type'].append(None)
                        self.phase_lines['First_mot'].append(None)
                    else:
                        self._parse_to_dictl(line, self.line_phase_params,
                            self.phase_lines)

                else:
                    pass
                    # remi.append(li)

        # Remove unrecognised lines for consistency.
        # if remi:
        #     for i in sorted(remi, reverse=True):
        #         del self.event_block[i]
        # else:
        #     pass

        year = int(self.first_line['YEAR'])
        month = int(self.first_line['MONTH'])
        day = int(self.first_line['DAY'])
        hour = int(self.first_line['HOUR'])
        minute = int(self.first_line['MIN'])
        sec = round(float(self.first_line['SEC']), 1)

        if self.i_line['oID']:
            self.ID = self.i_line['oID']
        else:
            event_ID = "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(
                year, month, day, hour, minute, int(sec)
            )
            if self.i_line:
                self.i_line['oID'] = event_ID

            self.ID = event_ID

            print('NO ID PRESENT. Set to {}.'.format(event_ID))

        if not self.h_line['YEAR']:
            self.h_line['YEAR'] = year
            self.h_line['MONTH'] = month
            self.h_line['DAY'] = day
            self.h_line['HOUR'] = hour
            self.h_line['MIN'] = minute
            self.h_line['SEC'] = sec
            self.h_line['LAT'] = self.first_line['LAT']
            self.h_line['LON'] = self.first_line['LON']
            self.h_line['DEPTH'] = self.first_line['DEPTH']
            self.h_line['RMS_t_ress'] = self.first_line['RMS_t_ress']
        else:
            pass

        h_year = self.h_line['YEAR']
        h_month = self.h_line['MONTH']
        h_day = self.h_line['DAY']
        h_hour = self.h_line['HOUR']
        h_min = self.h_line['MIN']
        h_sec = self.h_line['SEC']

        hms, hs = math.modf(h_sec)
        hs = int(round(hs))
        hms = int(round(hms * 10**6))

        if h_hour == 24:
            h_hour = 0
            tde = timedelta(days=1)
        else:
            tde = timedelta()

        if h_min == 60:
            h_min = 0
            tde = tde + timedelta(seconds=3600)
        else:
            pass

        if h_sec == 60.0:
            hs = 0
            tde = tde + timedelta(seconds=60)
        else:
            pass

        self.tev = datetime(h_year, h_month, h_day, h_hour, h_min, hs, hms) + tde

        self.phn = len(self.phase_lines['Stat_name'])
        epic_dists = list(filter(None, self.phase_lines['Epic_dist']))
        depth = self.h_line['DEPTH']

        if epic_dists and depth:
            min_dist = min(epic_dists)
            self.min_epic = min_dist
            self.dh_ratio = round(min_dist / float(depth), 2)
        else:
            self.min_epic = None
            self.dh_ratio = None

        phases = ['P', 'S']
        self.phc = {}

        ph_phases = self.phase_lines['Phase_ID']

        self.phase_lines['tph'] = []
        self.phase_lines['tt'] = []

        for phase in phases:
            self.phc[phase] = 0
            for iph, ph_phase in enumerate(ph_phases):
                ph_phase = str(ph_phase).upper()

                if ph_phase[0] in ['Q', '(']:
                    ph_phase = ph_phase[1:]
                else:
                    pass

                if phase == ph_phase[0]:
                    self.phc[phase] += 1
                else:
                    pass

        for iph, ph_phase in enumerate(ph_phases):
            ph_station = self.phase_lines['Stat_name'][iph]
            ph_ID = self.phase_lines['Phase_ID'][iph]
            ph_year = h_year
            ph_month = h_month
            ph_day = h_day
            ph_hour = self.phase_lines['Pick_hour'][iph]
            ph_min = self.phase_lines['Pick_minutes'][iph]
            ph_sec = self.phase_lines['Pick_seconds'][iph]

            if not ph_hour:
                ph_hour = 0
                tdp = timedelta()
            elif 23 < ph_hour < 48:
                ph_hour -= 24
                tdp = timedelta(days=1)
            elif ph_hour == 48:
                ph_hour = 0
                tdp = timedelta(days=1)
            else:
                tdp = timedelta()

            if not ph_min:
                ph_min = 0
            elif ph_min == 60:
                ph_min = 0
                tdp = tdp + timedelta(seconds=3600)
            else:
                pass

            if not ph_sec:
                ph_sec = 0
            elif ph_sec == 60.0:
                ph_sec = 0
                tdp = tdp + timedelta(seconds=60)
            else:
                pass

            ph_msec, ph_sec = math.modf(ph_sec)
            ph_sec = int(round(ph_sec))
            ph_msec = int(round(ph_msec * 10**6))

            try:
                tph = datetime(ph_year, ph_month, ph_day, ph_hour, ph_min,
                    ph_sec, ph_msec) + tdp
                self.phase_lines['tph'].append(tph)
                self.phase_lines['tt'].append(tph - self.tev)
            except TypeError:
                print('\nError in arrival time detected inside bulletin.')
                print('\nEvent data:')
                print('{}\n{}\n{} {} {} {} {} {} {} {}\n'.format(
                    self.ID, ph_station, ph_year, ph_month, ph_day,
                    ph_hour, ph_min, ph_sec, ph_msec, tdp))
                self.phase_lines['tph'].append(None)
                continue

    def reload_params(self):
        self.first_line = {}
        self.one_lines = {}
        self.e_line = {}
        self.h_line = {}
        self.i_line = {}
        self.phase_lines = {}

        for param in self.line_first_params:
            param_name = param[0]
            self.first_line[param_name] = None

        for param in self.line_one_params:
            param_name = param[0]
            self.one_lines[param_name] = []

        for param in self.line_e_params:
            param_name = param[0]
            self.e_line[param_name] = None

        for param in self.line_h_params:
            param_name = param[0]
            self.h_line[param_name] = None

        for param in self.line_i_params:
            param_name = param[0]
            self.i_line[param_name] = None

        for param in self.line_phase_params:
            param_name = param[0]
            self.phase_lines[param_name] = []

        self._parse_params()

    def filter_nordic_phases(self):
    # Filter phase lines based on backward indices. For filtered Nordic
    # output.
        self.event_block_f = copy.copy(self.event_block)

        if self.filt_bi_phl:
            for bi_phl in self.filt_bi_phl:
                if bi_phl:
                    for bi in bi_phl:
                        del self.event_block_f[bi]
                else:
                    pass
        else:
            pass

    def count_phases(self):
        self.first_line['PHN'] = len(self.phase_lines['Stat_name'])
        self.phn = len(self.phase_lines['Stat_name'])

        qis = [0, 1, 2, 3, 4, 9]

        for qi in qis:
            phn = len(list(filter(
                lambda x: x == qi, self.phase_lines['Uncert_ind'])))
            dict_str = 'PHN' + str(qi)
            self.first_line[dict_str] = phn

        phases = ['P', 'S', 'SP']
        self.phc = {}
        self.phcu = {}

        ph_phases = self.phase_lines['Phase_ID']
        ph_uncs = self.phase_lines['Uncert_ind']

        for phase in phases:
            self.phc[phase] = 0
            self.phcu[phase] = {}

            for qi in qis:
                self.phcu[phase][qi] = 0

            for i, ph_phase in enumerate(ph_phases):
                ph_phase = str(ph_phase).upper()

                if ph_phase[0] in ['Q', '(']:
                    ph_phase = ph_phase[1:]
                else:
                    pass

                if ph_phase[0] == phase:
                    self.phc[phase] += 1
                    unc = ph_uncs[i]

                    if unc is not None and unc in qis:
                        self.phcu[phase][unc] += 1
                    else:
                        pass

                else:
                    pass

    def count_stations(self):
        self.nst = len(set(self.phase_lines['Stat_name']))
        self.first_line['NST'] = self.nst

    def calc_rmsres(self, velest=False, overwrite=True, sweight=1.0, iqf=1):
        if velest:
            phs = self.phase_lines['Phase_ID']
            uncs = np.array(self.phase_lines['Uncert_ind'])

            if iqf == 1:
                weights_in = 1 / 2**uncs
            elif iqf == 0:
                weights_in = 1 / 2**(uncs * 2)
            else:
                weights_in = [1.0] * len(uncs)

            for iph, ph in enumerate(phs):
                if ph.upper() == 'S':
                    weights_in[iph] *= sweight
                else:
                    pass

        else:
            weights_in = self.phase_lines['Weight']

        residuals_in = self.phase_lines['Travel_t_res']

        weights = []
        residuals = []

        # Remove readings without weight and residual, and zero weight. 
        for i, w in enumerate(weights_in):
            resi = residuals_in[i]

            if w is not None and resi is not None:
                if w > 0.0:
                    weights.append(w)
                    residuals.append(resi)
            else:
                pass

        weights = np.array(weights)
        residuals = np.array(residuals)

        nobs = float(len(weights))
        wsum = float(sum(weights))

        if nobs > 1:
            # wns = list(map(lambda x: x * nobs / wsum), weights)
            wns = weights * nobs / wsum
            wres = sum(wns * residuals)
            sqrres = sum(wns**2 * residuals**2)
            self.rmsres = ((sqrres - (wres**2 / nobs)) / (nobs - 1))**0.5

            if overwrite:
                self.first_line['RMS_t_ress'] = round(self.rmsres, 1)
                self.h_line['RMS_t_ress'] = round(self.rmsres, 3)
            else:
                pass

            self.avres = wres / nobs
            self.wres = wres
            self.sqrres = sqrres
            self.nobsnot0 = nobs
        else:
            self.rmsres = None
            self.avres = None
            self.wres = None
            self.sqrres = None
            self.nobsnot0 = None

        # print(self.rmsres)

    def calc_gap(self, stas_obj, overwrite=True, use_epicd=False):
        obs_stas = self.phase_lines['Stat_name']
        # epic = np.array((self.h_line['LAT'], self.h_line['LON']))
        epic = np.array((self.h_line['LON'], self.h_line['LAT']))

        if epic.all():
            # lat_len, lon_len = dl.deg_length(epic)
            lat_len, lon_len = dl.deg_length((epic[1], epic[0]))

            if use_epicd:
                epic_dists = self.phase_lines['Epic_dist']

            sta_azimuths = []

            for i, sta in enumerate(obs_stas):
                # sta_lat = stas_obj.param_dict[sta]['LAT']
                # sta_lon = stas_obj.param_dict[sta]['LON']
                # sta_vect = np.array((sta_lat, sta_lon)) - epic
                # sta_vect[0] *= lat_len
                # sta_vect[1] *= lon_len

                sta_lon = stas_obj.param_dict[sta]['LON']
                sta_lat = stas_obj.param_dict[sta]['LAT']
                sta_vect = np.array((sta_lon, sta_lat)) - epic
                sta_vect[0] *= lon_len
                sta_vect[1] *= lat_len

                if use_epicd:
                    vlen = epic_dists[i]
                else:
                    vlen = sum(sta_vect**2)**0.5

                sta_nvect = sta_vect / vlen

                sta_azimuth = np.degrees(np.arctan2(*sta_nvect))

                # Azimuth is measured in CCW direction.
                if sta_azimuth < 0:
                    # sta_azimuth = 360.0 + sta_azimuth
                    sta_azimuth = sta_azimuth * (-1)
                else:
                    sta_azimuth = 360.0 - sta_azimuth
                    pass

                sta_azimuths.append(sta_azimuth)

            # print(obs_stas)
            # print(sta_azimuths)
            sta_azimuths = sorted(sta_azimuths)

            ista_angles = []

            for i, az in enumerate(sta_azimuths, 1):
                if i == len(sta_azimuths):
                    azi = 360.0 + sta_azimuths[0]
                else:
                    azi = sta_azimuths[i]
                
                ista_angle = abs(azi - az)
                ista_angles.append(ista_angle)

            # print(sorted(ista_angles))
            # print(max(ista_angles))

            gap = max(ista_angles)

            if overwrite:
                self.e_line['GAP'] = int(round(gap))
            else:
                pass
        else:
            pass

    def calc_dh(self):
        epic_dists = list(filter(None, self.phase_lines['Epic_dist']))
        depth = self.h_line['DEPTH']

        if epic_dists and depth:
            min_dist = min(epic_dists)
            self.min_epic = min_dist
            self.dh_ratio = round(min_dist / float(depth), 2)
        elif depth == 0:
            self.min_epic = 0.0
            self.dh_ratio = 9999.0
        else:
            self.min_epic = None
            self.dh_ratio = None

    def return_stations(self):
        return self.phase_lines['Stat_name']

    def assign_picks(self, picks_obj):
        self.phase_lines['Uncert_time'] = []
        self.phase_lines['Channel'] = []
        self.phase_lines['pick_obj'] = []
        stas = self.phase_lines['Stat_name']

        pick_list = sorted(picks_obj.picks[:], key=lambda x: x.tpick,
            reverse=True)

        for iph, ph_sta in enumerate(stas):
            tph = self.phase_lines['tph'][iph]
            ph_ID = self.phase_lines['Phase_ID'][iph]
            # ph_name = re.sub('[^P|S]', '', ph_ID)

            sel_pick = None

            for pick in pick_list:
                tpick = pick.tpick_nord
                pick_sta = pick.statID
                pick_ph = pick.phase
                if tph == tpick and ph_sta == pick_sta and ph_ID == pick_ph:
                    sel_pick = pick
                    break
                else:
                    pass

            if sel_pick:
                self.phase_lines['Uncert_time'].append(pick.unctime)
                self.phase_lines['Channel'].append(pick.channel)
                self.phase_lines['pick_obj'].append(pick)
                pick_list.remove(sel_pick)
            else:
                self.phase_lines['Uncert_time'].append(None)
                self.phase_lines['Channel'].append(None)
                self.phase_lines['pick_obj'].append(None)

    def reclassify(self, classes={0.03:0, 0.06:1, 0.10:2, 0.20:3, 0.50:4}):
        self.phase_lines['Uncert_reclass'] = []
        pick_objs = self.phase_lines['pick_obj']
        classts = sorted(classes)

        for pick_obj in pick_objs:
            if pick_obj:
                pick_unct = pick_obj.unctime
                if pick_unct <= classts[0]:
                    pick_cls = 0
                elif classts[0] < pick_unct <= classts[1]:
                    pick_cls = 1
                elif classts[1] < pick_unct <= classts[2]:
                    pick_cls = 2
                elif classts[2] < pick_unct <= classts[3]:
                    pick_cls = 3
                elif classts[3] < pick_unct <= classts[4]:
                    pick_cls = 4
                else:
                    pick_cls = 5
            else:
                pick_cls = 5

            self.phase_lines['Uncert_reclass'].append(pick_cls)


class EventDatabase(object):

    def __init__(self):
        self.events = []
        self.event_flist = []
        self.event_flist_alt = []

    def add_event(self, event):
        self.events.append(event)

    def add_events(self, event_list):
        self.events += event_list

    def reload_database(self):
        for event in self.events:
            event.reload_params()

    # Database filtering functions.
    def filter_sval(self, list_in, line_type_str, param_str, operator_str,
        filter_value=None, line_type_str_alt=None, param_str_alt=None,
        use_deepcopy=False, out_as_list=False):
        """Filtering is based on a single value for one parameter.
        Possible line type strings are "first_line" and "e_line",
        "h_line" and "i_line".
        Optionally use deepcopy on 'list_in' to make sure output
        list goes directly to 'self.event_flist_alt'. This way,
        'self.event_flist' stays unaltered."""

        list_out = []
        list_out_alt = []

        if use_deepcopy:
            list_in = copy.deepcopy(list_in)
        else:
            pass

        for event in list_in:
            if line_type_str:
                params = getattr(event, line_type_str)
                param_value = params[param_str]
            else:
                param_value = getattr(event, param_str)

            if param_value is not None:
                if operator_str == ">":
                    evaluation = param_value > filter_value
                elif operator_str == "<":
                    evaluation = param_value < filter_value
                elif operator_str == ">=":
                    evaluation = param_value >= filter_value
                elif operator_str == "<=":
                    evaluation = param_value <= filter_value
                elif operator_str == "==":
                    evaluation = param_value == filter_value
                elif operator_str == "!=":
                    evaluation = param_value != filter_value
                elif operator_str == "in":
                    evaluation = param_value in filter_value
                elif operator_str == "not in":
                    evaluation = param_value not in filter_value
                else:
                    print('Function arguments not given properly.')

                if not param_str_alt:
                    if evaluation:
                        list_out.append(event)
                    else:
                        continue
                else:
                    params_alt = getattr(event, line_type_str_alt)
                    param_value_alt = params_alt[param_str_alt]

                    if evaluation:
                        list_out.append(event)
                    elif not evaluation and param_value_alt is not None:
                        list_out_alt.append(event)
                    elif param_value_alt is not None:
                        list_out_alt.append(event)
                    else:
                        continue
            else:
                continue

        if out_as_list:
            return list_out
        else:
            if list_in == self.event_flist or not self.event_flist:
                if not param_str_alt:
                    self.event_flist = list_out
                else:
                    self.event_flist = list_out
                    self.event_flist_alt = list_out_alt
            else:
                if not param_str_alt:
                    self.event_flist_alt = list_out
                else:
                    self.event_flist = list_out
                    self.event_flist_alt = list_out_alt

    def filter_mval(self, list_in, line_type_str, param_str, operator_str,
        filter_value, bool_method='any', use_deepcopy=False,
        out_as_list=False):
        """Filtering is based on multiple values for one parameter.
        List of parameter values from multiple lines of same type
        for the same event. Each element in the list is evaluated and
        finally a list of boolean values is evaluated with "any" or
        "all" function. Removes events from list. Possible line type
        strings are "one_lines" and "phase_lines". Optionally use
        deepcopy on 'list_in' to make sure output list goes directly
        to 'self.event_flist_alt'. This way, 'self.event_flist' stays
        unaltered."""

        list_out = []

        if use_deepcopy:
            list_in = copy.deepcopy(list_in)
        else:
            pass

        for event in list_in:
            params = getattr(event, line_type_str)
            param_values = params[param_str]
            param_values = list(filter(None, param_values))

            if param_values:
                if operator_str == ">":
                    evaluation_list = list(map(
                        lambda x: x > filter_value, param_values))
                elif operator_str == "<":
                    evaluation_list = list(map(
                        lambda x: x < filter_value, param_values))
                elif operator_str == ">=":
                    evaluation_list = list(map(
                        lambda x: x >= filter_value, param_values))
                elif operator_str == "<=":
                    evaluation_list = list(map(
                        lambda x: x <= filter_value, param_values))
                elif operator_str == "==":
                    evaluation_list = list(map(
                        lambda x: x == filter_value, param_values))
                elif operator_str == "!=":
                    evaluation_list = list(map(
                        lambda x: x != filter_value, param_values))
                elif operator_str == "in":
                    evaluation_list = [filter_value in param_values]
                elif operator_str == "not in":
                    evaluation_list = [filter_value in param_values]
                else:
                    raise ValueError(
                    'Function arguments not given properly ("operator_str").'
                        )
                
                if bool_method == 'any':
                    if any(evaluation_list):
                        list_out.append(event)
                    else:
                        continue
                elif bool_method == 'all':
                    if all(evaluation_list):
                        list_out.append(event)
                    else:
                        continue
                else:
                    raise ValueError(
                    'Function arguments not given properly ("bool_method").'
                        )
            else:
                continue

        if out_as_list:
            return list_out
        else:
            if list_in == self.event_flist or not self.event_flist:
                self.event_flist = list_out
            else:
                self.event_flist_alt = list_out

    def filter_mval_proc(self, list_in, line_type_str, param_str,
        operator_str, filter_value=None, list_proc='auto', index=0,
        slice_start=0, slice_end=1, use_deepcopy=False, out_as_list=False):
        """Filtering is based on multiple values for one parameter.
        List of parameter values from multiple lines of same type
        for the same event. Removes events from list. Possible line
        type strings are "one_lines" and "phase_lines". Optionally use
        deepcopy on 'list_in' to make sure output list goes directly
        to 'self.event_flist_alt'. This way, 'self.event_flist' stays
        unaltered."""

        list_out = []

        if use_deepcopy:
            list_in = copy.deepcopy(list_in)
        else:
            pass

        for event in list_in:
            params = getattr(event, line_type_str)
            param_list = params[param_str]
            param_list = list(filter(None, param_list))

            if param_list:
                if list_proc == "auto":
                    if len(param_list) == 1:
                        param_value = param_list[0]
                    else:
                        param_value = sum(param_list) / len(param_list)
                elif list_proc == "index":
                    param_value = param_list[index]
                elif list_proc == "mean":
                    param_value = sum(param_list) / len(param_list)
                elif list_proc == "max":
                    param_value = max(param_list)
                elif list_proc == "min":
                    param_value = min(param_list)
                elif list_proc == "slice_list":
                    param_value = param_list[slice_start:slice_end]
                else:
                    print('Function arguments not given properly.')
                    break
            else:
                continue
            
            if operator_str == ">":
                evaluation = param_value > filter_value
            elif operator_str == "<":
                evaluation = param_value < filter_value
            elif operator_str == ">=":
                evaluation = param_value >= filter_value
            elif operator_str == "<=":
                evaluation = param_value <= filter_value
            elif operator_str == "==":
                evaluation = param_value == filter_value
            elif operator_str == "!=":
                evaluation = param_value != filter_value
            elif operator_str == "empty":
                if param_list:
                    evaluation = True
                else:
                    evaluation = False
            else:
                print('Function arguments not given properly.')

            if evaluation and param_value is not None:
                list_out.append(event)
            else:
                continue

        if out_as_list:
            return list_out
        else:
            if list_in == self.event_flist or not self.event_flist:
                self.event_flist = list_out
            else:
                self.event_flist_alt = list_out

    def filter_mlin(self, list_in, line_type_str, param_str, operator_str,
        filter_value=None, el_proc=None, slice_start=None, slice_end=None,
        use_deepcopy=False, out_as_list=False, write_back_ind=True,
        min_linenum=1):
        """Filtering of multiple lines for one parameter. List of
        parameter values from multiple lines of same type for same
        event. Removes lines from event. Possible line type strings
        are 'one_lines' and 'phase_lines'. Optionally use deepcopy
        on 'list_in' to make sure output list goes directly to
        'self.event_flist_alt'. This way, 'self.event_flist' stays
        unaltered."""

        list_out = []

        if use_deepcopy:
            list_in = copy.deepcopy(list_in)
        else:
            pass

        # if el_proc != None:
        #     slice_end = slice_start + len(str(filter_value))
        # else:
        #     pass

        for event in list_in:
            indices_list = []
            params = getattr(event, line_type_str)
            param_list = params[param_str]
            pll = len(param_list)

            if param_list:
                for i, param_value in enumerate(param_list):
                    if param_value is not None:
                        if el_proc == "slice":
                            param_value = str(
                                param_value)[slice_start:slice_end]
                        elif el_proc == "abs":
                            param_value = abs(param_value)
                        else:
                            pass
                        if operator_str == ">":
                            evaluation = param_value > filter_value
                        elif operator_str == "<":
                            evaluation = param_value < filter_value
                        elif operator_str == ">=":
                            evaluation = param_value >= filter_value
                        elif operator_str == "<=":
                            evaluation = param_value <= filter_value
                        elif operator_str == "==":
                            evaluation = param_value == filter_value
                        elif operator_str == "!=":
                            evaluation = param_value != filter_value
                        elif operator_str == "in":
                            evaluation = param_value in filter_value
                        elif operator_str == "not in":
                            evaluation = param_value not in filter_value
                        elif operator_str == "max":
                            evaluation = param_value == max(param_list)
                        elif operator_str == "min":
                            evaluation = param_value == min(param_list)
                        elif operator_str == "empty":
                            if param_list:
                                evaluation = True
                            else:
                                evaluation = False
                        else:
                            print('Function arguments not given properly.')
                        if not evaluation or param_value is None:
                            indices_list.append(i)
                        else:
                            continue
                    else:
                        indices_list.append(i)

                if write_back_ind and indices_list:
                    bind = list(map(lambda x: -(pll - x),
                        indices_list))
                    event.filt_bi_phl.append(bind)
                else:
                    pass

                for param_name in params:
                    if indices_list:
                        for i in sorted(indices_list, reverse=True):
                            del params[param_name][i]
                    else:
                        pass
                if len(params[param_str]) >= min_linenum:
                    list_out.append(event)
                else:
                    continue
            else:
                continue

        # print(params)

        if out_as_list:
            return list_out
        else:
            if list_in == self.event_flist or not self.event_flist:
                self.event_flist = list_out
            else:
                self.event_flist_alt = list_out

    def keep_first_arrivals(self, list_in, use_deepcopy=True,
        write_back_ind=True, min_phasenum=1):
        """Use before other phase filtering functions or in appropriate
        order. Do not filter for uncertainties beforehand. It can
        happen that the latter arrival has lower uncertainty. When
        using this method after other filtering methods, it is
        recommended to filter first for phase type."""

        fa_list_out = []

        if use_deepcopy:
            list_in = copy.deepcopy(list_in)
        else:
            pass

        for event in list_in:
            sta_list = []
            picki_remove = []
            picki_keep = []

            ph_year = event.first_line['YEAR']
            ph_month = event.first_line['MONTH']
            ph_day = event.first_line['DAY']
            ph_hour = event.first_line['HOUR']
            ph_min = event.first_line['MIN']
            ph_sec = event.first_line['SEC']
            ph_msec, ph_sec = math.modf(ph_sec)
            ph_sec = int(round(ph_sec))
            ph_msec = int(round(ph_msec * 10**6))

            if ph_hour == 24:
                    ph_hour = 0
                    tdp = timedelta(days=1)
            else:
                tdp = timedelta()
            
            if ph_min == 60:
                ph_min = 0
                tdp = tdp + timedelta(seconds=3600)
            else:
                pass
            
            if ph_sec == 60.0:
                ph_sec = 0
                tdp = tdp + timedelta(seconds=60)
            else:
                pass

            tph_keep = datetime(ph_year, ph_month, ph_day, ph_hour, ph_min,
                ph_sec, ph_msec) + tdp

            if len(event.phase_lines['Stat_name']) >= min_phasenum:
                phll = len(event.phase_lines['Stat_name'])

                for picki, sta in enumerate(event.phase_lines['Stat_name']):
                    ph_hour = event.phase_lines['Pick_hour'][picki]
                    ph_min = event.phase_lines['Pick_minutes'][picki]
                    ph_sec = event.phase_lines['Pick_seconds'][picki]

                    if not ph_hour:
                        # print(event.ID)
                        ph_hour = 0
                    else:
                        pass

                    if not ph_min:
                        # print(event.ID)
                        ph_min = 0
                    else:
                        pass

                    if not ph_sec:
                        # print(event.ID)
                        ph_sec = 0.0
                    else:
                        ph_sec = float(
                            event.phase_lines['Pick_seconds'][picki])

                    ph_msec, ph_sec = math.modf(ph_sec)
                    ph_sec = int(round(ph_sec))
                    ph_msec = int(round(ph_msec * 10**6))

                    if ph_hour == 24:
                        ph_hour = 0
                        tdp = timedelta(days=1)
                    else:
                        tdp = timedelta()
                    
                    if ph_min == 60:
                        ph_min = 0
                        tdp = tdp + timedelta(seconds=3600)
                    else:
                        pass
                    
                    if ph_sec == 60.0:
                        ph_sec = 0
                        tdp = tdp + timedelta(seconds=60)
                    else:
                        pass

                    tph = datetime(ph_year, ph_month, ph_day, ph_hour, ph_min,
                        ph_sec, ph_msec) + tdp

                    if sta in sta_list:
                        if tph > tph_keep:
                            picki_remove.append(picki)
                        else:
                            picki_remove.append(picki_keep.pop())
                            picki_keep.append(picki)
                            tph_keep = tph
                    else:
                        sta_list.append(sta)
                        picki_keep.append(picki)
                        tph_keep = tph

                if 0 < len(picki_remove) \
                     < len(event.phase_lines['Stat_name']):
                    if write_back_ind:
                            bind = list(map(lambda x: -(phll - x),
                                picki_remove))
                            event.filt_bi_phl.append(bind)
                    else:
                        pass

                    for par in event.phase_lines:
                        for i in sorted(picki_remove, reverse=True):
                            # USE del!
                            # event.phase_lines[par].pop(i)
                            del event.phase_lines[par][i]

                    fa_list_out.append(event)

                elif not picki_remove:
                    fa_list_out.append(event)

                else:
                    pass
            else:
                pass

        return fa_list_out

    def keep_first_arrivals_c(self, list_in, use_deepcopy=True,
        write_back_ind=True, min_phasenum=1, phases=['P', 'S']):
        """Use before other phase filtering functions or in appropriate
        order. Do not filter for uncertainties beforehand. It can
        happen that the latter arrival has lower uncertainty. When
        using this method after other filtering methods, it is
        recommended to filter first for phase type. Works with multiple
        phase types. Readings should be grouped by station and then
        optionally ordered by time."""

        # Make it so that no ordering is necessary. Sort through tuples
        # containing original indices. Or use a dictionary and 'index'
        # method on a list or dictionary for each station last phase.

        fa_list_out = []

        if use_deepcopy:
            list_in = copy.deepcopy(list_in)
        else:
            pass

        for event in list_in:
            picki_remove = []
            picki_keep = []

            ph_year = event.first_line['YEAR']
            ph_month = event.first_line['MONTH']
            ph_day = event.first_line['DAY']
            ph_hour = event.first_line['HOUR']
            ph_min = event.first_line['MIN']
            ph_sec = event.first_line['SEC']
            ph_msec, ph_sec = math.modf(ph_sec)
            ph_sec = int(round(ph_sec))
            ph_msec = int(round(ph_msec * 10**6))

            if ph_hour == 24:
                    ph_hour = 0
                    tdp = timedelta(days=1)
            else:
                tdp = timedelta()
            
            if ph_min == 60:
                ph_min = 0
                tdp = tdp + timedelta(seconds=3600)
            else:
                pass
            
            if ph_sec == 60.0:
                ph_sec = 0
                tdp = tdp + timedelta(seconds=60)
            else:
                pass

            tph_keep = datetime(ph_year, ph_month, ph_day, ph_hour, ph_min,
                ph_sec, ph_msec) + tdp

            if len(event.phase_lines['Stat_name']) >= min_phasenum:
                phll = len(event.phase_lines['Stat_name'])

                for phase in phases:
                    sta_list = []

                    for picki, sta in enumerate(
                        event.phase_lines['Stat_name']):
                        ph_hour = event.phase_lines['Pick_hour'][picki]
                        ph_min = event.phase_lines['Pick_minutes'][picki]
                        ph_sec = event.phase_lines['Pick_seconds'][picki]
                        ph_phase = event.phase_lines['Phase_ID'][picki]
                        ph_phase = str(ph_phase).upper()

                        if not ph_hour:
                            # print(event.ID)
                            ph_hour = 0
                        else:
                            pass

                        if not ph_min:
                            # print(event.ID)
                            ph_min = 0
                        else:
                            pass

                        if not ph_sec:
                            # print(event.ID)
                            ph_sec = 0.0
                        else:
                            ph_sec = float(
                                event.phase_lines['Pick_seconds'][picki])

                        ph_msec, ph_sec = math.modf(ph_sec)
                        ph_sec = int(round(ph_sec))
                        ph_msec = int(round(ph_msec * 10**6))

                        if ph_hour == 24:
                            ph_hour = 0
                            tdp = timedelta(days=1)
                        else:
                            tdp = timedelta()
                        
                        if ph_min == 60:
                            ph_min = 0
                            tdp = tdp + timedelta(seconds=3600)
                        else:
                            pass
                        
                        if ph_sec == 60.0:
                            ph_sec = 0
                            tdp = tdp + timedelta(seconds=60)
                        else:
                            pass

                        tph = datetime(ph_year, ph_month, ph_day, ph_hour,
                            ph_min, ph_sec, ph_msec) + tdp

                        # Not to remove any of the specified ones.
                        if any(x in ph_phase for x in phases):
                            # We use 'in' because we are interested in
                            # any first arrival of a current phase type
                            # (e.g. P, S). Cannot be used in more
                            # specific way, to search for example for
                            # all Pn first arrivals. One can use this
                            # method to search first for all first
                            # arrivals and then filter the needed
                            # phases out.
                            if phase in ph_phase:
                                if sta in sta_list:
                                    if tph > tph_keep:
                                        picki_remove.append(picki)
                                    else:
                                        picki_remove.append(picki_keep.pop())
                                        picki_keep.append(picki)
                                        tph_keep = tph
                                else:
                                    sta_list.append(sta)
                                    picki_keep.append(picki)
                                    tph_keep = tph
                            else:
                                pass
                        else:
                            picki_remove.append(picki)

                picki_remove = set(picki_remove)

                if 0 < len(picki_remove) \
                     < len(event.phase_lines['Stat_name']):
                    if write_back_ind:
                            bind = list(map(lambda x: -(phll - x),
                                picki_remove))
                            event.filt_bi_phl.append(bind)
                    else:
                        pass

                    for par in event.phase_lines:
                        for i in sorted(picki_remove, reverse=True):
                            # USE del!
                            # event.phase_lines[par].pop(i)
                            del event.phase_lines[par][i]

                    fa_list_out.append(event)

                elif not picki_remove:
                    fa_list_out.append(event)

                else:
                    pass
            else:
                pass

        return fa_list_out

    def keep_first_arrivals_test(self, list_in, use_deepcopy=True,
        write_back_ind=True, min_phasenum=1, phases=['P', 'S']):
        """Use before other phase filtering functions or in appropriate
        order. Do not filter for uncertainties beforehand. It can
        happen that the latter arrival has lower uncertainty. When
        using this method after other filtering methods, it is
        recommended to filter first for phase type. Works with multiple
        phase types."""

        # Make it so that no ordering is necessary. Sort through tuples
        # containing original indices. Or use a dictionary and 'index'
        # method on a list or dictionary for each station last phase.

        fa_list_out = []

        if use_deepcopy:
            list_in = copy.deepcopy(list_in)
        else:
            pass

        for event in list_in:
            picki_remove = []
            pick_dict = {}

            for phase in phases:
                pick_dict[phase] = {}

            ph_year = event.first_line['YEAR']
            ph_month = event.first_line['MONTH']
            ph_day = event.first_line['DAY']
            ph_hour = event.first_line['HOUR']
            ph_min = event.first_line['MIN']
            ph_sec = event.first_line['SEC']
            ph_msec, ph_sec = math.modf(ph_sec)
            ph_sec = int(round(ph_sec))
            ph_msec = int(round(ph_msec * 10**6))

            if ph_hour == 24:
                    ph_hour = 0
                    tdp = timedelta(days=1)
            else:
                tdp = timedelta()
            
            if ph_min == 60:
                ph_min = 0
                tdp = tdp + timedelta(seconds=3600)
            else:
                pass
            
            if ph_sec == 60.0:
                ph_sec = 0
                tdp = tdp + timedelta(seconds=60)
            else:
                pass

            if len(event.phase_lines['Stat_name']) >= min_phasenum:
                phll = len(event.phase_lines['Stat_name'])

                for picki, sta in enumerate(
                    event.phase_lines['Stat_name']):
                    # Filter out lines with None values.
                    ph_hour = event.phase_lines['Pick_hour'][picki]
                    ph_min = event.phase_lines['Pick_minutes'][picki]
                    ph_sec = event.phase_lines['Pick_seconds'][picki]
                    ph_phase = event.phase_lines['Phase_ID'][picki]
                    ph_phase = str(ph_phase).upper()

                    # Modify string of questionable picks.
                    if ph_phase[0] in ['Q', '(']:
                        ph_phase = ph_phase[1:]
                    else:
                        pass

                    if not ph_hour:
                        # print(event.ID)
                        ph_hour = 0
                    else:
                        pass

                    if not ph_min:
                        # print(event.ID)
                        ph_min = 0
                    else:
                        pass

                    if not ph_sec:
                        # print(event.ID)
                        ph_sec = 0.0
                    else:
                        ph_sec = float(
                            event.phase_lines['Pick_seconds'][picki])

                    ph_msec, ph_sec = math.modf(ph_sec)
                    ph_sec = int(round(ph_sec))
                    ph_msec = int(round(ph_msec * 10**6))

                    if ph_hour == 24:
                        ph_hour = 0
                        tdp = timedelta(days=1)
                    else:
                        tdp = timedelta()
                    
                    if ph_min == 60:
                        ph_min = 0
                        tdp = tdp + timedelta(seconds=3600)
                    else:
                        pass
                    
                    if ph_sec == 60.0:
                        ph_sec = 0
                        tdp = tdp + timedelta(seconds=60)
                    else:
                        pass

                    tph = datetime(ph_year, ph_month, ph_day, ph_hour,
                        ph_min, ph_sec, ph_msec) + tdp

                    # Not to remove any of the picks for the specified
                    # phases.
                    if any(x in ph_phase for x in phases):
                        # Loop through specified phases.
                        for phase in phases:
                            # Recognize phase.
                            if phase == ph_phase[0]:
                                if sta in list(pick_dict[phase]):
                                    if tph >= pick_dict[phase][sta]['time']:
                                        picki_remove.append(picki)
                                    else:
                                        picki_remove.append(
                                            pick_dict[phase][sta]['keep'])
                                        pick_dict[phase][sta]['keep'] = picki
                                        pick_dict[phase][sta]['time'] = tph
                                else:
                                    pick_dict[phase][sta] = {}
                                    pick_dict[phase][sta]['keep'] = picki
                                    pick_dict[phase][sta]['time'] = tph
                            else:
                                pass
                    else:
                        picki_remove.append(picki)

                # Only to be on the safe side.
                picki_remove = set(picki_remove)

                if 0 < len(picki_remove) \
                     < len(event.phase_lines['Stat_name']):
                    if write_back_ind:
                            bind = list(map(lambda x: -(phll - x),
                                picki_remove))
                            event.filt_bi_phl.append(bind)
                    else:
                        pass

                    for par in event.phase_lines:
                        for i in sorted(picki_remove, reverse=True):
                            # USE del!
                            # event.phase_lines[par].pop(i)
                            del event.phase_lines[par][i]

                    fa_list_out.append(event)

                elif not picki_remove:
                    fa_list_out.append(event)

                else:
                    pass
            else:
                pass

        return fa_list_out

    def return_plot_vals(self, list_in, line_type_str_list, param_str_list):
        # list_clc = []
        dict_out = {}

        for param_str in param_str_list:
            dict_out[param_str] = []

        # Create list of only those events, which contain values for
        # all given parameters. This makes lists for different
        # parameters consistent in number, which we need for plotting.
        for event in sorted(list_in, key=lambda x: x.ID):
            param_in_count = 0

            for i, line_type in enumerate(line_type_str_list):
                params = getattr(event, line_type)
                param_in = param_str_list[i]

                if param_in in params:
                    if params[param_in] != []:
                        param_in_count += 1
                    else:
                        continue
                else:
                    continue

            if param_in_count == len(param_str_list):
            # list_clc.append(event)
            # Extract values for given parameters.
                for i, line_type in enumerate(line_type_str_list):
                    params = getattr(event, line_type)
                    param_in = param_str_list[i]

                    if line_type in ['one_lines', 'phase_lines']:
                        # WATCH OUT! Takes only first line, when
                        # multiple lines are present.
                        # param_value = params[param_in][0]
                        # dict_out[param_in].append(param_value)
                        # Should work fine like this.
                        param_values = params[param_in]
                        dict_out[param_in].extend(param_values)
                    else:
                        param_value = params[param_in]
                        dict_out[param_in].append(param_value)

            else:
                continue

        # Extract values for given parameters.
        # for event in list_clc:
        #     for line_type in line_type_str_list:
        #         params = getattr(event, line_type)
        #         for param_in in params:
        #             if param_in in param_str_list:
        #                 if line_type == "one_lines" or \
        #                     line_type == "phase_lines":
        #                     # Take only first line, when multiple are present.
        #                     param_value = params[param_in][0]
        #                     dict_out[param_in].append(param_value)
        #                 else:
        #                     param_value = params[param_in]
        #                     dict_out[param_in].append(param_value)
        #             else:
        #                 continue

        # Remove 'None' values from parameter lists. Not needed!
        # for param_str in dict_out:
        #     for vali, val in enumerate(dict_out[param_str]):
        #         if val == None:
        #             indices_list.append(vali)
        #         else:
        #             continue

        # for i in sorted(set(indices_list), reverse=True):
        #     for param_str in dict_out:
        #         del dict_out[param_str][i]

        return dict_out

    def return_all_vals(self, list_in, line_type_str_list, param_str_list):
        '''Useful when we need min and max. WARNING: Lists between
        different parameters are not necessary of the same length.'''

        dict_out = {}

        for param_str in param_str_list:
            dict_out[param_str] = []

        for event in list_in:

            for line_type in line_type_str_list:
                params = getattr(event, line_type)

                for param_in in params:
                    if param_in in param_str_list:
                        if line_type == "one_lines" or \
                            line_type == "phase_lines":
                            dict_out[param_in] = \
                                dict_out[param_in] + params[param_in]
                        else: 
                            dict_out[param_in].append(params[param_in])
                    else:
                        continue

        # Remove 'None' values from parameter lists. Same as above?
        for param_str in dict_out:
            dict_out[param_str] = list(filter(None, dict_out[param_str]))

        return dict_out

    def build_velest(self, root_folder, list_in, phase_list=[],
        phase_list_logic='not in', sta_list=[], sta_list_logic='not in',
        use_deepcopy=True, phsort=1, nevline=1, min_phnum=1, tt_cut=999,
        use_hacc=False, add_name='', out_fp=None, reclassify=False,
        simulps=False, add_SP=False):

        if out_fp:
            root_folder = os.path.split(out_fp)[0]
        else:
            pass

        try:
            os.makedirs(root_folder)
        except OSError:
            if not os.path.isdir(root_folder):
                raise

        if use_deepcopy:
            list_in = copy.deepcopy(list_in)
        else:
            pass

        # Filter phase lines without phase type. Implement this in
        # filtering function (by default). Everything that is filtered,
        # goes to 'self.event_flist_alt', because of deepcopy. List
        # self.event_flist stays intact.
        self.event_velist = self.filter_mlin(list_in, 'phase_lines',
            'Phase_ID', 'empty', out_as_list=True, write_back_ind=False)

        # Filter out phase types.
        if add_SP:
            simulps = True
            _ = self.return_S_P(list_in)
            self.event_velist = self.filter_mlin(list_in, 'phase_lines',
                'Phase_ID', 'in', ['P', 'SP'], out_as_list=True,
                write_back_ind=False)
        else:
            self.event_velist = self.filter_mlin(list_in, 'phase_lines',
                'Phase_ID', phase_list_logic, phase_list, out_as_list=True,
                write_back_ind=False)

        if sta_list:
            sta_list = list(map(lambda x: x[0:4], sta_list))
        else:
            pass

        # Keep only phase lines of given seismic stations. Include this
        # when selecting earthquakes along with minimum phase number. 
        self.event_velist = self.filter_mlin(list_in, 'phase_lines',
            'Stat_name', sta_list_logic, sta_list, el_proc='slice',
            slice_start=0, slice_end=4, out_as_list=True,
            write_back_ind=False)

        # Keep only phase lines of first arrivals.
        if not add_SP:
            self.event_velist = self.keep_first_arrivals_test(list_in,
                use_deepcopy=False, write_back_ind=False,
                min_phasenum=min_phnum)
        else:
            pass

        # Exclude events with no remaining phase lines?

        # print(len(self.event_velist))

        if out_fp:
            velest_file = open(out_fp, encoding='utf-8', mode='w')
        else:
            velest_file = open(root_folder + f'/velest_lines{add_name}.cnv',
                encoding='utf-8', mode='w')

        velest_file.truncate()

        for event in sorted(self.event_velist, key=lambda x: x.ID):
            # print(event.phase_lines)
            event_lines = []

            if use_hacc:
                ev_year = event.h_line['YEAR']
                ev_month = event.h_line['MONTH']
                ev_day = event.h_line['DAY']
                ev_hour = event.h_line['HOUR']
                ev_min = event.h_line['MIN']
                ev_sec = round(event.h_line['SEC'], 2)
                ev_lat = round(event.h_line['LAT'], 4)
                ev_lon = round(event.h_line['LON'], 4)
                ev_depth = round(event.h_line['DEPTH'], 2)
                ev_rms = round(event.h_line['RMS_t_ress'], 2)
            else:
                ev_year = event.first_line['YEAR']
                ev_month = event.first_line['MONTH']
                ev_day = event.first_line['DAY']
                ev_hour = event.first_line['HOUR']
                ev_min = event.first_line['MIN']
                ev_sec = round(event.first_line['SEC'], 2)
                ev_lat = round(event.first_line['LAT'], 4)
                ev_lon = round(event.first_line['LON'], 4)
                ev_depth = round(event.first_line['DEPTH'], 2)
                ev_rms = round(event.first_line['RMS_t_ress'], 2)

            tev = event.tev

            ev_ms, ev_s = math.modf(ev_sec)
            ev_s = int(round(ev_s))
            ev_ms = int(round(ev_ms * 10**6))

            if event.first_line['MAG1']:
                ev_mag = round(event.first_line['MAG1'], 2)
            else:
                ev_mag = 0.00

            ev_gap = event.e_line['GAP']

            if ev_lat >= 0:
                ev_NS = "N"
            else:
                ev_NS = "S"

            if ev_lon >= 0:
                ev_EW = "E"
            else:
                ev_EW = "W"

            if simulps:
                nevline = 2
            else:
                pass

            if nevline == 1:
                event_line = "{:2d}{:2d}{:2d} {:2d}{:2d} {:5.2f} {:7.4f}{} \
{:8.4f}{}{:7.2f}{:7.2f}    {:3d}      {:4.2f}  EVID: {}\n".format(
                int(str(tev.year)[-2:]), tev.month, tev.day, tev.hour,
                tev.minute, round(tev.second + (tev.microsecond * 10**-6), 2),
                ev_lat, ev_NS, ev_lon, ev_EW, ev_depth, ev_mag,
                ev_gap, ev_rms, event.ID
                )
            elif nevline == 2:
                lat_frac, lat_int = math.modf(ev_lat)
                lon_frac, lon_int = math.modf(ev_lon)
                lat_deg = int(lat_int)
                lat_min = round(60 * lat_frac, 2)
                lon_deg = int(lon_int)
                lon_min = round(60 * lon_frac, 2)
                # event_line = "{:2d}{:2d}{:2d} {:2d}{:2d} {:5.2f} {:2d}{}{:5.2f} \
# {:3d}{}{:5.2f}{:7.2f}{:7.2f}\n".format(
                event_line = "{:2d}{:2d}{:2d} {:2d}{:2d} {:5.2f} {:2d}{}{:5.2f} \
{:3d}{}{:5.2f} {:6.2f}  {:5.2f}\n".format(
                int(str(tev.year)[-2:]), tev.month, tev.day, tev.hour,
                tev.minute, round(tev.second + (tev.microsecond * 10**-6), 2),
                lat_deg, ev_NS, lat_min, lon_deg, ev_EW, lon_min, ev_depth,
                ev_mag
                )
            else:
                event_line = "{:2d}{:2d}{:2d} {:2d}{:2d} {:5.2f} {:7.4f}{} \
{:8.4f}{}{:8.2f}{:7.2f}     99  0.0 0.00  1.0  1.0 \n".format(
                int(str(tev.year)[-2:]), tev.month, tev.day, tev.hour,
                tev.minute, round(tev.second + (tev.microsecond * 10**-6), 2),
                ev_lat, ev_NS, ev_lon, ev_EW, ev_depth, ev_mag
                )

            # velest_file.write(event_line)
            event_lines.append(event_line)

            wphi = 1
            ph_num = len(event.phase_lines['Phase_ID'])

            event_stas = event.phase_lines['Stat_name']
            event_phs = event.phase_lines['Phase_ID']
            phs_tph = event.phase_lines['tph']
            # S-P time for 'SP' phase and travel time for other phases
            phs_dtime = event.phase_lines['tt']

            event_pIDs = [i + j for i, j in zip(event_stas, event_phs)]

            sort_list = zip(phs_tph, event_pIDs)

            if phsort == 1:
                # Sort by arrival times.
                # Try Numpy's argsort.
                sort_list = sorted(sort_list, key=lambda x: x[0])
                # Make a list with sorted phase IDs.
                sort_pIDs = [x[1] for x in sort_list]
                # Get original indices in sorted order.
                stas_sind = list(map(
                    lambda x: event_pIDs.index(x), sort_pIDs))
            elif phsort == 2:
                # Sort by station names.
                sort_list = sorted(sort_list, key=lambda x: x[1])
                sort_pIDs = [x[1] for x in sort_list]
                stas_sind = list(map(
                    lambda x: event_pIDs.index(x), sort_pIDs))
            else:
                # Keep original order.
                stas_sind = list(map(
                    lambda x: event_pIDs.index(x), event_pIDs))

            for phi, iph in enumerate(stas_sind, 1):
                ph_station = event.phase_lines['Stat_name'][iph]
                ph_ID = event.phase_lines['Phase_ID'][iph]
                ph_name = re.sub('[^P|S|-]', '', ph_ID)
                tph = event.phase_lines['tph'][iph]
                ph_year = tph.year
                ph_month = tph.month
                ph_day = tph.day
                ph_hour = tph.hour
                ph_min = tph.minute
                ph_sec = tph.second + tph.microsecond * 10**-6
                ph_unc_class = event.phase_lines['Uncert_ind'][iph]
                ph_qind = event.phase_lines['Q_ind'][iph]
                ph_dtime = event.phase_lines['tt'][iph]

                if reclassify:
                    if not ph_qind or ph_qind.upper() == "E":
                        ph_unc_class += 1
                    else:
                        pass

                ph_msec, ph_sec = math.modf(ph_sec)
                ph_sec = int(round(ph_sec))
                ph_msec = int(round(ph_msec * 10**6))

                try:
                    ph_rdays = ph_dtime.days
                    ph_rsecs = ph_dtime.seconds
                    ph_rmsecs = ph_dtime.microseconds
                    ph_rtime = round(
                    (ph_rdays * 24 * 3600) + ph_rsecs + (ph_rmsecs * 10**-6), 2
                        )
                except AttributeError:
                    ph_rtime = ph_dtime

                # Add check for negative travel times.
                if abs(ph_rtime) <= tt_cut:
                    if simulps:
                        if ph_name == 'S':
                            ph_name = 'SP'
                        else:
                            pass
                        if wphi % 6.0 == 0 or phi == ph_num:
                            phase_block = "{:4s} {:2s}{}{:6.2f}\n".format(
                                ph_station[:4], ph_name, ph_unc_class,
                                ph_rtime)
                        else:
                            phase_block = "{:4s} {:2s}{}{:6.2f}".format(
                                ph_station[:4], ph_name, ph_unc_class,
                                ph_rtime)
                    else:
                        if wphi % 6.0 == 0 or phi == ph_num:
                            phase_block = "{:4s}{}{}{:6.2f}\n".format(
                                ph_station[:4], ph_name, ph_unc_class,
                                ph_rtime)
                        else:
                            phase_block = "{:4s}{}{}{:6.2f}".format(
                                ph_station[:4], ph_name, ph_unc_class,
                                ph_rtime)

                    # velest_file.write(phase_block)
                    event_lines.append(phase_block)
                    wphi += 1

                elif abs(ph_rtime) > tt_cut and phi == ph_num:
                    print('\nTravel time exceeded travel time limit.')
                    print(f'Specified travel time limit: {tt_cut} s.')
                    print('Event data:')
                    print('{}\n{}\n{} {} {} {} {} {} {} {}\n'.format(
                        event.ID, ph_station, ph_year, ph_month, ph_day,
                        ph_hour, ph_min, ph_sec, ph_msec, ph_rtime))
                    # velest_file.write('\n')
                    last_entry = event_lines.pop()
                    event_lines.append(last_entry + '\n')

                else:
                    print('\nTravel time exceeded travel time limit.')
                    print(f'Specified travel time limit: {tt_cut} s.')
                    print('Event data:')
                    print('{}\n{}\n{} {} {} {} {} {} {} {}\n'.format(
                        event.ID, ph_station, ph_year, ph_month, ph_day,
                        ph_hour, ph_min, ph_sec, ph_msec, ph_rtime))
                    print(tev, tph)
                    print(event.phase_lines)
                    pass

            if len(event_lines) > 1:
                write_lines = "".join(event_lines)
                velest_file.write(write_lines)
                velest_file.write('\n')
            else:
                pass

        velest_file.close()

    def export_events(self, root_folder, list_in, phase_list=[],
        phase_list_logic='not in', sta_list=[], sta_list_logic='not in',
        use_deepcopy=True, min_phnum=1, sp_add=''):
        # Implement option for high accuracy parameters (H line) and
        # check for folder existence (all such methods).

        try:
            os.makedirs(root_folder)
        except OSError:
            if not os.path.isdir(root_folder):
                raise

        if use_deepcopy:
            list_in = copy.deepcopy(list_in)
        else:
            pass

        # Filter phase lines without phase type. Implement this in
        # filtering function (by default). Everything that is filtered,
        # goes to 'self.event_flist_alt', because of deepcopy. List
        # self.event_flist stays intact.
        self.event_exlist = self.filter_mlin(list_in, 'phase_lines',
            'Phase_ID', 'empty', out_as_list=True, write_back_ind=False)

        # Filter out phase types.
        self.event_exlist = self.filter_mlin(list_in, 'phase_lines',
            'Phase_ID', phase_list_logic, phase_list, out_as_list=True,
            write_back_ind=False)

        if sta_list:
            sta_list = list(map(lambda x: x[0:4], sta_list))
        else:
            pass

        # Keep only phase lines of given seismic stations.
        self.event_exlist = self.filter_mlin(list_in, 'phase_lines',
            'Stat_name', sta_list_logic, sta_list, el_proc='slice',
            slice_start=0, slice_end=4, out_as_list=True,
            write_back_ind=False)

        # Keep only phase lines of first arrivals.
        self.event_exlist = self.keep_first_arrivals_c(list_in,
            use_deepcopy=False, write_back_ind=False, min_phasenum=min_phnum)

        # print(len(self.event_exlist))

        event_sp = os.path.join(root_folder, 'exported_events' + sp_add)
        event_file = open(event_sp, encoding='utf-8', mode='a')
        event_file.truncate()

        head = "YEAR\tMONTH\tDAY\tHOUR\tMIN\tSEC\tLAT\tLON\tDEPTH\tMAG\tPHN\n"
        event_file.write(head)

        for event in sorted(
           self.event_exlist, key=lambda x: x.first_line['YEAR']):
            ev_year = event.first_line['YEAR']
            ev_month = event.first_line['MONTH']
            ev_day = event.first_line['DAY']
            ev_hour = event.first_line['HOUR']
            ev_min = event.first_line['MIN']
            ev_sec = event.first_line['SEC']

            ev_lat = round(event.first_line['LAT'], 4)
            ev_lon = round(event.first_line['LON'], 4)
            ev_depth = round(event.first_line['DEPTH'], 2)

            if event.first_line['MAG1']:
                ev_mag = round(event.first_line['MAG1'], 2)
            else:
                ev_mag = None

            ph_num = len(event.phase_lines['Phase_ID'])

            ev_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                ev_year, ev_month, ev_day, ev_hour, ev_min,
                ev_sec, ev_lat, ev_lon, ev_depth, ev_mag, ph_num
                )

            event_file.write(ev_line)

            # event_file.write('\n')

        event_file.close()

    def export_events_un(self, root_folder, list_in, grid_inst=None):
        # Implement option for high accuracy parameters (H line) and
        # check for folder existence (all such methods).

        try:
            os.makedirs(root_folder)
        except OSError:
            if not os.path.isdir(root_folder):
                raise

        event_file = open(root_folder + '/exported_events_un',
            encoding='utf-8', mode='a')
        event_file.truncate()

        head = \
    "YEAR\tMONTH\tDAY\tHOUR\tMIN\tSEC\tLAT\tLON\tDEPTH\tMAG\tPHN\tCAT\tID\n"
        event_file.write(head)

        for event in sorted(list_in, key=lambda x: x.ID):

            if getattr(event, 'params', None):
                ev_year = event.params['YEAR']
                ev_month = event.params['MONTH']
                ev_day = event.params['DAY']
                ev_hour = event.params['HOUR']
                ev_min = event.params['MIN']
                ev_sec = event.params['SEC']
                ev_lat = round(event.params['LAT'], 4)
                ev_lon = round(event.params['LON'], 4)
                ev_depth = round(event.params['DEPTH'], 2)
                ev_mag = round(event.params['MAG1'], 2)
                ev_phn = event.params['PHN']
            else:
                ev_year = event.first_line['YEAR']
                ev_month = event.first_line['MONTH']
                ev_day = event.first_line['DAY']
                ev_hour = event.first_line['HOUR']
                ev_min = event.first_line['MIN']
                ev_sec = event.first_line['SEC']
                ev_lat = round(event.first_line['LAT'], 4)
                ev_lon = round(event.first_line['LON'], 4)
                ev_depth = round(event.first_line['DEPTH'], 2)
                ev_mag = round(event.first_line['MAG1'], 2)
                ev_phn = event.first_line['PHN']

            ev_line = \
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    ev_year, ev_month, ev_day, ev_hour, ev_min, ev_sec,
                    ev_lat, ev_lon, ev_depth, ev_mag, ev_phn,
                    event.catalog_name, event.ID
                )

            event_file.write(ev_line)

        event_file.close()

        if grid_inst:
            event_file_c = open(root_folder + '/exported_events_un_cells',
                encoding='utf-8', mode='a')
            event_file_c.truncate()
            event_file_c.write(head)

            for cell in grid_inst:
                if cell.selected_objects:
                    write_cID = "\nCELL ID: {}\n".format(cell.ID)
                    event_file_c.write(write_cID)

                    for event in cell.selected_objects:

                        if getattr(event, 'params', None):
                            ev_year = event.params['YEAR']
                            ev_month = event.params['MONTH']
                            ev_day = event.params['DAY']
                            ev_hour = event.params['HOUR']
                            ev_min = event.params['MIN']
                            ev_sec = event.params['SEC']
                            ev_lat = round(event.params['LAT'], 4)
                            ev_lon = round(event.params['LON'], 4)
                            ev_depth = round(event.params['DEPTH'], 2)
                            ev_mag = round(event.params['MAG1'], 2)
                            ev_phn = event.params['PHN']
                        else:
                            ev_year = event.first_line['YEAR']
                            ev_month = event.first_line['MONTH']
                            ev_day = event.first_line['DAY']
                            ev_hour = event.first_line['HOUR']
                            ev_min = event.first_line['MIN']
                            ev_sec = event.first_line['SEC']
                            ev_lat = round(event.first_line['LAT'], 4)
                            ev_lon = round(event.first_line['LON'], 4)
                            ev_depth = round(event.first_line['DEPTH'], 2)
                            ev_mag = round(event.first_line['MAG1'], 2)
                            ev_phn = event.first_line['PHN']

                        ev_line = \
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        ev_year, ev_month, ev_day, ev_hour, ev_min, ev_sec,
                        ev_lat, ev_lon, ev_depth, ev_mag, ev_phn,
                        event.catalog_name, event.ID
                            )

                        event_file_c.write(ev_line)

            event_file_c.close()

        else:
            pass


    def fetch_trexcerpt(self, root_folder, list_in, db_out, db_in=[],
        by_station=False, duration_sec=300, orig_delta_sec=0,
        chan_regex='=~/.*/', sta_regex=None, sta_list=None,
        use_deepcopy=True, min_phnum=1):

        try:
            os.makedirs(root_folder)
        except OSError:
            if not os.path.isdir(root_folder):
                raise

        self.trexcerpt_lines = []

        if use_deepcopy:
            list_in = copy.deepcopy(list_in)
        else:
            pass

        trexcerpt_file = open(root_folder + '/trexcerpt_lines',
            encoding='utf-8', mode='w')
        trexcerpt_file.truncate()
        trexcerpt_file.write('#!/bin/bash\n')

        # # Filter phase lines without phase type. Implement this in
        # # filtering function (by default).
        # self.event_txlist = self.filter_mlin(list_in, 'phase_lines',
        #     'Phase_ID', 'empty', out_as_list=True, write_back_ind=False)

        # # Filter out magnitude picks.
        # self.event_txlist = self.filter_mlin(list_in, 'phase_lines',
        #     'Phase_ID', 'not in', ['m', 'M'], out_as_list=True,
        #     write_back_ind=False)

        # # Keep only phase lines of first arrivals.
        # self.event_txlist = self.keep_first_arrivals_c(list_in,
        #     use_deepcopy=False, write_back_ind=False, min_phasenum=min_phnum)

        # print(len(self.event_txlist))
        
        nevs = len(list_in)

        # for event in self.event_txlist:
        for ei, event in enumerate(sorted(list_in, key=lambda x: x.ID)):

            if getattr(event, 'params', None):
                ev_year = event.params['YEAR']
                ev_month = event.params['MONTH']
                ev_day = event.params['DAY']
                ev_hour = event.params['HOUR']
                ev_min = event.params['MIN']
                ev_sec = event.params['SEC']
            else:
                ev_year = event.first_line['YEAR']
                ev_month = event.first_line['MONTH']
                ev_day = event.first_line['DAY']
                ev_hour = event.first_line['HOUR']
                ev_min = event.first_line['MIN']
                ev_sec = event.first_line['SEC']

            ev_msec, ev_sec = math.modf(ev_sec)
            ev_sec = int(round(ev_sec))
            ev_msec = int(round(ev_msec * 10**6))

            if ev_hour == 24:
                ev_hour = 0
                tde = timedelta(days=1)
            else:
                tde = timedelta()
            
            if ev_min == 60:
                ev_min = 0
                tde = tde + timedelta(seconds=3600)
            else:
                pass
            
            if ev_sec == 60.0:
                ev_sec = 0
                tde = tde + timedelta(seconds=60)
            else:
                pass

            t0 = datetime(ev_year, ev_month, ev_day, ev_hour, ev_min, ev_sec,
                ev_msec) + tde
            td0 = t0 + timedelta(seconds=orig_delta_sec)
            start_time = "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:05.2f}".format(
                td0.year, td0.month, td0.day, td0.hour, td0.minute,
                td0.second + round(td0.microsecond * 10**-6, 2))
            duration_sec_add = duration_sec + abs(orig_delta_sec)

            if not db_in:
                dbt_in = f"/mnt/szm/{ev_year}/db/arhiv"
                # OE seismic network.
                # db_in = f"/seismik2/wf/all_{ev_year}"
            else:
                dbt_in = db_in
            
            wf_name = f"/mnt/szm/gregor/events_fs/{event.ID}/seismograms/%"

            # t1 = datetime.now(datetime.UTC) - timedelta(days=365)
            # dt = t1 - t0
            # days = dt.total_seconds() / (3600 * 24)

            if by_station:
                sta_list = list(set(event.phase_lines['Stat_name']))
                sta_list.append('RIY')
            else:
                pass

            if sta_list:
                sta_regex = "=~/{}/".format(('|').join(sorted(sta_list)))
            else:
                sta_regex = '=~/.*/'

#             if ei < nevs - 1:
#                 trexcerpt_str = \
# "trexcerpt -Dv -o sd -c \'sta{} && chan{}\' -w \'{}\' \'{}\' \'{}\' \'{}\' {} && \n"
#             else:
#                 trexcerpt_str = \
# "trexcerpt -D -o sd -c \'sta{} && chan{}\' -w \'{}\' \'{}\' \'{}\' \'{}\' {}\n"

            trexcerpt_str = \
"trexcerpt -Dv -o sd -c \'sta{} && chan{}\' -w \'{}\' \'{}\' \'{}\' \'{}\' {} \n"

            trexcerpt_str = trexcerpt_str.format(sta_regex, chan_regex,
                wf_name, dbt_in, db_out, start_time, duration_sec_add)
            trexcerpt_file.write(trexcerpt_str)
            self.trexcerpt_lines.append(trexcerpt_str)

        trexcerpt_file.close()

    def event_fs(self, save_dir, list_in, pyrocko_markers=False):
        # Implement option for filtered Nordic block.

        uncert_dict = {0:0.099, 1:0.199, 2:0.499, 3:0.999}
        fmot_dict = {'C':1, 'D':-1}

        try:
            os.makedirs(save_dir)
        except OSError:
            if not os.path.isdir(save_dir):
                raise

        # save_dir_fs = "{}events_fs/".format(save_dir)
        save_dir_fs = os.path.join(save_dir, 'events_fs')

        try:
            os.makedirs(save_dir_fs)
        except OSError:
            if not os.path.isdir(save_dir_fs):
                raise

        # Write Snuffler event file.
        # Template.
        # --- !pf.Event
        # lat: 36.9895
        # lon: 27.7647
        # time: 2009-10-05 20:17:46.00
        # name: ''
        # depth: 13500
        # magnitude: 2.3
        # catalog: ''

        for event in list_in:
            # save_dir_fse = "{}{}/".format(save_dir_fs, event.ID)
            # file_name_Nordic = "{}Nordic_event".format(save_dir_fse)
            # file_name_Snuffler = "{}Snuffler_YAML_event".format(save_dir_fse)
            save_dir_fse = os.path.join(save_dir_fs, event.ID)
            file_name_Nordic = os.path.join(save_dir_fse, 'Nordic_event')
            file_name_Snuffler = os.path.join(save_dir_fse, 'Snuffler_YAML_event')
            file_name_markers = os.path.join(save_dir_fse, event.ID + '.pf')

            try:
                os.makedirs(save_dir_fse)
            except OSError:
                if not os.path.isdir(save_dir_fse):
                    raise

            # if getattr(event, 'Nordic', None):
            if event.Nordic:
                file_write_Nordic = open(file_name_Nordic, 'a')
                file_write_Nordic.truncate()
                Nordic_block = event.event_block
                for line in Nordic_block:
                    line_write = "{}\n".format(line)
                    file_write_Nordic.write(line_write)
                file_write_Nordic.write('\n')
                file_write_Nordic.close()
            else:
                pass

            # if getattr(event, 'params', None):
            if not event.Nordic:
                lat = event.params['LAT']
                lon = event.params['LON']
                year = event.params['YEAR']
                month = event.params['MONTH']
                day = event.params['DAY']
                hour = event.params['HOUR']
                minutes = event.params['MIN']
                seconds = event.params['SEC']
                depth = event.params['DEPTH']
                mag = event.params['MAG1']
            else:
                # lat = event.first_line['LAT']
                # lon = event.first_line['LON']
                # year = event.first_line['YEAR']
                # month = event.first_line['MONTH']
                # day = event.first_line['DAY']
                # hour = event.first_line['HOUR']
                # minutes = event.first_line['MIN']
                # seconds = event.first_line['SEC']
                # depth = event.first_line['DEPTH']
                # mag = event.first_line['MAG1']
                lat = event.h_line['LAT']
                lon = event.h_line['LON']
                year = event.h_line['YEAR']
                month = event.h_line['MONTH']
                day = event.h_line['DAY']
                hour = event.h_line['HOUR']
                minutes = event.h_line['MIN']
                seconds = event.h_line['SEC']
                depth = event.h_line['DEPTH']
                mag = event.first_line['MAG1']

            file_write_Snuffler = open(file_name_Snuffler, 'a')
            file_write_Snuffler.truncate()
            Snuffler_block = """\
--- !pf.Event
lat: {}
lon: {}
time: {:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02.2f}
name: '{}'
depth: {}
magnitude: {}
catalog: '{}'
        """.format(
                # event.first_line['LAT'], event.first_line['LON'],
                # event.first_line['YEAR'], event.first_line['MONTH'],
                # event.first_line['DAY'], event.first_line['HOUR'],
                # event.first_line['MIN'], event.first_line['SEC'], event.ID,
                # event.first_line['DEPTH'] * 1000, event.first_line['MAG1'],
                # event.catalog_name
                # event.params['LAT'], event.params['LON'],
                # event.params['YEAR'], event.params['MONTH'],
                # event.params['DAY'], event.params['HOUR'],
                # event.params['MIN'], event.params['SEC'], event.ID,
                # event.params['DEPTH'] * 1000, event.params.get('MAG1'),
                # event.catalog_name
                lat, lon, year, month, day, hour, minutes, seconds, event.ID,
                depth * 1000, mag, event.catalog_name
                )
            file_write_Snuffler.write(Snuffler_block)
            file_write_Snuffler.write('\n')
            file_write_Snuffler.close()

            if pyrocko_markers:
                try:
                    from pyrocko.gui.marker import (EventMarker, Marker,
                                                    PhaseMarker,
                                                    associate_phases_to_events,
                                                    save_markers)
                    from pyrocko.model import Event

                    if event.Nordic:
                        markers = []

                        # epoch = datetime.utcfromtimestamp(0)
                        epoch = datetime(1970, 1, 1, 0, 0)

                        ev_time = (event.tev - epoch).total_seconds()
                        # ev_time = event.tev.timestamp()

                        pyr_event = Event(lat=lat, lon=lon, time=ev_time,
                            name=event.ID, depth=depth, magnitude=mag,
                            catalog=event.catalog_name)
                        ev_hash = pyr_event.get_hash()
                        pyr_eventm = EventMarker(
                            pyr_event, event_hash=ev_hash
                            )

                        markers.append(pyr_eventm)

                        stas = event.phase_lines['Stat_name']

                        for phi, sta in enumerate(stas):
                            pick_obj = event.phase_lines['pick_obj'][phi]

                            if pick_obj:
                                tph = pick_obj.tpick
                                ph_channel = pick_obj.channel
                                ph_sta = pick_obj.statID
                                ph_unct = pick_obj.unctime
                                ph_ID = pick_obj.phase
                            else:
                                tph = event.phase_lines['tph'][phi]
                                component = event.phase_lines['Component'][phi]
                                if component:
                                    ph_channel = "*" + component
                                else:
                                    phchannel = "*"
                                ph_sta = event.phase_lines['Stat_name'][phi]
                                ph_unct = None
                                ph_ID = event.phase_lines['Phase_ID'][phi]

                            if ph_ID:
                                ph_name = re.sub('[^P|S]', '', ph_ID)
                            else:
                                ph_name = "?"

                            ph_firstmot = event.phase_lines['First_mot'][phi]

                            if ph_firstmot:
                                ph_firstmot = fmot_dict[ph_firstmot]
                            else:
                                ph_firstmot = None

                            ph_nslc = [('*', ph_sta, '', ph_channel)]
                            ph_time = (tph - epoch).total_seconds()

                            pyr_phm = PhaseMarker(ph_nslc, ph_time, ph_time,
                                event=pyr_event, event_hash=ev_hash,
                                phasename=ph_name, polarity=ph_firstmot,
                                uncertainty=ph_unct)

                            markers.append(pyr_phm)

                    if markers:
                        save_markers(markers, file_name_markers, fdigits=3)
                    else:
                        pass

                except ImportError:
                    print('\nNo Pyrocko modules found.')

    def fdsnws_fetch(self, client_user, root_folder, list_in,
        sta_objects=None, channel_list=None, duration_sec=300,
        orig_delta_sec=0, save_format='MSEED', credentials={},
        client_pass='', client_port=18002, client_inst='Anonymous',
        debug=False, timestamp=None):

        if not timestamp:
            st = list(time.localtime(time.time()))[0:-4]
            timestamp = "{:04d}{:02d}{:02d}-{:02d}{:02d}".format(
                st[0], st[1], st[2], st[3], st[4])
        else:
            pass

        log_write = open(root_folder + f'log_fdsnws_fetch_{timestamp}', 'a')
        log_write.truncate()

        try:
            import obspy
            import_bool = True
        except ImportError as e:
            print('\nObsPy module not found.')
            print('No seismograms have been downloaded.')
            print('Writing lines for "fdsnws_fetch" script ...\n')
            log_write.write('\n' + str(e))
            log_write.write('\n')
            import_bool = False

        if import_bool:
            from obspy.clients.fdsn import RoutingClient

            client = RoutingClient(
                "eida-routing", credentials=credentials,
                debug=debug
                )
        else:
            file_write = open(root_folder + f'fdsnws_fetch_{timestamp}', 'a')
            file_write.truncate()

        for event in list_in:

            wf_name = f"/events_fs/{event.ID}/seismograms/fdsnws_fetch/"

            try:
                os.makedirs(root_folder + wf_name)
            except OSError:
                if not os.path.isdir(root_folder):
                    raise

            if getattr(event, 'params', None):
                ev_year = event.params['YEAR']
                ev_month = event.params['MONTH']
                ev_day = event.params['DAY']
                ev_hour = event.params['HOUR']
                ev_min = event.params['MIN']
                ev_sec = event.params['SEC']
            else:
                ev_year = event.first_line['YEAR']
                ev_month = event.first_line['MONTH']
                ev_day = event.first_line['DAY']
                ev_hour = event.first_line['HOUR']
                ev_min = event.first_line['MIN']
                ev_sec = event.first_line['SEC']

            ev_msec, ev_sec = math.modf(ev_sec)
            ev_sec = int(round(ev_sec))
            ev_msec = int(round(ev_msec * 10**6))

            if ev_hour == 24:
                ev_hour = 0
                tde = timedelta(days=1)
            else:
                tde = timedelta()
            
            if ev_min == 60:
                ev_min = 0
                tde = tde + timedelta(seconds=3600)
            else:
                pass
            
            if ev_sec == 60.0:
                ev_sec = 0
                tde = tde + timedelta(seconds=60)
            else:
                pass

            t0 = datetime(ev_year, ev_month, ev_day, ev_hour, ev_min,
                ev_sec, ev_msec) + tde
            td0 = t0 + timedelta(seconds=orig_delta_sec)
            duration_sec_add = duration_sec + abs(orig_delta_sec)
            td1 = td0 + timedelta(seconds=duration_sec_add)
            
            if import_bool:
                stime = obspy.UTCDateTime(td0)
                etime = obspy.UTCDateTime(td1)
            else:
                stime = \
                    "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}:{:06d}".format(
                        td0.year, td0.month, td0.day, td0.hour, td0.minute,
                        td0.second, td0.microsecond
                    )
                etime = \
                    "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}:{:06d}".format(
                        td1.year, td1.month, td1.day, td1.hour, td1.minute,
                        td1.second, td1.microsecond
                    )

            for sta in sta_objects:
                ntwrk = sta.params['NTWRK']
                acrym = sta.params['ACRYM']

                for chn in channel_list:
                    fn = '{}.{}..{}.{}.{}.{}.{}.{}.{}'.format(
                        ntwrk, acrym, chn, td0.year, td0.month, td0.day,
                        td0.hour, td0.minute, td0.second)
                    fnt = '.{}.{}.{}.{}.{}.{}'.format(
                        td0.year, td0.month, td0.day, td0.hour, td0.minute,
                        td0.second)
                    save_path = wf_name + fn

                    if import_bool:
                        try:
                            strm = client.get_waveforms(network=ntwrk,
                                station=acrym, location='*', channel=chn,
                                starttime=stime, endtime=etime, )
                                # filename=root_folder + save_path
                            print('\n', fn)
                            print(strm)
                            log_write.write('\n' + fn)
                            log_write.write('\n' + str(strm))
                            log_write.write('\n')
                            if len(strm) > 0:
                                strm.merge(method=1, fill_value=0)
                                # strm.trim(stime, etime)
                                # strm.plot()
                                # strm_save = root_folder + save_path
                                # strm.write(strm_save, format='MSEED')
                                # All components in one stream and output by
                                # traces.
                                for trc in strm:
                                    print('\t', trc.id)
                                    log_write.write('\t' + trc.id)
                                    log_write.write('\n')
                                    trc_save = \
                                        root_folder + wf_name + trc.id + fnt
                                    trc.write(trc_save, format='MSEED')
                            else:
                                pass
                        except Exception as e:
                            print('\n', fn)
                            print(e)
                            log_write.write('\n' + fn)
                            log_write.write('\n' + str(e))
                            log_write.write('\n')
                    else:
                        fdsnws__line = \
'fdsnws_fetch -v -r 2 -w 5 -t 15 -N "{}" -S "{}" -L "*" -C "{}" -s "{}" \
-e "{}" -o ".{}" -a "{}"\n'.format(
    ntwrk, acrym, chn, stime, etime, save_path + '.MSEED',
    credentials.get('EIDA_TOKEN')
)
                        file_write.write(fdsnws__line)

        if not import_bool:
            file_write.close()
        else:
            pass

        log_write.close()

    def write_nordics(self, save_dir, list_in, filtered=0, grid_inst=None, 
        timestamp=None, sp_add=''):

        try:
            os.makedirs(save_dir)
        except OSError:
            if not os.path.isdir(save_dir):
                raise

        if not timestamp:
            st = list(time.localtime(time.time()))[0:-4]
            timestamp = "{:04d}{:02d}{:02d}-{:02d}{:02d}".format(
                st[0], st[1], st[2], st[3], st[4])
        else:
            pass

        # Write Nordic blocks of selected events to file.
        file_name = os.path.join(save_dir, f'sel_Nordic_{timestamp}{sp_add}')
        file_write = open(file_name, 'a')
        file_write.truncate()

        for event in sorted(list_in, key=lambda x: x.ID):
            Nordic_block = event.event_block
            for line in Nordic_block:
                line_write = "{}\n".format(line)
                file_write.write(line_write)
            file_write.write('\n')

        file_write.close()

        if grid_inst:
            # Write Nordic blocks of selected events to file by cell.
            file_name_c = os.path.join(
                save_dir, f'sel_Nordic_cells_{timestamp}{sp_add}'
                )
            file_write_c = open(file_name_c, 'a')
            file_write_c.truncate()

            for cell in grid_inst:
                if cell.selected_objects:
                    write_cID = "CELL ID: {}\n\n".format(cell.ID)
                    file_write_c.write(write_cID)
                    sobj = sorted(cell.selected_objects, key=lambda x: x.ID)
                    for event in sobj:
                        Nordic_block = event.event_block
                        for line in Nordic_block:
                            line_write = "{}\n".format(line)
                            file_write_c.write(line_write)
                        file_write_c.write('\n')
                    file_write_c.write('\n')

            file_write_c.close()
        else:
            pass

        if filtered == 1:
            # Write Nordic blocks of selected events to file.
            # Filtered phase lines.
            file_name_f = os.path.join(
                save_dir, f'sel_Nordic_filtered_{timestamp}{sp_add}'
                )
            file_write_f = open(file_name_f, 'a')
            file_write_f.truncate()

            for event in sorted(list_in, key=lambda x: x.ID):
                event.filter_nordic_phases()
                Nordic_block_f = event.event_block_f
                for line in Nordic_block_f:
                    line_write = "{}\n".format(line)
                    file_write_f.write(line_write)
                file_write_f.write('\n')

            file_write_f.close()

            if grid_inst:
                # Write Nordic blocks of selected events to file by cell.
                # Filtered phase lines.
                file_name_cf = os.path.join(
                    save_dir, f'sel_Nordic_cells_filtered_{timestamp}{sp_add}'
                    )
                file_write_cf = open(file_name_cf, 'a')
                file_write_cf.truncate()

                for cell in grid_inst:
                    if cell.selected_objects:
                        write_cID = "CELL ID: {}\n\n".format(cell.ID)
                        file_write_cf.write(write_cID)
                        sobj = sorted(
                            cell.selected_objects, key=lambda x: x.ID)
                        for event in sobj:
                            event.filter_nordic_phases()
                            Nordic_block_f = event.event_block_f
                            for line in Nordic_block_f:
                                line_write = "{}\n".format(line)
                                file_write_cf.write(line_write)
                            file_write_cf.write('\n')
                        file_write_cf.write('\n')

                file_write_cf.close()
            else:
                pass

        else:
            pass

    def write_IDs(self, save_dir, list_in=None, timestamp=None):

        try:
            os.makedirs(save_dir)
        except OSError:
            if not os.path.isdir(save_dir):
                raise

        if not timestamp:
            st = list(time.localtime(time.time()))[0:-4]
            timestamp = "{:04d}{:02d}{:02d}-{:02d}{:02d}".format(
                st[0], st[1], st[2], st[3], st[4])
        else:
            pass

        # Write IDs of selected events to file.
        file_name = os.path.join(save_dir, f'sel_IDs_{timestamp}')
        file_write = open(file_name, 'a')
        file_write.truncate()

        if list_in:
            event_list = list_in
        else:
            event_list = self.events

        id_list = list(
            map(lambda x: x.ID, sorted(event_list, key=lambda x: x.ID)))
        id_str = '\n'.join(id_list)
        file_write.write(id_str)

        file_write.close()

    def sta_stats(self, root_folder, list_in, phase_list=[],
        phase_list_logic='not in', sta_list=[], sta_list_logic='not in',
        use_deepcopy=True, min_phnum=1):
        # Implement option for high accuracy parameters (H line).

        try:
            os.makedirs(root_folder)
        except OSError:
            if not os.path.isdir(root_folder):
                raise

        if use_deepcopy:
            list_in = copy.deepcopy(list_in)
        else:
            pass

        all_uncs = {0:0, 1:0, 2:0, 3:0, 4:0, 9:0}

        # Filter phase lines without phase type. Implement this in
        # filtering function (by default). Everything that is filtered,
        # goes to 'self.event_flist_alt', because of deepcopy. List
        # self.event_flist stays intact.
        self.event_velist = self.filter_mlin(list_in, 'phase_lines',
            'Phase_ID', 'empty', out_as_list=True, write_back_ind=False)

        # Filter out phase types.
        self.event_velist = self.filter_mlin(list_in, 'phase_lines',
            'Phase_ID', phase_list_logic, phase_list, out_as_list=True,
            write_back_ind=False)

        if sta_list:
            sta_list = list(map(lambda x: x[0:4], sta_list))
        else:
            pass

        # Keep only phase lines of given seismic stations. Include this
        # when selecting earthquakes along with minimum phase number. 
        self.event_velist = self.filter_mlin(list_in, 'phase_lines',
            'Stat_name', sta_list_logic, sta_list, el_proc='slice',
            slice_start=0, slice_end=4, out_as_list=True,
            write_back_ind=False)

        # Keep only phase lines of first arrivals.
        self.event_velist = self.keep_first_arrivals_c(list_in,
            use_deepcopy=False, write_back_ind=False, min_phasenum=min_phnum)

        # Exclude events with no remaining phase lines?

        # print(len(self.event_velist))

        sta_stat = open(root_folder + '/sta_stats.txt',
            encoding='utf-8', mode='a')
        sta_stat.truncate()

        sta_dict = {}

        for event in sorted(self.event_velist, key=lambda x: x.ID):
            event_year = event.first_line['YEAR']
            event_stas = event.phase_lines['Stat_name']

            for iph, sta in enumerate(event_stas):
                if sta not in list(sta_dict):
                    sta_dict[sta] = {}
                    sta_dict[sta]['uncs'] = []
                    sta_dict[sta]['years'] = []
                else:
                    pass

                ph_unc_class = event.phase_lines['Uncert_ind'][iph]
                sta_dict[sta]['uncs'].append(ph_unc_class)
                sta_dict[sta]['years'].append(event_year)

        sta_sort = sorted(
            sta_dict, key=lambda x: len(sta_dict[x]['uncs']), reverse=True)

        for sta in sta_sort:
            phn = len(sta_dict[sta]['uncs'])
            sta_dict[sta]['phn'] = phn
            phn_unc = Counter(sta_dict[sta]['uncs'])
            year_min = min(sta_dict[sta]['years'])
            year_max = max(sta_dict[sta]['years'])
            unc_line = ""

            for unc in sorted(phn_unc):
                unc_line += " {}:{:6d}".format(unc, phn_unc[unc])
                all_uncs[unc] += phn_unc[unc]

            stat_line = "{:5s} {:4d} {:4d} {:6d} {}\n".format(
                sta, year_min, year_max, phn, unc_line)
            sta_stat.write(stat_line)

        aunc_line = ""

        for unc in sorted(all_uncs):
            aunc_line += " {}:{:6d}".format(unc, all_uncs[unc])

        astat_line = "{:5s} {:4s} {:4s} {:6d} {}\n".format(
            'SUM', '', '', phn, aunc_line)
        sta_stat.write(astat_line)

        sta_stat.write('\n')
        sta_stat.close()

        self.sta_dict = sta_dict

    def select_by_ID(self, ID_file, list_in=None, return_list=False):
        self.selected_IDs = []

        with open(ID_file) as input_data:
            for line in input_data:
                self.selected_IDs.append(line.strip())
        input_data.close()

        if list_in:
            events_in = copy.copy(list_in)
        else:
            events_in = copy.copy(self.events)

        # Add routine for duplicate IDs. Make user decide.
        selected_events = list(filter(
            lambda x: x.ID in self.selected_IDs, events_in))

        if return_list:
            return selected_events
        else:
            self.event_flist = selected_events

    def read_ID_file(self, ID_file, return_list=False):
        self.selected_IDs = []

        with open(ID_file) as input_data:
            for line in input_data:
                self.selected_IDs.append(line.strip())
        input_data.close()

        if return_list:
            return self.selected_IDs
        else:
            pass

    def remove_inacc_clock(self, list_in=None, return_list=False,
        write_back_ind=True):

        if list_in:
            events_in = copy.copy(list_in)
        else:
            events_in = copy.copy(self.events)

        events_out = []

        for event in events_in:
            sta_rem = []
            bindices = []

            phll = len(event.phase_lines['Stat_name'])

            for i, unc in enumerate(event.phase_lines['Uncert_ind']):
                if unc == 9:
                    sta = event.phase_lines['Stat_name'][i]
                    sta_rem.append(sta)

            for i, sta in enumerate(event.phase_lines['Stat_name']):
                if sta in sta_rem:
                    bindices.append(-(phll - i))

            if write_back_ind and bindices:
                event.filt_bi_phl.append(bindices)

            if bindices:
                for par in event.phase_lines:
                    for bi in bindices:
                        # print(event.phase_lines[par])
                        # print(bi, par, event.ID)
                        del event.phase_lines[par][bi]
            else:
                pass

            if event.phase_lines['Stat_name']:
                events_out.append(event)

        if return_list:
            return events_out
        else:
            self.event_flist = events_out

    def calc_rmsres(self, event_list, velest=False, overwrite=True,
        sweight=1.0, iqf=1):

        sqrress = []
        wress = []
        nobss = []

        for ev in event_list:
            ev.calc_rmsres(velest=velest, overwrite=overwrite,
                sweight=sweight, iqf=iqf)

            if ev.rmsres is not None:
                sqrress.append(ev.sqrres)
                wress.append(ev.wres)
                nobss.append(ev.nobsnot0)
            else:
                pass

        wress = np.array(wress)
        sqrress = np.array(sqrress)
        nobss = np.array(nobss)

        tnobs = float(sum(nobss))
        dres = sum(wress)
        dsqrres = sum(sqrress)
        mdsqrres = (dsqrres - (dres**2 / tnobs)) / tnobs
        rmsres = mdsqrres**0.5

        return mdsqrres, rmsres

    def calc_gap(self, event_list, stas_obj, overwrite=True,
        use_epicd=False):
        for ev in event_list:
            ev.calc_gap(stas_obj, overwrite, use_epicd)

    def calc_dh(self, event_list):
        for ev in event_list:
            ev.calc_dh()

    def count_phases(self, event_list):
        for ev in event_list:
            ev.count_phases()

    def count_stations(self, event_list):
        for ev in event_list:
            ev.count_stations()

    def return_S_P(self, event_list, tt_cut=900):
        phase_pairs = {
            'P': 'S',
            'PG': 'SG',
            'PB': 'SB',
            'PN': 'SN',
            'PG)': 'SG)'
        }

        ptts = []
        puncs = []
        # pws = []
        stts = []
        suncs = []
        # sws = []
        sptts = []
        oterrs = []
        eventids = []
        stationacryms = []

        for event in event_list:
            ret_sp_lines = getattr(event, 'sp_lines', None)

            if ret_sp_lines:
                sp_exists = True
            else:
                sp_exists = False
                event.sp_lines = {}
                event.sp_lines['Phase_ID'] = []
                event.sp_lines['SP_time'] = []
                event.sp_lines['Stat_name'] = []
                event.sp_lines['Uncert_ind'] = []
                event.sp_lines['tph'] = []
                event.sp_lines['tt'] = []

            tev = event.tev
            ev_eot = event.e_line['Orig_t_err']

            if not ev_eot:
                # continue
                ev_eot = 0

            stas = event.phase_lines['Stat_name']

            stations = []
            arrts = []
            phases = []
            uncs = []
            evids = []
            tphs = []
            ph_dtimes = []
            # weights = []

            for ista, sta in enumerate(stas):
                ph_station = event.phase_lines['Stat_name'][ista]
                ph_ID = event.phase_lines['Phase_ID'][ista].upper()
                tph = event.phase_lines['tph'][ista]
                ph_year = tph.year
                ph_month = tph.month
                ph_day = tph.day
                ph_hour = tph.hour
                ph_min = tph.minute
                ph_sec = tph.second + tph.microsecond * 10**-6
                ph_unc = event.phase_lines['Uncert_ind'][ista]
                ph_qind = event.phase_lines['Q_ind'][ista]
                # ph_w = event.phase_lines['Weight'][ista] / 10.0
                ph_dtime = event.phase_lines['tt'][ista]

                if ph_ID == 'SP':
                    continue
                else:
                    pass

                ph_msec, ph_sec = math.modf(ph_sec)
                ph_sec = int(round(ph_sec))
                ph_msec = int(round(ph_msec * 10**6))

                ph_rdays = ph_dtime.days
                ph_rsecs = ph_dtime.seconds
                ph_rmsecs = ph_dtime.microseconds
                ph_rtime = \
                    (ph_rdays * 24 * 3600) + ph_rsecs + (ph_rmsecs * 10**-6)

                # Add check for negative travel times.
                if abs(ph_rtime) <= tt_cut:
                    stations.append(ph_station)
                    arrts.append(ph_rtime)
                    phases.append(ph_ID)
                    uncs.append(ph_unc)
                    evids.append(event.ID)
                    tphs.append(tph)
                    ph_dtimes.append(ph_dtime)
                    # weights.append(ph_w)

                else:
                    print('\nTravel time exceeded travel time limit.')
                    print(f'Specified travel time limit: {tt_cut} s.')
                    print('Event data:')
                    print('{}\n{}\n{} {} {} {} {} {} {} {}\n'.format(
                        event.ID, ph_station, ph_year, ph_month, ph_day,
                        ph_hour, ph_min, ph_sec, ph_msec, ph_rtime))

            # ph_data = zip(stas, arrts, phases, uncs, weights)
            ph_data = list(zip(stas, arrts, phases, uncs, evids, tphs,
                ph_dtimes))

            for sta in set(stations):
                # keep only phases of a particular station
                phdf = list(filter(lambda x: x[0] == sta, ph_data))

                # continue to next station if only one phase is left
                if len(phdf) > 1:
                    # sort by arrival times, still contains all phases
                    phdf = sorted(phdf, key=lambda x: x[1])
                    # extract phase of first arrival at current station
                    # it should be P arrival
                    fa_php = phdf[0][2]
                else:
                    continue

                # check if this really is P arrival and extract values
                if fa_php in list(phase_pairs):
                    ptt = phdf[0][1]
                    punc = phdf[0][3]
                    # pw = phdf[0][4]
                    phs = phase_pairs[fa_php]
                    # extract S phase
                    phdfs = list(filter(lambda x: x[2] == phs, phdf))
                else:
                    continue

                if len(phdfs) == 1:
                    stt = phdfs[0][1]
                    sunc = phdfs[0][3]
                    # sw = phdfs[0][4]
                    stph = phdfs[0][5]
                    sptt = stt - ptt
                    ptts.append(ptt)
                    puncs.append(punc)
                    # pws.append(pw)
                    stts.append(stt)
                    suncs.append(sunc)
                    # sws.append(sw)
                    sptts.append(sptt)
                    oterrs.append(ev_eot)
                    eventids.append(event.ID)
                    stationacryms.append(sta)

                    if not sp_exists:
                        event.phase_lines['Phase_ID'].append('SP')
                        event.phase_lines['Stat_name'].append(sta)
                        event.phase_lines['Uncert_ind'].append(sunc)
                        # add phase time and travel time of S phase
                        event.phase_lines['tph'].append(stph)
                        event.phase_lines['tt'].append(sptt)
                        event.phase_lines['Pick_hour'].append(stph.hour)
                        event.phase_lines['Pick_minutes'].append(stph.minute)
                        event.phase_lines['Pick_seconds'].append(
                            stph.second + stph.microsecond * 10**-6)

                        currlen = len(event.phase_lines['Phase_ID'])

                        for ph_key in list(event.phase_lines):
                            if len(event.phase_lines[ph_key]) < currlen:
                                event.phase_lines[ph_key].append(None)

                        event.sp_lines['Phase_ID'].append('SP')
                        event.sp_lines['SP_time'].append(sptt)
                        event.sp_lines['Stat_name'].append(sta)
                        event.sp_lines['Uncert_ind'].append(sunc)
                        # add phase time and travel time of S phase
                        event.sp_lines['tph'].append(stph)
                        event.sp_lines['tt'].append(phdfs[0][6])

                        event.sp_lines[sta] = {}
                        event.sp_lines[sta]['Uncert_ind'] = sunc
                        event.sp_lines[sta]['SP_time'] = sptt
                        # add phase time and travel time of S phase
                        event.sp_lines[sta]['tph'] = stph
                        event.sp_lines[sta]['tt'] = phdfs[0][6]
                    else:
                        pass

                else:
                    continue

        return ptts, puncs, stts, suncs, sptts, oterrs, eventids, stationacryms

    def assign_picks(self, picks_obj, events=None):
        if not events:
            events = self.events
        else:
            pass

        pick_list = sorted(picks_obj.picks[:], key=lambda x: x.tpick,
            reverse=True)
        pick_stas = np.array([x.statID for x in pick_list])
        pick_times = np.array([x.tpick_nord for x in pick_list])
        pick_phts = np.array([x.phase for x in pick_list])

        for ie, event in enumerate(sorted(
            events, key=lambda x: x.tev, reverse=True)):
            print(f'{ie}/{len(events)}', event, event.ID)
            event.phase_lines['Uncert_time'] = []
            event.phase_lines['Channel'] = []
            event.phase_lines['pick_obj'] = []
            stas = event.phase_lines['Stat_name']

            for iph, ph_sta in enumerate(stas):
                tph = event.phase_lines['tph'][iph]
                ph_ID = event.phase_lines['Phase_ID'][iph]

                # ista = np.argwhere(pick_stas == ph_sta)
                # itime = np.argwhere(pick_times == tph)
                # ipht = np.argwhere(pick_phts == ph_ID)

                ista = np.where(pick_stas == ph_sta)
                itime = np.where(pick_times == tph)
                ipht = np.where(pick_phts == ph_ID)

                # sel_pick = list(filter(
                #     lambda x: ph_sta == x.statID and tph == x.tpick_nord and ph_ID == x.phase,
                #     pick_list))

                iinter = reduce(np.intersect1d, (ista, itime, ipht))

                if iinter.size > 0:
                # if len(sel_pick) > 0:
                    sel_pick = pick_list.pop(iinter[0])
                    # sel_pick = sel_pick[0]
                    # pick_list.remove(sel_pick)
                    # print(len(pick_list))
                    pick_stas = np.delete(pick_stas, iinter[0])
                    pick_times = np.delete(pick_times, iinter[0])
                    pick_phts = np.delete(pick_phts, iinter[0])
                    event.phase_lines['Uncert_time'].append(sel_pick.unctime)
                    event.phase_lines['Channel'].append(sel_pick.channel)
                    event.phase_lines['pick_obj'].append(sel_pick)
                else:
                    event.phase_lines['Uncert_time'].append(None)
                    event.phase_lines['Channel'].append(None)
                    event.phase_lines['pick_obj'].append(None)

        # for event in sorted(self.events, key=lambda x: x.tev, reverse=True):
        #     print(event, event.ID)
        #     event.assign_picks(picks_obj)

    def add_noise(self, events=None):
        from numpy.random import default_rng

        rng = default_rng()

        unc_dict = {0: (0.0, 0.1), 1: (0.1, 0.2), 2: (0.2, 0.5), 3: (0.5, 1.0),
            4: (1.0, 1.0 * 10**9)}

        if not events:
            events = self.events
        else:
            pass

        addn = []
        aaddn = []

        for event in events:
            ph_uncs = event.phase_lines['Uncert_ind']

            tphs = []

            for iph, ph_unc in enumerate(ph_uncs):
                # tph = event.phase_lines['tph'][iph]
                tph = event.phase_lines['tt'][iph]
                unc_int = unc_dict[ph_unc]
                std = unc_int[1]
                noise = rng.normal(0.0, std)
                # print(unc_int, noise)

                while not unc_int[0] <= abs(noise) < unc_int[1]:
                    noise = rng.normal(0.0, std)
                    # print('IN', noise)

                # print('OUT', noise)
                tph = tph + timedelta(seconds=noise)
                tphs.append(tph)
                addn.append(noise)
                aaddn.append(abs(noise))

            # event.phase_lines['tph'] = tphs
            event.phase_lines['tt'] = tphs

        print(f'Average of added noise: {np.average(addn)}.')
        print(f'Absolute average of added noise: {np.average(aaddn)}.\n')


class NordicBlocks(object):

    def __init__(self):
        self.input_file = None
        self.input_lines = []
        self.event_IDs = []
        self.event_blocks = {}

    def open_nordic(self, input_nordic_file):
        with open(input_nordic_file) as input_data:
            for line in input_data:
                self.input_lines.append(line)
        input_data.close()

    # def split_events(self):
    #     """Format and write event IDs, find last line, separate events
    #     based on blank lines."""
    #     year = int(self.input_lines[0][1:5])
    #     month = int(self.input_lines[0][6:8])
    #     day = int(self.input_lines[0][8:10])
    #     hour = int(self.input_lines[0][11:13])
    #     minute = int(self.input_lines[0][13:15])
    #     sec = round(float(self.input_lines[0][16:20]), 1)
    #     event_ID = "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(
    #         year, month, day, hour, minute, int(sec)
    #     )
    #     # event_ID = "{:04d}{:02d}{:02d}{:02d}{:02d}{:03d}".format(
    #     #     year, month, day, hour, minute, int(str(sec).replace('.', ''))
    #     # )
    #     event_block = []

    #     for line_ind, line in enumerate(self.input_lines):

    #         line_num = line_ind + 1

    #         if line.strip().replace(' ', '') == "" and \
    #             line_num < len(self.input_lines):
    #             event_block = []
    #             year = int(self.input_lines[line_num][1:5])
    #             month = int(self.input_lines[line_num][6:8])
    #             day = int(self.input_lines[line_num][8:10])
    #             hour = int(self.input_lines[line_num][11:13])
    #             minute = int(self.input_lines[line_num][13:15])
    #             sec = round(float(self.input_lines[line_num][16:20]), 1)
    #             event_ID = "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(
    #                 year, month, day, hour, minute, int(sec)
    #             )
    #             # event_ID = "{:04d}{:02d}{:02d}{:02d}{:02d}{:03d}".format(
    #             #     year, month, day, hour, minute,
    #             #     int(str(sec).replace('.', ''))
    #             # )
    #         elif line.strip().replace(' ', '') == "" and \
    #             line_num == len(self.input_lines):
    #             break
    #         else:
    #             event_block.append(line.strip('\r\n'))
    #             self.event_blocks[event_ID] = event_block
    #             self.event_IDs.append(event_ID)

    # def parse_events(self):
    #     event_db = EventDatabase()

    #     for event_ID in self.event_blocks:
    #         event_block = self.event_blocks[event_ID]
    #         event = Event_Nordic(event_ID, event_block)
    #         event._parse_params()
    #         event_db.add_event(event)

    #     return event_db

    def parse_events(self):
        """Format and write event IDs, find last line, separate events
        based on blank lines and parse them to database."""

        event_db = EventDatabase()

        event_block = []

        for line_ind, line in enumerate(self.input_lines):

            line_num = line_ind + 1

            if not line.strip().replace(' ', '') and \
                line_num < len(self.input_lines):
                event = Event_Nordic(event_block)
                event._parse_params()
                event_db.add_event(event)
                self.event_blocks[event.ID] = event_block
                self.event_IDs.append(event.ID)

                event_block = []

            elif not line.strip().replace(' ', '') and \
                line_num == len(self.input_lines):
                event = Event_Nordic(event_block)
                event._parse_params()
                event_db.add_event(event)
                self.event_blocks[event.ID] = event_block
                self.event_IDs.append(event.ID)
                break

            else:
                event_block.append(line.strip('\r\n'))

        return event_db


class SeiStat(object):

    def __init__(self, station_line, header=None, velest=False, maxchar=5):
        self.station_line = station_line
        self.header = header
        self.params = {}
        self.statID = None

        if header and not velest:
            """Header column order may be arbitrary, but their names should
            match the default ones: ACRYM, LAT, LON, NTWRK."""

            for col_index, col_name in enumerate(self.header):
                param = self.station_line[col_index]

                try:
                    param = int(param)
                except ValueError:
                    try:
                        param = float(param)
                    except ValueError:
                        pass

                self.params[col_name] = param

            if maxchar < 5:
                self.params['ACRYM'] = self.params['ACRYM'][:maxchar]
            else:
                pass

            self.statID = self.params['ACRYM']

        elif velest:
            self.params['ACRYM'] = station_line[0]
            self.params['LAT'] = float(station_line[1][:-1])
            self.params['LON'] = float(station_line[2][:-1])
            self.params['ELEVATION'] = float(station_line[3])
            self.params['NTWRK'] = None
            self.statID = self.params['ACRYM']
            self.params['PDELAY'] = float(station_line[6])
            self.params['SDELAY'] = float(station_line[7])
            self.params['PRDNS'] = int(station_line[8])
            # self.params['SRDNS'] = int(station_line[9])
            # self.params['SRDNS'] = None
            try:
                self.params['SRDNS'] = int(station_line[9])
            except:
                self.params['SRDNS'] = 0

        else:
            if maxchar < 5:
                self.params['ACRYM'] = station_line[0][:maxchar]
            else:
                self.params['ACRYM'] = station_line[0]

            self.params['LAT'] = float(station_line[1])
            self.params['LON'] = float(station_line[2])
            self.params['NTWRK'] = station_line[3]
            self.statID = self.params['ACRYM']


class SeiStations(object):

    def __init__(self, input_station_file, delimiter=None, header=False,
        velest=False, maxchar=5):
        self.input_file = input_station_file
        self.input_lines = []
        self.header_line = None
        self.header_data = None
        self.stations = []
        self.IDs = []
        self.networks = []
        self.lats = []
        self.lons = []
        self.pdelays = []
        self.param_dict = {}

        with open(input_station_file) as input_data:
            for line in input_data:
                if not line.startswith('#'):
                    self.input_lines.append(line)
                else:
                    pass
        input_data.close()

        if header or velest:
            self.header_line = self.input_lines[0]
            self.header_data = self.header_line.strip().split(delimiter)
            self.input_lines = self.input_lines[1:]

            if velest:
                self.input_lines = self.input_lines[:-2]
            else:
                pass

        else:
            pass

        for line in filter(lambda x: x.strip(), self.input_lines):
            if delimiter:
                prep_line = line.strip().split(delimiter)
            else:
                prep_line = line.strip().split()

            stat_obj = SeiStat(
                prep_line, self.header_data, velest=velest, maxchar=maxchar
            )

            self.stations.append(stat_obj)
            self.IDs.append(stat_obj.statID)
            self.networks.append(stat_obj.params['NTWRK'])
            self.lats.append(stat_obj.params['LAT'])
            self.lons.append(stat_obj.params['LON'])
            self.param_dict[stat_obj.statID] = stat_obj.params

            if velest:
                self.pdelays.append(stat_obj.params['PDELAY'])
            else:
                pass

    def reload_lists(self):
        self.IDs = []
        self.networks = []
        self.lats = []
        self.lons = []
        self.param_dict = {}

        for stat_obj in self.stations:
            self.IDs.append(stat_obj.statID)
            self.networks.append(stat_obj.params['NTWRK'])
            self.lats.append(stat_obj.params['LAT'])
            self.lons.append(stat_obj.params['LON'])
            self.param_dict[stat_obj.statID] = stat_obj.params

    def keep_observed(self, events):
        all_obsta = []

        for event in events:
            all_obsta.extend(event.return_stations())

        all_obsta = set(all_obsta)

        self.obstations = list(filter(
            lambda x: x in all_obsta, self.stations
        ))

    def return_extent(self, stations):
        return [
            min(self.lons), min(self.lats), max(self.lons), max(self.lats)
        ]

    def filter_by_distance(self, opoint, distance):
        pass

    def filter_by_extent(self, extent):
        # Pass extent as (UL, UR, LL, LR).

        lons, lats = list(zip(*extent))
        min_lon = min(lons)
        max_lon = max(lons)
        min_lat = min(lats)
        max_lat = max(lats)

        stations_in = []

        for sta in self.stations:
            if min_lon <= sta.params['LON'] <= max_lon and \
                min_lat <= sta.params['LAT'] <= max_lat:
                stations_in.append(sta)

        self.stations = stations_in

        self.reload_lists()


class Pick(object):

    def __init__(self, pick_line, stamaxchar=5):
        self.pick_line = pick_line
        self.channel = pick_line[1]
        self.phase = pick_line[2]
        self.date = pick_line[3].split('/')
        self.month = int(self.date[0])
        self.day = int(self.date[1])
        self.year = int(self.date[2])
        self.time = pick_line[4].split(':')
        self.hour = int(self.time[0])
        self.minutes = int(self.time[1])
        self.seconds = float(self.time[2])
        self.dsecs, self.secs = math.modf(self.seconds)
        self.secs = int(round(self.secs))
        self.msecs = int(round(self.dsecs * 10**6))
        self.unctime = float(pick_line[5])

        if stamaxchar < 5:
            self.statID = pick_line[0][:stamaxchar]
        else:
            self.statID = pick_line[0]

        self.tpick = datetime(self.year, self.month, self.day, self.hour,
            self.minutes, self.secs, self.msecs)

        # print(self.dsecs)
        # print(round(self.dsecs, 2))
        # print(round(self.dsecs, 2) * 10**6)
        # print(round(round(self.dsecs, 2) * 10**6))

        self.msecsr = int(round(round(self.dsecs, 2) * 10**6))

        if self.msecsr == 1000000:
            self.tpick_nord = datetime(self.year, self.month, self.day, self.hour,
                self.minutes, self.secs) + timedelta(seconds=1.0)
        else:
            self.tpick_nord = datetime(self.year, self.month, self.day, self.hour,
                self.minutes, self.secs, self.msecsr)


class Picks(object):

    def __init__(self, input_pick_file, delimiter='', stamaxchar=5):
        self.input_file = input_pick_file
        self.input_lines = []
        self.header_line = None
        self.header_data = None
        self.picks = []

        with open(input_pick_file) as input_data:
            for line in input_data:
                    self.input_lines.append(line)

        input_data.close()

        for line in filter(lambda x: x.strip(), self.input_lines):
            if delimiter:
                prep_line = line.strip().split(delimiter)
            else:
                prep_line = line.strip().split()

            pick_obj = Pick(prep_line, stamaxchar=stamaxchar)

            self.picks.append(pick_obj)


# anomaly in input file for event 2017  827  5 04 54.7DD, redundant whitespace
# between hour and minutes column
# therefore implement check event formatting warning per parameter
# "%02d"
# phase uncertainty 9? inacurate clock

# TO-DO
##############################################################################
# DONE dictionary filtering, filter event IDs based on parameter values
# DONE filtering, based on phase lines add 'lines_phase' parameter
# DONE afterwards remove None values for choosen parameters
#
# plotting
#
# DONE parsing filtered events to traxcerpt and/or output file
# DONE separate callable script
# DONE seismogram time window based on origin time and maximum of arrival times
# DONE list of only selected stations per event or all stations
# possibility to zip all lines for choosen phase parameter for each station
# count lines of type 1 and parse them to dictionary - other sources
##############################################################################

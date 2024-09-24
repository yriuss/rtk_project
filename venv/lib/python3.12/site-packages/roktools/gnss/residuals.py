import datetime
from typing import List, Tuple, Dict

import pandas as pd

CONSTELLATION_FIELD_STR = 'constellation'
SAT_FIELD_STR = 'sat'
EPOCH_FIELD_STR = 'epoch'
AZIMUTH_FIELD_STR = 'az'
ELEVATION_FIELD_STR = 'el'
SNR_FIELD_STR = 'snr_dbHz'


class Residuals(object):

    def __init__(self, dataframe):
        self.df = dataframe

    def get_sat_visibility(self) -> Dict[str, List[Tuple[float, float]]]:

        out = {}

        for sat, group in self.df.groupby(SAT_FIELD_STR):
            # FIXME this should take into account channel to avoid overwriting the info from different channels
            out[sat] = list(zip(group[AZIMUTH_FIELD_STR], group[ELEVATION_FIELD_STR], group[SNR_FIELD_STR]))

        return out

    def get_constellation_count(self) -> List[Tuple[datetime.datetime, dict]]:

        n_sats = []

        for epoch, epoch_group in self.df.groupby(EPOCH_FIELD_STR):
            n_sats_per_epoch = {}
            for constellation, df in epoch_group.groupby(CONSTELLATION_FIELD_STR):
                n_sats_per_epoch[constellation] = len(pd.unique(df[SAT_FIELD_STR]))

            n_sats.append((epoch.to_pydatetime(), n_sats_per_epoch))

        return n_sats

    def __len__(self):
        return len(self.df)

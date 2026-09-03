"""Checked conversion between VIMS band numbers and array positions."""

from titan_limb.models.core import Channel

FIRST_BAND = 1
LAST_VISIBLE_BAND = 96
FIRST_INFRARED_BAND = 97
LAST_BAND = 352


def band_to_index(band: int) -> int:
    if band < FIRST_BAND or band > LAST_BAND:
        raise ValueError(f"VIMS band must be between {FIRST_BAND} and {LAST_BAND}")
    return band - FIRST_BAND


def index_to_band(index: int) -> int:
    if index < 0 or index >= LAST_BAND:
        raise ValueError(f"VIMS band index must be between 0 and {LAST_BAND - 1}")
    return index + FIRST_BAND


def channel_for_band(band: int) -> Channel:
    band_to_index(band)
    if band <= LAST_VISIBLE_BAND:
        return Channel.VISIBLE
    return Channel.INFRARED

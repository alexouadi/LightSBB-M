"""Unpaired transport baselines compared against LightSBB-M on ALAE latents."""

from .neural_ot import NOT
from .otcfm import OTCFM
from .sf2m import SF2M

BASELINES = {
    OTCFM.name: OTCFM,
    NOT.name: NOT,
    SF2M.name: SF2M,
}

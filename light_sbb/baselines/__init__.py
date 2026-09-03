"""Unpaired transport baselines compared against LightSBB-M on ALAE latents."""

from .lightsb import LightSB
from .lightsbb import LightSBB
from .neural_ot import NOT
from .otcfm import OTCFM
from .sf2m import SF2M

BASELINES = {
    OTCFM.name: OTCFM,
    NOT.name: NOT,
    SF2M.name: SF2M,
    LightSBB.name: LightSBB,
    LightSB.name: LightSB,
}

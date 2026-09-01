"""Unpaired transport baselines compared against LightSBB-M on ALAE latents."""

from .neural_ot import NOT
from .otcfm import OTCFM

BASELINES = {
    OTCFM.name: OTCFM,
    NOT.name: NOT,
}

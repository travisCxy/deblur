from .native import Augment


def tia_distort(src, segment=4):
    return Augment.GenerateDistort(src, segment)


def tia_stretch(src, segment=4):
    return Augment.GenerateStretch(src, segment)


def tia_perspective(src):
    return Augment.GeneratePerspective(src)
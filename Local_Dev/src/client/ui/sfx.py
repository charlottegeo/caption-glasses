import pygame

from client.theme import SFX_CATEGORY_STYLES, SFX_DEFAULT_STYLE, sfx_font

_icon_cache: dict[tuple[str, tuple], pygame.Surface] = {}

#draw sfx icons eventually
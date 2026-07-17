import pygame
from client import theme
from client.sfx_categories import category_icon_char, normalize_sfx_category
from client.state import state

_chip_cache: dict[tuple, pygame.Surface] = {}

CHIP_PAD_X = 8
CHIP_PAD_Y = 4
CHIP_GAP = 8
CHIP_ICON_GAP = 6

def _style_for(category: str | None) -> dict:
    cat = normalize_sfx_category(category)
    return theme.SFX_CATEGORY_STYLES.get(cat, theme.SFX_DEFAULT_STYLE)

def _render_chip(label: str, category: str | None) -> pygame.Surface:
    cat = normalize_sfx_category(category)
    key = (label, cat, id(theme.sfx_font), id(theme.small_font))
    cached = _chip_cache.get(key)
    if cached is not None:
        return cached

    style = _style_for(cat)
    icon = category_icon_char(cat)
    icon_surf = theme.sfx_font.render(icon, True, style["fg"])
    text_surf = theme.small_font.render(label, True, style["fg"])
    width = CHIP_PAD_X * 2 + icon_surf.get_width() + CHIP_ICON_GAP + text_surf.get_width()
    height = max(icon_surf.get_height(), text_surf.get_height()) + CHIP_PAD_Y * 2
    chip = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(chip, style["bg"], chip.get_rect(), border_radius=6)
    chip.blit(icon_surf, (CHIP_PAD_X, (height - icon_surf.get_height()) // 2))
    chip.blit(
        text_surf,
        (
            CHIP_PAD_X + icon_surf.get_width() + CHIP_ICON_GAP,
            (height - text_surf.get_height()) // 2,
        ),
    )
    _chip_cache[key] = chip
    if len(_chip_cache) > 64:
        _chip_cache.clear()
        _chip_cache[key] = chip
    return chip

def visible_sound_labels() -> list[dict]:
    enabled = state.get("sfx_enabled_categories") or {}
    out = []
    for item in state.get("sound_labels") or []:
        cat = normalize_sfx_category(item.get("category"))
        if enabled.get(cat, True):
            out.append(item)
    return out

def draw_sound_labels(screen: pygame.Surface, x: int, y: int) -> None:
    labels = visible_sound_labels()
    if not labels:
        return
    cursor_x = x
    for item in labels:
        chip = _render_chip(str(item.get("label", "")), item.get("category"))
        screen.blit(chip, (cursor_x, y - chip.get_height()))
        cursor_x += chip.get_width() + CHIP_GAP
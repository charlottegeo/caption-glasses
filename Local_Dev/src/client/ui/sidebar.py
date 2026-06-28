import math
import time
import pygame

from client.state import _btn_last_click, state
from client.theme import hint_font, ui_font
from client.tuning_specs import (
    ALL_SLIDER_SPECS,
    BTN_DEBOUNCE_SEC,
    BTN_GAP,
    BTN_H,
    HEADER_H,
    MEDIA_SFX_DEFAULTS,
    SIDEBAR_PAD,
    SIDEBAR_W,
    SLIDER_H,
    SLIDER_SPECS,
    SFX_SLIDER_SPECS,
    YAMNET_PROFILES,
)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def slider_value(spec, ratio):
    raw = spec["min"] + clamp(ratio, 0.0, 1.0) * (spec["max"] - spec["min"])
    step = spec["step"]
    if step >= 1:
        return int(round(raw / step) * step)
    places = max(0, -int(round(math.log10(step))))
    return round(round(raw / step) * step, places)

def slider_ratio(spec, value):
    span = spec["max"] - spec["min"]
    return clamp((value - spec["min"]) / span, 0.0, 1.0) if span > 0 else 0.0

def fmt_val(spec, value):
    sfx = spec.get("suffix", "")
    if spec["fmt"] == "d":
        return f"{int(round(value))}{sfx}"
    return f"{value:{spec['fmt']}}{sfx}"

def btn_label(name):
    if name == "caption_translate":
        return "Mode: Translate" if state["translate_mode"] else "Mode: Captions"
    if name == "singing":
        return "Singing: ON" if state["singing_mode"] else "Singing: OFF"
    if name == "profanity":
        return "Profanity filter: ON" if state["profanity_filter"] else "Profanity filter: OFF"
    if name == "yamnet":
        return f"SFX profile: {YAMNET_PROFILES[state['yamnet_profile_index']]}"
    if name == "telemetry":
        return "Telemetry: ON" if state["caption_telemetry"] else "Telemetry: OFF"
    if name == "speakers":
        return "Speakers: ON" if state["show_speakers"] else "Speakers: OFF"
    if name == "reset_speech":
        return "Reset: speech defaults"
    if name == "reset_lyrics":
        return "Reset: singing defaults"
    return name

def sfx_section_y():
    return HEADER_H + 6 + len(SLIDER_SPECS) * SLIDER_H + 18

def sfx_slider_y(i):
    return sfx_section_y() + i * SLIDER_H

def buttons_y():
    return sfx_section_y() + len(SFX_SLIDER_SPECS) * SLIDER_H + 8

def sidebar_height():
    return buttons_y() + 8 * (BTN_H + BTN_GAP) + 8

def slider_y(i):
    return HEADER_H + 6 + i * SLIDER_H

def sidebar_x(w):
    return w - SIDEBAR_W

def btn_ready(name: str) -> bool:
    now = time.monotonic()
    if now - _btn_last_click.get(name, 0.0) < BTN_DEBOUNCE_SEC:
        return False
    _btn_last_click[name] = now
    return True

def draw_slider_row(screen, spec, row_y, track_w, x, active, colors):
    val = state["slider_values"][spec["key"]]
    ratio = slider_ratio(spec, val)
    val_txt = fmt_val(spec, val)
    val_surf = hint_font.render(val_txt, True, (160, 200, 160))
    label_max = track_w - val_surf.get_width() - 6
    label_txt = spec["label"]
    if ui_font.size(label_txt)[0] > label_max:
        while label_txt and ui_font.size(label_txt + "…")[0] > label_max:
            label_txt = label_txt[:-1]
        label_txt += "…"
    screen.blit(ui_font.render(label_txt, True, (225, 225, 225)), (x, row_y))
    screen.blit(val_surf, (x + track_w - val_surf.get_width(), row_y))
    screen.blit(hint_font.render(fit_hint(spec["hint"], track_w), True, (115, 115, 115)), (x, row_y + 17))
    track = pygame.Rect(x, row_y + 36, track_w, 8)
    pygame.draw.rect(screen, (50, 50, 50), track, border_radius=3)
    fill = max(2, int(track_w * ratio))
    idle, hot = colors
    pygame.draw.rect(
        screen,
        hot if active else idle,
        pygame.Rect(track.x, track.y, fill, track.height),
        border_radius=3,
    )
    pygame.draw.circle(screen, (235, 235, 235), (track.x + fill, track.centery), 5)

def slider_track_rect(sb_x, row_y, scroll, track_w):
    return pygame.Rect(sb_x + SIDEBAR_PAD, row_y - scroll + 36, track_w, 8)

def handle_slider_hit(event_pos, sb_x, scroll, track_w):
    for i, spec in enumerate(SLIDER_SPECS):
        row_y = slider_y(i)
        track = slider_track_rect(sb_x, row_y, scroll, track_w)
        if track.collidepoint(event_pos):
            ratio = (event_pos[0] - track.x) / max(1, track.width)
            state["slider_values"][spec["key"]] = slider_value(spec, ratio)
            return spec["key"]
    for i, spec in enumerate(SFX_SLIDER_SPECS):
        row_y = sfx_slider_y(i)
        track = slider_track_rect(sb_x, row_y, scroll, track_w)
        if track.collidepoint(event_pos):
            ratio = (event_pos[0] - track.x) / max(1, track.width)
            state["slider_values"][spec["key"]] = slider_value(spec, ratio)
            return spec["key"]
    return None

def slider_track_for_key(key, sb_x, scroll, track_w):
    spec = next(s for s in ALL_SLIDER_SPECS if s["key"] == key)
    if spec in SLIDER_SPECS:
        i = SLIDER_SPECS.index(spec)
        return slider_track_rect(sb_x, slider_y(i), scroll, track_w), spec
    i = SFX_SLIDER_SPECS.index(spec)
    return slider_track_rect(sb_x, sfx_slider_y(i), scroll, track_w), spec

def button_color(name, on, hovered):
    if on and name == "caption_translate":
        col = (45, 130, 55)
    elif on and name == "singing":
        col = (110, 70, 150)
    elif on and name == "speakers":
        col = (70, 95, 130)
    elif on and name == "profanity":
        col = (150, 55, 55)
    elif on and name == "telemetry":
        col = (50, 80, 120)
    elif name.startswith("reset_"):
        col = (72, 72, 82)
    else:
        col = (62, 62, 62)
    if hovered:
        col = tuple(min(255, c + 24) for c in col)
    return col
"""Read and flip the Windows apps light/dark setting, for reproducing a live theme flip.

Qt derives ``QStyleHints.colorScheme()`` on Windows from one registry value,
``HKCU\\...\\Themes\\Personalize\\AppsUseLightTheme``, and emits ``colorSchemeChanged`` when it
changes: so writing that value is a *real* theme flip as far as the app is concerned, not a
simulation. The scheduled flips users hit in the wild (PowerToys *Light Switch*, Auto Dark
Mode) work exactly this way too.

Writing the value is only half of what the Settings app does, though, and the missing half turns
out to matter: see :func:`broadcast_theme_change`. Flips from here broadcast by default, so they
arrive the same way a hand-toggle in Settings does.

Why this exists rather than clicking Settings → Personalization → Colors: the Windows Server
image used for build/test has no Personalization page at all, and the live theme-flip crash
(see the ``_ThemeWatcher`` docstring in ``vodparse.gui.common``) only happens on a live flip.
Used by the main GUI's ``Debug → Flip OS dark mode now…`` action.

Windows-only and Qt-free: ``winreg`` is importable nowhere else, so every function raises
:class:`OSError` off Windows instead of pretending to work.
"""

import sys

# The key Windows stores the light/dark preference under, and the two values inside it. Qt reads
# APPS_VALUE; SYSTEM_VALUE is the taskbar/Start chrome, included because a real user flip in the
# Settings app moves both and the difference is worth being able to test.
THEME_KEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
APPS_VALUE = "AppsUseLightTheme"
SYSTEM_VALUE = "SystemUsesLightTheme"


def is_supported() -> bool:
    """True on Windows, where these functions work at all."""
    return sys.platform == "win32"


def _unsupported() -> OSError:
    """The error every accessor raises off Windows, where there is no such OS setting."""
    return OSError(f"the Windows theme registry setting does not exist on {sys.platform}")


def apps_use_light_theme() -> bool:
    """The current setting: True for light, False for dark.

    Defaults to True (light) when the value or its key is missing, matching how Windows itself
    treats an absent value: a fresh Server image genuinely has neither.
    """
    # Written as a literal sys.platform comparison rather than via is_supported(): that is the
    # form mypy understands as a platform guard, so the Windows-only winreg stubs below are only
    # type-checked on Windows (and the raise is what makes the guard true at runtime too).
    if sys.platform != "win32":
        raise _unsupported()
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, THEME_KEY) as key:
            value, _type = winreg.QueryValueEx(key, APPS_VALUE)
    except FileNotFoundError:
        return True
    return bool(value)


# What the Settings app broadcasts after writing the value: WM_SETTINGCHANGE carrying the string
# "ImmersiveColorSet", to every top-level window. Sent, not posted, so each window handles it
# inside its own window procedure — which is the difference that matters here (see broadcast_theme_change).
_HWND_BROADCAST = 0xFFFF
_WM_SETTINGCHANGE = 0x001A
_SMTO_ABORTIFHUNG = 0x0002
_BROADCAST_TIMEOUT_MS = 1000
_IMMERSIVE_COLOR_SET = "ImmersiveColorSet"


def broadcast_theme_change() -> None:
    """Broadcast the theme change the way the Settings app does, after writing the value.

    **This is the part a bare registry write leaves out, and it is not cosmetic.** Writing the
    value alone still reaches Qt — it watches the key on its own thread — so the app re-themes
    either way, which is why this was originally thought unnecessary. But the two arrive by
    completely different routes: a watched-key notification lands on Qt's own thread and is
    delivered on a clean stack, whereas this broadcast is *sent* (not posted), so every window
    handles it **inside its window procedure**, and anything Qt does in response — including
    emitting ``colorSchemeChanged`` — runs on that callback stack.

    Reproducing the theme-flip crash needs that second route: a hand toggle in the Settings app
    killed the app where a registry-only flip did not, and a whole day of bisection ran against
    the weaker stimulus before this was noticed.
    """
    if sys.platform != "win32":  # literal comparison for mypy's benefit; see above
        raise _unsupported()
    import ctypes

    result = ctypes.c_ulong()
    ctypes.windll.user32.SendMessageTimeoutW(
        _HWND_BROADCAST, _WM_SETTINGCHANGE, 0, ctypes.c_wchar_p(_IMMERSIVE_COLOR_SET),
        _SMTO_ABORTIFHUNG, _BROADCAST_TIMEOUT_MS, ctypes.byref(result))


def set_apps_use_light_theme(light: bool, *, also_system: bool = False,
                             broadcast: bool = True) -> None:
    """Write the apps light/dark setting (and optionally the system chrome one).

    Creates the key when missing, which is what makes this work on a stock Windows Server image.
    ``broadcast`` then announces the change as the Settings app does; leave it on unless you are
    deliberately testing the difference, since without it the app re-themes by a different (and
    apparently safer) route — see :func:`broadcast_theme_change`.
    """
    if sys.platform != "win32":  # literal comparison for mypy's benefit; see above
        raise _unsupported()
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, THEME_KEY, 0,
                            winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, APPS_VALUE, 0, winreg.REG_DWORD, int(light))
        if also_system:
            winreg.SetValueEx(key, SYSTEM_VALUE, 0, winreg.REG_DWORD, int(light))
    if broadcast:
        broadcast_theme_change()


def flip_apps_theme(*, also_system: bool = False, broadcast: bool = True) -> bool:
    """Toggle the setting and return the new state (True = light). See the module docstring."""
    light = not apps_use_light_theme()
    set_apps_use_light_theme(light, also_system=also_system, broadcast=broadcast)
    return light

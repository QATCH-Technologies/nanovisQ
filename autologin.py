"""
nanovisq_autologin.py
---------------------
Watches for the QATCH nanovisQ Real-Time GUI window and automatically
fills in the Initials + Password fields, then clicks Sign In.

Requirements:
    pip install pywinauto pyautogui

Usage:
    1. Fill in your credentials in the CONFIG block below.
    2. Run this script BEFORE or AFTER launching nanovisQ — it will
       wait up to TIMEOUT_SEC seconds for the window to appear.
    3. python nanovisq_autologin.py
"""

import time
import sys

# ── CONFIG ────────────────────────────────────────────────────────────────────
USERNAME = "PM"
PASSWORD = "jjjjjjjj"
TIMEOUT_SEC = 60
POLL_INTERVAL = 2
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_TITLE_PATTERN = "QATCH nanovisQ Real-Time GUI"


def find_nanovisq_window():
    """Return a pywinauto WindowSpecification if the window exists, else None."""
    from pywinauto import Desktop

    try:
        windows = Desktop(backend="uia").windows()
        for w in windows:
            try:
                title = w.window_text()
                if WINDOW_TITLE_PATTERN in title:
                    return w
            except Exception:
                continue
    except Exception:
        pass
    return None


def autologin(window):
    """Fill credentials and click Sign In."""
    import pyautogui
    import pywinauto

    print("[*] nanovisQ window detected. Bringing to foreground ...")
    try:
        window.set_focus()
    except Exception:
        pass
    time.sleep(0.5)

    # ── Strategy 1: UIA control tree (most reliable for Qt) ──────────────────
    try:
        print("[*] Trying UIA control search ...")
        # Find the Initials edit box
        initials_ctrl = window.child_window(control_type="Edit", found_index=0)
        initials_ctrl.set_focus()
        initials_ctrl.set_edit_text(USERNAME)
        time.sleep(0.2)

        # Find the Password edit box
        password_ctrl = window.child_window(control_type="Edit", found_index=1)
        password_ctrl.set_focus()
        password_ctrl.set_edit_text(PASSWORD)
        time.sleep(0.2)

        # Find and click the Sign In button
        sign_in_btn = window.child_window(title="Sign In", control_type="Button")
        sign_in_btn.click()
        print("[+] Sign In clicked via UIA. Done!")
        return True

    except Exception as e:
        print(f"[!] UIA strategy failed ({e}), falling back to pyautogui ...")

    # ── Strategy 2: pyautogui image-based / coordinate fallback ──────────────
    try:
        rect = window.rectangle()
        win_left = rect.left
        win_top = rect.top
        win_width = rect.width()
        win_height = rect.height()

        # Approximate positions derived from the screenshot layout
        # Initials field is roughly at 46% from left, 38% from top
        initials_x = win_left + int(win_width * 0.46)
        initials_y = win_top + int(win_height * 0.385)

        password_x = win_left + int(win_width * 0.46)
        password_y = win_top + int(win_height * 0.410)

        sign_in_x = win_left + int(win_width * 0.46)
        sign_in_y = win_top + int(win_height * 0.437)

        # Fill Initials
        pyautogui.click(initials_x, initials_y)
        time.sleep(0.2)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.typewrite(USERNAME, interval=0.05)

        # Fill Password
        pyautogui.click(password_x, password_y)
        time.sleep(0.2)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.typewrite(PASSWORD, interval=0.05)

        # Click Sign In
        time.sleep(0.2)
        pyautogui.click(sign_in_x, sign_in_y)
        print("[+] Sign In clicked via pyautogui. Done!")
        return True

    except Exception as e:
        print(f"[!] pyautogui strategy also failed: {e}")
        return False


def main():
    if USERNAME == "YOUR_INITIALS" or PASSWORD == "YOUR_PASSWORD":
        print("[!] Please edit USERNAME and PASSWORD in the CONFIG block before running.")
        sys.exit(1)

    print(f"[*] Waiting up to {TIMEOUT_SEC}s for '{WINDOW_TITLE_PATTERN}' ...")
    deadline = time.time() + TIMEOUT_SEC

    while time.time() < deadline:
        window = find_nanovisq_window()
        if window:
            # Give the UI a moment to finish rendering the login dialog
            time.sleep(1.5)
            success = autologin(window)
            sys.exit(0 if success else 1)

        remaining = int(deadline - time.time())
        print(f"    Window not found yet — {remaining}s remaining ...", end="\r")
        time.sleep(POLL_INTERVAL)

    print(f"\n[!] Timed out after {TIMEOUT_SEC}s. Is nanovisQ running?")
    sys.exit(1)


if __name__ == "__main__":
    main()

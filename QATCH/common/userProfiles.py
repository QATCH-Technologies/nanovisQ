import os
import sys
import hashlib
import datetime as dt
from xml.dom import minidom
from enum import Enum
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QDateTime
from QATCH.common.logger import Logger as Log
from QATCH.common.fileManager import FileManager
from QATCH.common.fileStorage import FileStorage
from QATCH.core.constants import Constants, UserRoles
from QATCH.ui.widgets.create_user_widget import CreateUserWidget
from typing import Union

TAG = "[UserProfiles]"


class UserConstants:
    # Specify how long until dev mode expires
    DEV_EXPIRE_LEN = 365  # days
    # DevMode will remain active across multiple SW versions, without requiring re-activation
    DEV_PERSIST_VER = True
    REQ_ADMIN_UPDATES = False  # set dynamically as well

    try:
        if os.path.isfile(Constants.user_constants_path):
            with open(Constants.user_constants_path, "r") as uc:
                uc_file_contents = uc.read()
            uc_settings = {}
            exec(uc_file_contents, uc_settings)
            REQ_ADMIN_UPDATES = uc_settings["REQ_ADMIN_UPDATES"]
    except:
        Log.e("Failed to import user constants file. Using defaults.")


###############################################################################
# User Profiles: init, create, modify, destroy
###############################################################################
class UserProfiles:
    user_preferences = None
    PATH = Constants.user_profiles_path

    ###########################################################################
    # Creates logging file (.txt)
    ###########################################################################
    @staticmethod
    def count():
        FileManager.create_dir(UserProfiles.PATH)
        user_files = os.listdir(UserProfiles.PATH)
        user_files = [x for x in user_files if ".xml" in x]  # remove folders
        if len(user_files) == 0:
            Log.d("No user profiles found.")
        return len(user_files)

    @staticmethod
    def get_all_user_info():
        FileManager.create_dir(UserProfiles.PATH)
        user_files = os.listdir(UserProfiles.PATH)
        user_files = [x for x in user_files if ".xml" in x]  # remove folders
        user_info = []
        for filename in user_files:
            user_info.append(UserProfiles.get_user_info(filename))
        return user_files, user_info

    @staticmethod
    def get_user_info(filename):
        if filename is None:
            filename = ""
        file = os.path.join(UserProfiles.PATH, filename)
        if os.path.isfile(file):
            xml = minidom.parse(file)
            up = xml.getElementsByTagName("secure_user_info")
            for u in up:
                name = u.getAttribute("name") if u.hasAttribute("name") else "[Missing]"
                initials = u.getAttribute("initials") if u.hasAttribute("initials") else "[Missing]"
                p = u.getAttribute("password") if u.hasAttribute("password") else "[Missing]"
                s = u.getAttribute("signature") if u.hasAttribute("signature") else "[Missing]"
                try:
                    # allow exception if missing
                    role = UserRoles(int(u.getAttribute("role")))
                except:
                    role = UserRoles.INVALID

            ts = xml.getElementsByTagName("timestamp")
            now = dt.datetime.now().isoformat()
            today = now[0 : now.find("T") + 1]
            created = "[NEVER]"
            modified = "[NEVER]"
            accessed = "[NEVER]"
            for t in ts:
                x = t.getAttribute("type") if t.hasAttribute("type") else "[Missing]"
                y = t.getAttribute("value") if t.hasAttribute("value") else "[Missing]"
                if y != "[Missing]":
                    y = y[0 : y.find(".")]  # remove subseconds
                # Log.w(f"processing ts: {x}, {y}")
                if y.startswith(today):
                    y = y.replace(today, "Today, ")
                else:
                    y = y.replace("T", " ")
                if x == "created":
                    created = y
                    modified = y  # set both
                if x == "modified":
                    modified = y
                if x == "accessed":
                    accessed = y

            return [name, initials, role.name, created, modified, accessed]
        else:
            return [None, None, None, None, None, None]

    @staticmethod
    def create_new_user():
        Log.w("Prompting to create a new user...")

        existing_initials = []
        _files, user_info_list = UserProfiles.get_all_user_info()
        for info in user_info_list:
            if info and len(info) > 1 and info[1] not in (None, "[Missing]"):
                existing_initials.append(info[1])

        parent_window = QtWidgets.QApplication.activeWindow()
        widget = CreateUserWidget(existing_initials, parent=parent_window)
        widget.show()

        loop = QtCore.QEventLoop()
        widget.destroyed.connect(loop.quit)
        loop.exec_()

        if widget.is_accepted:
            data = widget.result_data

            UserProfiles.create(data["name"], data["initials"], data["role"], data["password"])
        else:
            Log.w("User creation cancelled.")

    @staticmethod
    def change(requiredRole=UserRoles.ANY):
        if UserProfiles.count() == 0:
            return None, None, 0

        title = (
            "Sign In" if requiredRole.value == UserRoles.ANY.value else requiredRole.name.title()
        )

        ok = False
        while not ok:
            initials, ok = QtWidgets.QInputDialog.getText(
                None, title, "Initials:", QtWidgets.QLineEdit.Normal
            )
            if not ok:
                return None, None, 0
            initials = initials.upper()
            if initials.find(" ") >= 0 or len(initials) < 2 or len(initials) > 4:
                Log.w("Please enter your initials.")
                ok = False

        ok = False
        while not ok:
            pwd, ok = QtWidgets.QInputDialog.getText(
                None, title, "Password:", QtWidgets.QLineEdit.Password
            )
            if not ok:
                return None, None, 0
            if len(pwd) < 8:
                Log.w("Passwords must be at least 8 characters. Please try again.")
                ok = False

        authenticated, filename, params = UserProfiles.auth(initials, pwd, requiredRole)
        if authenticated:
            Log.i(f"Welcome, {params[0]}! Your assigned role is {params[2].name}.")
            return params[0], params[1], params[2].value
        else:
            return None, None, 0

    @staticmethod
    def change_password(filename):
        title = "Change Password"

        Log.w("Changing password...")
        user_info = UserProfiles.get_user_info(filename)
        initials = user_info[1]

        file = os.path.join(UserProfiles.PATH, filename)

        doc = minidom.parse(file)
        xml = doc.documentElement
        up = doc.getElementsByTagName("secure_user_info")

        for u in up:
            try:
                # allow exception if missing
                role = UserRoles(int(u.getAttribute("role")))
            except:
                role = UserRoles.INVALID
            password = u.getAttribute("password") if u.hasAttribute("password") else "[Missing]"
            initials = u.getAttribute("initials") if u.hasAttribute("password") else "[Missing]"
            name = u.getAttribute("name") if u.hasAttribute("name") else "[Missing]"
            if u.attributes["password"].value.find("X") < 0:
                # invalidate password hash, but allow recovery for audits
                u.attributes["password"].value += "X"

        ok = False
        while not ok:
            current_password, ok = QtWidgets.QInputDialog.getText(
                None, title, "Current Password:", QtWidgets.QLineEdit.Password
            )
            if not ok:
                return
            if len(current_password) < 8:
                Log.w("Passwords must be at least 8 characters. Please try again.")
                ok = False

        authenticated, filename, params = UserProfiles.auth(
            initials, current_password, UserRoles.ANY
        )
        if not authenticated:
            Log.e("User did not authenticate action to change their password. Aborted.")
            return

        match = False
        while not match:
            ok = False
            while not ok:
                pwd1, ok = QtWidgets.QInputDialog.getText(
                    None, title, "New Password:", QtWidgets.QLineEdit.Password
                )
                if not ok:
                    return
                if len(pwd1) < 8:
                    Log.w("Passwords must be at least 8 characters. Please try again.")
                    ok = False

            ok = False
            while not ok:
                pwd2, ok = QtWidgets.QInputDialog.getText(
                    None, title, "Confirm Password:", QtWidgets.QLineEdit.Password
                )
                if not ok:
                    return
                if len(pwd2) < 8:
                    Log.w("Passwords must be at least 8 characters. Please try again.")
                    ok = False

            match = True if pwd1 == pwd2 else False
            if not match:
                Log.w("Passwords entered do not match. Please try again.")
        pwd = pwd1

        # recalculate security hashes and put in user profile
        salt = filename[:-4]
        if True:  # recalculate hash for new password
            hash = hashlib.sha256()
            hash.update(salt.encode())
            hash.update(pwd.encode())
            password = hash.hexdigest()
        hash = hashlib.sha256()
        hash.update(salt.encode())
        hash.update(name.encode())
        hash.update(initials.encode())
        hash.update(role.name.encode())
        hash.update(password.encode())
        signature = hash.hexdigest()

        info = doc.createElement("secure_user_info")
        xml.appendChild(info)
        info.setAttribute("name", name)
        info.setAttribute("initials", initials)
        info.setAttribute("role", str(role.value))
        info.setAttribute("password", password)
        info.setAttribute("signature", signature)

        ts_type = "modified"
        ts_val = dt.datetime.now().isoformat()
        hash = hashlib.sha256()
        hash.update(salt.encode())
        hash.update(ts_type.encode())
        hash.update(ts_val.encode())
        signature = hash.hexdigest()

        ts1 = doc.createElement("timestamp")
        xml.appendChild(ts1)
        ts1.setAttribute("type", ts_type)
        ts1.setAttribute("value", ts_val)
        ts1.setAttribute("signature", signature)

        # append new secure_user_info to xml
        try:
            with open(file, "w") as f:
                xml_str = doc.toxml(encoding="ascii").decode(encoding="utf-8", errors="ignore")
                f.write(xml_str)
                Log.d(f"Saved XML file: {file}")
        except OSError as ose:  # FileNotFoundError
            Log.e(f"Error writing '{ts_type}' record: {file}")
            Log.e("Error Details:", ose.strerror)
            return
        except UnicodeError as ue:  # UnicodeEncodeError, UnicodeDecodeError
            Log.e(f"Unicode error writing XML: {file}")
            Log.e("Error Details:", ue.reason)
            return

        Log.w("Password changed: " + ("*" * len(pwd)))

    @staticmethod
    def session_create(salt):
        file = os.path.join(UserProfiles.PATH, "session.key")
        today = dt.datetime.now().isoformat().split("T")[0]
        hash = hashlib.sha256()
        hash.update(salt.encode())
        hash.update(today.encode())
        session_key = hash.hexdigest()
        try:
            with open(file, "w") as f:
                f.write(session_key)
                Log.d("User session created.")
        except OSError as ose:  # FileNotFoundError
            Log.e(f"Error writing session key: {ose}")
        UserProfiles.user_preferences = UserPreferences(UserProfiles.get_session_file())
        UserProfiles.user_preferences.set_preferences()
        # print(UserProfiles.user_preferences.get_preferences())
        # print(UserProfiles.user_preferences.get_folder_save_path(
        #     device_id=12345678, port_id=1))
        # print(UserProfiles.user_preferences.get_file_save_path(
        #     device_id=12345678, port_id=1))

    @staticmethod
    def session_info():
        file = os.path.join(UserProfiles.PATH, "session.key")
        today = dt.datetime.now().isoformat().split("T")[0]
        if os.path.exists(file):
            files, infos = UserProfiles.get_all_user_info()
            try:
                with open(file, "r") as f:
                    session_key = f.read()
                    for i, file in enumerate(files):
                        salt = file[:-4]
                        hash = hashlib.sha256()
                        hash.update(salt.encode())
                        hash.update(today.encode())
                        file_key = hash.hexdigest()
                        if file_key == session_key:
                            Log.d("User session is active.")
                            return True, infos[i]  # valid session
            except OSError as ose:  # FileNotFoundError
                Log.e(f"Error reading session key: {ose}")
            Log.d("User session is expired.")
            return False, None  # invalid session
        else:
            Log.d("User session is NOT active.")
            return False, None  # no active session

    @staticmethod
    def get_session_file() -> Union[str, None]:
        """
        Reports the session file name to the caller.

        Given the file from the session key and todays date, the method
        returns the encrypted name of the user session file. If there is
        no active user session or the user session has expired, the method
        returns None.

        Args:
            None

        Returns:
            str: The encrypted string specifying the user file.
            None: If there is no active user or the user session has expired.

        Logs:
            Logs debug information if the user session has expired or is not
            active.
        """
        file = os.path.join(UserProfiles.PATH, "session.key")
        today = dt.datetime.now().isoformat().split("T")[0]
        if os.path.exists(file):
            files, infos = UserProfiles.get_all_user_info()
            try:
                with open(file, "r") as f:
                    session_key = f.read()
                    for i, file in enumerate(files):
                        salt = file[:-4]
                        hash = hashlib.sha256()
                        hash.update(salt.encode())
                        hash.update(today.encode())
                        file_key = hash.hexdigest()
                        if file_key == session_key:
                            Log.d("User session is active.")
                            file = file.split(".xml")[0]
                            return file  # valid session
            except OSError as ose:  # FileNotFoundError
                Log.e(f"Error reading session key: {ose}")
            Log.d("User session is expired.")
            return None  # invalid session
        else:
            Log.d("User session is NOT active.")
            return None  # no active session

    @staticmethod
    def session_end():
        file = os.path.join(UserProfiles.PATH, "session.key")
        if os.path.exists(file):
            Log.d("User session ended.")
            os.remove(file)
        if hasattr(UserProfiles.user_preferences, "_user_preferences_path"):
            delattr(UserProfiles.user_preferences, "_user_preferences_path")

    @staticmethod
    def manage(existingUserName, existingUserRole):
        allow_manage = False
        admin_name = None
        if UserProfiles.count() == 0:
            UserProfiles.create_new_user(UserRoles.ADMIN)
            allow_manage = UserProfiles.count() == 1
            if allow_manage:
                user_files, user_info = UserProfiles.get_all_user_info()
                admin_name = user_info[0][0]
        else:
            # require auth from role 'admin'
            if UserProfiles.check(existingUserRole, UserRoles.ADMIN):
                Log.d("Current User has required ADMIN role")
                admin_name = existingUserName
            else:
                if existingUserRole != UserRoles.NONE:
                    Log.w("Current User does not have required ADMIN role")
                Log.w("Please sign in with ADMIN account to make changes.")
                admin_name, _, _ = UserProfiles.change(UserRoles.ADMIN)
            # Log.d(f"admin is '{admin_name}'")
            if admin_name != None:
                allow_manage = True

        return allow_manage, admin_name

    @staticmethod
    def find(name, initials):
        user_files = os.listdir(UserProfiles.PATH)
        user_files = [x for x in user_files if ".xml" in x]  # remove folders
        for filename in user_files:
            file = os.path.join(UserProfiles.PATH, filename)
            if os.path.isfile(file):
                xml = minidom.parse(file)
                up = xml.getElementsByTagName("secure_user_info")
                for u in up:
                    n = u.getAttribute("name") if u.hasAttribute("name") else "[Missing]"
                    i = u.getAttribute("initials") if u.hasAttribute("initials") else "[Missing]"
                if n == name or i == initials:  # only check most recent secure_user_info record
                    return True, filename
        return False, None

    @staticmethod
    def create(name, initials, role, pwd):
        found, filename = UserProfiles.find(name, initials)
        sign_in_user = UserProfiles.count() == 0
        if not found:
            Log.i(
                f"Create user, Name: {name}, initials: {initials}, role: {role.name} password: {'*'*len(pwd)}"
            )
            while True:
                salt = os.urandom(8).hex()
                file = os.path.join(UserProfiles.PATH, f"{salt}.xml")
                if not os.path.exists(file):
                    break
            hash = hashlib.sha256()
            hash.update(salt.encode())
            hash.update(pwd.encode())
            password = hash.hexdigest()
            hash = hashlib.sha256()
            hash.update(salt.encode())
            hash.update(name.encode())
            hash.update(initials.encode())
            hash.update(role.name.encode())
            hash.update(password.encode())
            signature = hash.hexdigest()
            Log.d(f"salt = {salt}, hash = {signature}")

            doc = minidom.Document()
            xml = doc.createElement("user_profile")
            doc.appendChild(xml)

            info = doc.createElement("secure_user_info")
            xml.appendChild(info)
            info.setAttribute("name", name)
            info.setAttribute("initials", initials)
            info.setAttribute("role", str(role.value))
            info.setAttribute("password", password)
            info.setAttribute("signature", signature)

            ts_type = "created"
            ts_val = dt.datetime.now().isoformat()
            hash = hashlib.sha256()
            hash.update(salt.encode())
            hash.update(ts_type.encode())
            hash.update(ts_val.encode())
            signature = hash.hexdigest()

            ts1 = doc.createElement("timestamp")
            xml.appendChild(ts1)
            ts1.setAttribute("type", ts_type)
            ts1.setAttribute("value", ts_val)
            ts1.setAttribute("signature", signature)
            # ts2 = doc.createElement('timestamp')
            # xml.appendChild(ts2)
            # ts2.setAttribute('name', "modified")
            # ts2.setAttribute('value', ts)

            # Log.d(doc)
            try:
                with open(file, "w") as f:
                    xml_str = doc.toxml(encoding="ascii").decode(encoding="utf-8", errors="ignore")
                    f.write(xml_str)
                    Log.d(f"Saved XML file: {file}")
            except OSError as ose:  # FileNotFoundError
                Log.e(f"Error writing '{ts_type}' record: {file}")
                Log.e("Error Details:", ose.strerror)
                return
            except UnicodeError as ue:  # UnicodeEncodeError, UnicodeDecodeError
                Log.e(f"Unicode error writing XML: {file}")
                Log.e("Error Details:", ue.reason)
                return

            if sign_in_user:
                # create session
                UserProfiles.auth(initials, pwd, UserRoles.ADMIN)

        else:
            Log.e(
                f"Failed to create user. User info conflicts with user profile '{filename[:-4]}'."
            )

    ### check() ###
    # RETURNS: One of: [None, False, True]
    # None when: Users exist AND one of: 1) no one signed in, 2) session is not valid, 3) session is valid, but conflicted
    # False when: User is signed in but not authorized for the required role
    # True when: 1) No users in system, or 2) user signed in AND authorized for role
    @staticmethod
    def check(userRole, requiredRole):
        if UserProfiles().count() == 0:
            Log.d("No user profiles. Skipping user check.")
            return True
        if userRole == UserRoles.NONE:
            Log.d("User must sign in prior to user check.")
            return None
        valid, infos = UserProfiles().session_info()
        if not valid:
            Log.w("User session is not valid. Please sign in again.")
            return None
        if valid and userRole.name != infos[2]:
            Log.w("User session is conflicted. Please sign in again.")
            return None
        try:
            maskedRole = UserRoles(userRole.value & requiredRole.value)
        except:
            maskedRole = UserRoles.INVALID
        result = True if maskedRole == requiredRole else False
        Log.d(
            f"User role check: user={userRole.name}, required={requiredRole.name}, result={result}"
        )
        return result

    @staticmethod
    def auth(initials, pwd, requiredRole=UserRoles.ANY):
        found, filename = UserProfiles.find(None, initials)
        if found:
            file = os.path.join(UserProfiles.PATH, filename)
            if os.path.isfile(file):
                xml = minidom.parse(file)
                up = xml.getElementsByTagName("secure_user_info")
                for u in up:
                    name = u.getAttribute("name") if u.hasAttribute("name") else "[Missing]"
                    initials = (
                        u.getAttribute("initials") if u.hasAttribute("initials") else "[Missing]"
                    )
                    p = u.getAttribute("password") if u.hasAttribute("password") else "[Missing]"
                    s = u.getAttribute("signature") if u.hasAttribute("signature") else "[Missing]"

                    try:
                        # allow exception if missing
                        role = UserRoles(int(u.getAttribute("role")))
                    except:
                        role = UserRoles.INVALID

                UserProfiles.session_create(filename[:-4])
                # only check most recent secure_user_info record
                if not UserProfiles.check(role, requiredRole):
                    Log.e(
                        f"User {initials} does not have the required {requiredRole.name} role privileges."
                    )
                    UserProfiles.session_end()
                    return False, filename, None
                salt = filename[:-4]
                hash = hashlib.sha256()
                hash.update(salt.encode())
                hash.update(pwd.encode())
                password = hash.hexdigest()
                hash = hashlib.sha256()
                hash.update(salt.encode())
                hash.update(name.encode())
                hash.update(initials.encode())
                hash.update(role.name.encode())
                # password hash from file, not what the user entered
                hash.update(p.encode())
                signature = hash.hexdigest()

                if p == password and s == signature:
                    ts_type = "accessed"
                    ts_val = dt.datetime.now().isoformat()
                    hash = hashlib.sha256()
                    hash.update(salt.encode())
                    hash.update(ts_type.encode())
                    hash.update(ts_val.encode())
                    signature = hash.hexdigest()
                    try:
                        with open(file, "rb+") as f:
                            f.seek(-15, 2)
                            f.write(
                                f'<timestamp type="{ts_type}" value="{ts_val}" signature="{signature}"/></user_profile>'.encode()
                            )
                            # f.write(f'</user_profile>\r\n'.encode())
                    except OSError as ose:  # FileNotFoundError
                        Log.e(f"Error writing '{ts_type}' record: {file}")
                        Log.e("Error Details:", ose.strerror)
                        return False, filename, None
                    except UnicodeError as ue:  # UnicodeEncodeError, UnicodeDecodeError
                        Log.e(f"Unicode error writing XML: {file}")
                        Log.e("Error Details:", ue.reason)
                        return False, filename, None
                    Log.d(f"User {initials} authenticated successfully.")
                    return True, filename, [name, initials, role]
                elif p != password and s == signature:
                    Log.e(f"Auth failure: Invalid credentials.")
                    Log.d(
                        f"Auth failure details: user={initials}, p={p == password}, s={s == signature}"
                    )
                    UserProfiles.session_end()
                    return False, filename, None
                else:  # signature not valid
                    Log.e(f"Auth failure: Corrupt user profile {initials}. See an administrator.")
                    Log.e(
                        f"File security checks indicate your profile is invalid or had unauthorized changes made to it."
                    )
                    Log.d(
                        f"Auth failure details: file={filename}, p={p == password}, s={s == signature}"
                    )
                    UserProfiles.session_end()
                    return False, filename, None
            else:
                Log.e(f"Auth failure: Invalid user account.")
                Log.d(f"Auth failure details: User profile is not a file! Not found: {file}")
                UserProfiles.session_end()
                return False, filename, None
        else:
            Log.e(f"Auth failure: Invalid credentials.")
            Log.d(f"Auth failure details: User {initials} not found")
            UserProfiles.session_end()
            return False, None, None

    @staticmethod
    def checkDevMode():
        enabled = False
        is_error = False
        expires_at = ""
        build_date = ""
        try:
            dev_path = os.path.join(Constants.local_app_data_path, ".dev_mode")
            if os.path.exists(dev_path):
                with open(dev_path, "r") as dev:
                    try:
                        encode_str1 = dev.readline()
                        encode_str2 = dev.readline()
                        encode_key = "DEADBEEFDEADBEEFDEAD"
                        hexify_str1 = bytes(
                            [(ord(a) ^ ord(b)) for a, b in zip(encode_str1, encode_key)]
                        ).decode()
                        hexify_str2 = bytes(
                            [(ord(a) ^ ord(b)) for a, b in zip(encode_str2, encode_key)]
                        ).decode()
                        time_stamp = bytearray.fromhex(hexify_str1).decode()
                        build_date = bytearray.fromhex(hexify_str2).decode()
                        expires_on = dt.datetime.fromisoformat(time_stamp)
                        expires_at = str(expires_on.date())
                    except ValueError:
                        expires_on = "invalid"
                    except:
                        expires_on = "exception"

                    now = dt.datetime.now()
                    if expires_on == "invalid" or expires_on == "exception":
                        Log.e(
                            f"Developer Mode encoded expiration value could not be parsed ({expires_on})! Please renew or disable."
                        )
                        is_error = True
                    elif build_date != Constants.app_date and not UserConstants.DEV_PERSIST_VER:
                        Log.e(
                            "Developer Mode was enabled for another SW version and is invalid here. Please renew or disable."
                        )
                        is_error = True
                        expires_at = ""
                    elif expires_on > now - dt.timedelta(days=1):
                        if expires_on < now + dt.timedelta(days=UserConstants.DEV_EXPIRE_LEN):
                            enabled = True
                        else:
                            Log.e(
                                f"Developer Mode expiration date ({expires_at}) is invalid! Please renew or disable."
                            )
                            is_error = True
                            expires_at = ""
                    else:
                        Log.e(f"Developer Mode expired on {expires_at}. Please renew or disable.")
                        is_error = True
            else:
                pass
        except:
            Log.e("Error checking Developer Mode")
            is_error = True
            expires_at = ""

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        return enabled, is_error, expires_at


class UserPreferences:
    """Manages user preferences based on a user session.

    This class is responsible for initializing the user preferences for a given
    session by setting the user session key and performing any additional setup
    required for managing user-specific configurations.

    Attributes:
        _user_session (str): The session key associated with the user.
    """

    def __init__(self, user_session_key: str) -> None:
        """Initializes a new instance of UserPreferences.

        This constructor sets the user session key by calling the internal
        `_set_user_session` method and then calls `setup` to perform further
        initialization of user preferences.

        Args:
            user_session_key (str):  A users session key as a string.
        """
        self._set_user_session(user_session_key)
        self.setup()

    def setup(self) -> None:
        """
        Sets up the user and global preferences for the application.

        This function performs the following steps:
        1. Retrieves the current user session information.
        2. Constructs file paths for user-specific and global preferences based on the application's local data path.
        3. Ensures that the directories for these paths exist; creates them if necessary.
        4. Checks for an existing global preferences file:
            - If it exists, sets the global preferences path.
            - If it does not exist, logs an error, writes the default global preferences file, and then sets the path.
        5. Checks for an existing user preferences file:
            - If it exists, logs that user preferences will be used and sets the path, disabling global preferences.
            - If it does not exist and user session information is available, logs a warning, writes the default user preferences file, and sets the path.
            - If user session information is not available, logs a warning that user preferences cannot be created.

        Raises:
            Exception: Propagates any exception encountered during session retrieval, directory creation,
                    or writing default preferences.
        """
        user_info = self._get_user_session()

        # Build file paths for user and global preferences
        user_preferences_path = os.path.join(
            Constants.local_app_data_path, "profiles/users", f"{user_info}-preferences.json"
        )
        global_preferences_path = os.path.join(
            Constants.local_app_data_path, "global_preferences.json"
        )

        # Ensure the directories for preferences exist
        try:
            os.makedirs(os.path.dirname(user_preferences_path), exist_ok=True)
        except Exception as e:
            Log.e(TAG, f"Error creating user preferences directory: {e}")
            raise

        try:
            os.makedirs(os.path.dirname(global_preferences_path), exist_ok=True)
        except Exception as e:
            Log.e(TAG, f"Error creating global preferences directory: {e}")
            raise

        # Start with global preferences enabled
        self.set_use_global(True)

        # Check if the global preferences file exists
        if os.path.exists(global_preferences_path):
            self._set_global_preferences_path(global_preferences_path)
        else:
            Log.e(TAG, "No global file format preferences found. Writing global and using.")
            try:
                FileStorage.DEV_write_default_preferences(global_preferences_path)
            except Exception as e:
                Log.e(TAG, f"Error writing default global preferences: {e}")
                raise
            self._set_global_preferences_path(global_preferences_path)

        # Check if the user preferences file exists
        if os.path.exists(user_preferences_path):
            Log.d(TAG, "Using User Preferences")
            self._set_user_preferences_path(user_preferences_path)
            self.set_use_global(False)
        elif user_info is not None:
            Log.w(TAG, "Creating User Preferences from default format and using global.")
            try:
                FileStorage.DEV_write_default_preferences(user_preferences_path)
            except Exception as e:
                Log.e(TAG, f"Error writing default user preferences: {e}")
                raise
            self._set_user_preferences_path(user_preferences_path)
        else:
            Log.w(TAG, "User session information is not available. User preferences not created.")

    def set_preferences(self) -> None:
        """
        Loads and applies user or global preferences from a JSON file.

        This method performs the following steps:
        1. Determines whether to use the global or user preferences file.
        2. Opens and loads the preferences JSON file.
        3. Validates that all required keys are present in the loaded JSON data.
        4. Extracts the necessary configuration parameters and applies them using
            the corresponding setter methods.

        Raises:
            FileNotFoundError: If the preferences file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            KeyError: If any required preference key is missing.
            Exception: For any other errors that occur during file I/O or setting preferences.
        """
        import json

        preferences_data = None

        # Attempt to open and load the correct preferences file
        try:
            if self.get_use_global():
                global_path = self._get_global_preferences_path()
                with open(global_path, "r") as preferences_file:
                    preferences_data = json.load(preferences_file)
            else:
                user_path = self._get_user_preferences_path()
                with open(user_path, "r") as preferences_file:
                    preferences_data = json.load(preferences_file)
        except FileNotFoundError as fnf_err:
            Log.e(TAG, f"Preferences file not found: {fnf_err}")
            raise
        except json.JSONDecodeError as json_err:
            Log.e(TAG, f"Error decoding JSON from preferences file: {json_err}")
            raise
        except Exception as e:
            Log.e(TAG, f"Unexpected error loading preferences: {e}")
            raise

        # Validate required keys in the JSON data
        required_keys = [
            "load_data_path",
            "write_data_path",
            "folder_format",
            "filename_format",
            "folder_format_delimiter",
            "filename_format_delimiter",
            "date_format",
            "time_format",
        ]
        for key in required_keys:
            if key not in preferences_data:
                error_msg = f"Missing required preference key: {key}"
                Log.e(TAG, error_msg)
                raise KeyError(error_msg)

        # Parse the preferences and apply them using the corresponding setters
        try:
            load_data_path = str(preferences_data["load_data_path"])
            write_data_path = str(preferences_data["write_data_path"])
            folder_tag_format = str(preferences_data["folder_format"])
            file_tag_format = str(preferences_data["filename_format"])
            folder_delimiter = str(preferences_data["folder_format_delimiter"])
            filename_delimiter = str(preferences_data["filename_format_delimiter"])
            date_format = str(preferences_data["date_format"])
            time_format = str(preferences_data["time_format"])

            self._set_load_data_path(load_data_path)
            self._set_write_data_path(write_data_path)
            self._set_folder_delimiter(folder_delimiter)
            self._set_folder_format_pattern(folder_tag_format)
            self._set_file_delimiter(filename_delimiter)
            self._set_file_format_pattern(file_tag_format)
            self._set_date_format(date_format)
            self._set_time_format(time_format)
        except Exception as e:
            Log.e(TAG, f"Error applying preferences: {e}")
            raise

    def get_preferences(self) -> dict:
        """
        Retrieves the current preferences settings.

        This method collects the current configuration for load and write data paths,
        file and folder formatting patterns, delimiters, and date/time formats via the
        respective accessor methods, and returns them as a dictionary.

        Returns:
            dict: A dictionary containing the following keys:
                - load_data_path
                - write_data_path
                - folder_format
                - filename_format
                - folder_format_delimiter
                - filename_format_delimiter
                - date_format
                - time_format

        Raises:
            Exception: If an error occurs while retrieving any of the preferences.
        """
        try:
            preferences_dict = {
                "load_data_path": self._get_load_data_path(),
                "write_data_path": self._get_write_data_path(),
                "folder_format": self._get_folder_format_pattern(),
                "filename_format": self._get_file_format_pattern(),
                "folder_format_delimiter": self._get_folder_delimiter(),
                "filename_format_delimiter": self._get_file_delimiter(),
                "date_format": self._get_date_format(),
                "time_format": self._get_time_format(),
            }
        except Exception as e:
            Log.e(TAG, f"Error retrieving preferences: {e}")
            raise

        return preferences_dict

    def load_user_preferences(self) -> dict:
        """
        Loads and returns the user preferences from the user preferences file.

        Returns:
            dict: The preferences data loaded from the user preferences file.

        Raises:
            FileNotFoundError: If the user preferences file does not exist.
            Exception: For any other errors encountered during loading.
        """
        try:
            user_preferences_path = self._get_user_preferences_path()
            preferences = FileStorage.DEV_load_preferences(user_preferences_path)
            return preferences
        except Exception as e:
            Log.e(TAG, f"Error loading user preferences: {e}")
            raise

    def load_global_preferences(self) -> dict:
        """
        Loads and returns the global preferences from the global preferences file.

        Returns:
            dict: The preferences data loaded from the global preferences file.

        Raises:
            FileNotFoundError: If the global preferences file does not exist.
            Exception: For any other errors encountered during loading.
        """
        try:
            global_preferences_path = self._get_global_preferences_path()
            preferences = FileStorage.DEV_load_preferences(global_preferences_path)
            return preferences
        except Exception as e:
            Log.e(TAG, f"Error loading global preferences: {e}")
            raise

    def reset_global_preferences(self) -> None:
        """
        Resets the global preferences to their default values by writing the default
        preferences to the global preferences file.

        Raises:
            Exception: If an error occurs while writing default preferences.
        """
        try:
            global_preferences_path = self._get_global_preferences_path()
            FileStorage.DEV_write_default_preferences(save_path=global_preferences_path)
        except Exception as e:
            Log.e(TAG, f"Error resetting global preferences: {e}")
            raise

    def write_global_preferences(self) -> None:
        """
        Writes the current preferences to the global preferences file.

        The current preferences are retrieved using get_preferences() and then saved.

        Raises:
            Exception: If an error occurs during writing the preferences.
        """
        try:
            global_preferences_path = self._get_global_preferences_path()
            preferences = self.get_preferences()
            FileStorage.DEV_write_preferences(
                save_path=global_preferences_path, preferences=preferences
            )
        except Exception as e:
            Log.e(TAG, f"Error writing global preferences: {e}")
            raise

    def reset_user_preferences(self) -> None:
        """
        Resets the user preferences to their default values by writing the default
        preferences to the user preferences file.

        Raises:
            Exception: If an error occurs while writing default preferences.
        """
        try:
            user_preferences_path = self._get_user_preferences_path()
            FileStorage.DEV_write_default_preferences(save_path=user_preferences_path)
        except Exception as e:
            Log.e(TAG, f"Error resetting user preferences: {e}")
            raise

    def write_user_preferences(self) -> None:
        """
        Writes the current preferences to the user preferences file.

        The current preferences are retrieved using get_preferences() and then saved.

        Raises:
            Exception: If an error occurs during writing the preferences.
        """
        try:
            user_preferences_path = self._get_user_preferences_path()
            preferences = self.get_preferences()
            FileStorage.DEV_write_preferences(
                save_path=user_preferences_path, preferences=preferences
            )
        except Exception as e:
            Log.e(TAG, f"Error writing user preferences: {e}")
            raise

    def get_folder_save_path(self, runname: str, device_id: int, port_id: int) -> str:
        """
        Constructs and returns the folder save path based on the folder format pattern.

        The folder format pattern is obtained and split using the configured delimiter,
        then passed to the internal _build_save_path method along with runname, device_id,
        and port_id.

        Args:
            runname (str): The name of the run.
            device_id (int): The identifier for the device.
            port_id (int): The identifier for the port.

        Returns:
            str: The constructed folder save path.

        Raises:
            Exception: If an error occurs while building the folder save path.
        """
        try:
            pattern = self._get_folder_format_pattern().split(self._get_folder_delimiter())
            delimiter = self._get_folder_delimiter()
            return self._build_save_path(
                pattern=pattern,
                delimiter=delimiter,
                runname=runname,
                device_id=device_id,
                port_id=port_id,
            )
        except Exception as e:
            Log.e(TAG, f"Error constructing folder save path: {e}")
            raise

    def get_file_save_path(self, runname: str, device_id: int, port_id: int) -> str:
        """
        Constructs and returns the file save path based on the file format pattern.

        The file format pattern is obtained and split using the configured delimiter,
        then passed to the internal _build_save_path method along with runname, device_id,
        and port_id.

        Args:
            runname (str): The name of the run.
            device_id (int): The identifier for the device.
            port_id (int): The identifier for the port.

        Returns:
            str: The constructed file save path.

        Raises:
            Exception: If an error occurs while building the file save path.
        """
        try:
            pattern = self._get_file_format_pattern().split(self._get_file_delimiter())
            delimiter = self._get_file_delimiter()
            return self._build_save_path(
                pattern=pattern,
                delimiter=delimiter,
                runname=runname,
                device_id=device_id,
                port_id=port_id,
            )
        except Exception as e:
            Log.e(TAG, f"Error constructing file save path: {e}")
            raise

    def set_use_global(self, use_global: bool) -> None:
        """
        Sets the flag indicating whether to use global preferences.

        Args:
            use_global (bool): True to use global preferences; False to use user preferences.

        Raises:
            TypeError: If the provided use_global value is not a boolean.
        """
        if not isinstance(use_global, bool):
            error_msg = f"use_global must be a boolean, got {type(use_global).__name__}"
            Log.e(TAG, error_msg)
            raise TypeError(error_msg)
        self.use_global = use_global

    def get_use_global(self) -> bool:
        """
        Retrieves the current setting that indicates whether global preferences are used.

        Returns:
            bool: True if global preferences are in use; False otherwise.

        Raises:
            AttributeError: If the 'use_global' attribute is not defined.
            TypeError: If the value of 'use_global' is not a boolean.
        """
        if not hasattr(self, "use_global"):
            error_msg = "The 'use_global' attribute is not set."
            Log.e(TAG, error_msg)
            raise AttributeError(error_msg)
        if not isinstance(self.use_global, bool):
            error_msg = f"The 'use_global' attribute should be a boolean, but got {type(self.use_global).__name__}."
            Log.e(TAG, error_msg)
            raise TypeError(error_msg)
        return self.use_global

    # -- Private Utilities -- #

    def _build_save_path(
        self, pattern: list, runname: str, delimiter: str, device_id: int, port_id: int
    ) -> str:
        save_path = ""

        for i, tag in enumerate(pattern):
            if tag == Constants.valid_tags[0]:
                save_path = save_path + self._on_username()
            elif tag == Constants.valid_tags[1]:
                save_path = save_path + self._on_initials()
            elif tag == Constants.valid_tags[2]:
                save_path = save_path + self._on_device(device_id)
            elif tag == Constants.valid_tags[3]:
                save_path = save_path + self._on_runname(runname)
            elif tag == Constants.valid_tags[4]:
                save_path = save_path + self._on_date()
            elif tag == Constants.valid_tags[5]:
                save_path = save_path + self._on_time()
            elif tag == Constants.valid_tags[6]:
                if port_id == 0:
                    Log.d(TAG, 'Single device does not use "Port" tag, skipping')
                    continue  # skip "Port" tag if single device
                save_path = save_path + self._on_port(port_id)
            elif tag == Constants.subfolder_field:
                # only add '/' for new folder, stripping any delimeters
                save_path = save_path.strip(delimiter) + "/"
                continue  # skip adding another delimeter
            elif tag == Constants.select_tag_prompt:
                Log.w(TAG, "Ignoring empty folder format tag pattern")
                continue  # skip adding another delimeter
            else:
                Log.e(TAG, f"Invalid folder format tag pattern: {tag}")
                raise ValueError("Invalid folder format tag pattern")
            if i < len(pattern) - 1:
                save_path = save_path + delimiter

        # Prevent path from starting/ending with '/' character
        save_path = save_path.strip(Constants.slash)

        return save_path

    def _on_username(self) -> str:
        is_valid, user_info = UserProfiles.session_info()
        if is_valid:
            username = user_info[0]
            return username
        else:
            Log.e(TAG, 'Invalid user session. Using username "Anonymous User".')
            return "Anonymous User"  # raise ValueError("Invalid user session")

    def _on_initials(self) -> str:
        is_valid, user_info = UserProfiles.session_info()
        if is_valid:
            initials = user_info[1]
            return initials
        else:
            Log.e(TAG, 'Invalid user session. Using initials "ANON".')
            return "ANON"  # raise ValueError("Invalid user session")

    def _on_device(self, device_id: int) -> str:
        return str(device_id)

    def _on_runname(self, runname: str) -> str:
        return str(runname)

    def _on_date(self) -> str:
        from datetime import datetime

        format_mapping = {
            "YYYY": "%Y",  # Year with century as a decimal number
            "MM": "%m",  # Month as a zero-padded decimal number
            "DD": "%d",  # Day of the month as a zero-padded decimal number
            # Hour (24-hour clock) as a zero-padded decimal number
            "hh": "%H",
            "mm": "%M",  # Minute as a zero-padded decimal number
            "ss": "%S",  # Second as a zero-padded decimal number
            "A": "%p",  # AM/PM (12-hour clock)
        }
        date_format = self._get_date_format()
        for key, value in format_mapping.items():
            date_format = date_format.replace(key, value)

        return datetime.now().strftime(date_format)

    def _on_time(self) -> str:
        return QDateTime.currentDateTime().toString(self._get_time_format())

    def _on_port(self, port_id: int) -> str:
        return self._portIDfromIndex(port_id)

    def _portIDfromIndex(self, pid):  # convert ASCII byte to character
        # For 4x1 system: expect pid 1-4, return "1" thru "4"
        # For 4x6 system: expect pid 0xA1-0xD6, return "A1" thru "D6"
        return hex(pid)[2:].upper()

    # -- ACCESSOR METHODS -- #

    def _set_user_session(self, user_session_key: str) -> None:
        self._user_session = user_session_key

    def _set_folder_format_pattern(self, folder_format_pattern: str) -> None:
        # for delimiter in Constants.path_delimiters:
        #     folder_format_pattern = folder_format_pattern.replace(
        #         delimiter, '|')
        delimeter = self._get_folder_delimiter()
        split_parts = folder_format_pattern.split(delimeter)
        tokens = [part for part in split_parts]
        self._folder_format_pattern = tokens

    def _set_file_format_pattern(self, file_format_pattern: str) -> None:
        # for delimiter in Constants.path_delimiters:
        #     file_format_pattern = file_format_pattern.replace(
        #         delimiter, '|')
        delimeter = self._get_file_delimiter()
        split_parts = file_format_pattern.split(delimeter)
        tokens = [part for part in split_parts]
        self._file_format_pattern = tokens

    def _set_folder_delimiter(self, folder_delimiter: str) -> None:
        self._folder_delimiter = folder_delimiter

    def _set_file_delimiter(self, file_delimiter: str) -> None:
        self._file_delimiter = file_delimiter

    def _set_date_format(self, date_format: str) -> None:
        self._date_format = date_format

    def _set_time_format(self, time_format: str) -> None:
        self._time_format = time_format

    def _set_user_preferences_path(self, user_preferences_path: str) -> None:
        self._user_preferences_path = user_preferences_path

    def _set_global_preferences_path(self, global_preferences_path: str) -> None:
        self._global_preferences_path = global_preferences_path

    def _set_load_data_path(self, load_data_path: str) -> None:
        self._load_data_path = load_data_path
        Constants.log_prefer_path = self._load_data_path

    def _set_write_data_path(self, write_data_path: str) -> None:
        self._write_data_path = write_data_path

    # -- MUTATOR METHODS -- #

    def _get_user_session(self) -> UserProfiles:
        return self._user_session

    def _get_folder_format_pattern(self) -> str:
        folder_format = ""
        for i, tok in enumerate(self._folder_format_pattern):
            folder_format = folder_format + tok
            if i < len(self._folder_format_pattern) - 1:
                folder_format = folder_format + self._get_folder_delimiter()

        return folder_format

    def _get_file_format_pattern(self) -> str:
        file_format = ""
        for i, tok in enumerate(self._file_format_pattern):
            file_format = file_format + tok
            if i < len(self._file_format_pattern) - 1:
                file_format = file_format + self._get_file_delimiter()

        return file_format

    def _get_folder_delimiter(self) -> str:
        return self._folder_delimiter

    def _get_file_delimiter(self) -> str:
        return self._file_delimiter

    def _get_time_format(self) -> str:
        return self._time_format

    def _get_date_format(self) -> str:
        return self._date_format

    def _get_user_preferences_path(self) -> str:
        return self._user_preferences_path

    def _get_global_preferences_path(self) -> str:
        return self._global_preferences_path

    def _get_load_data_path(self) -> str:
        return self._load_data_path

    def _get_write_data_path(self) -> str:
        return self._write_data_path

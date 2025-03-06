import os
import sys
import hashlib
import send2trash
import datetime as dt
from time import sleep
from xml.dom import minidom
from enum import Enum
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QDateTime
from QATCH.common.logger import Logger as Log
from QATCH.common.architecture import Architecture
from QATCH.common.fileManager import FileManager
from QATCH.common.fileStorage import FileStorage
from QATCH.core.constants import Constants
from QATCH.ui.popUp import PopUp
from typing import Union
import re
TAG = '[UserProfiles]'


class UserRoles(Enum):
    NONE = 0x00
    CAPTURE = 0x01
    ANALYZE = 0x10
    OPERATE = CAPTURE | ANALYZE
    ADMIN = 0xFF
    ANY = NONE
    INVALID = NONE


class UserConstants:
    # Specify how long until dev mode expires
    DEV_EXPIRE_LEN = 365  # days
    # DevMode will remain active across multiple SW versions, without requiring re-activation
    DEV_PERSIST_VER = True
    REQ_ADMIN_UPDATES = False  # set dynamically as well

    try:
        if os.path.isfile(Constants.user_constants_path):
            with open(Constants.user_constants_path, 'r') as uc:
                uc_file_contents = uc.read()
            uc_settings = {}
            exec(uc_file_contents, uc_settings)
            REQ_ADMIN_UPDATES = uc_settings['REQ_ADMIN_UPDATES']
    except:
        Log.e("Failed to import user constants file. Using defaults.")


class UserProfilesManager(QtWidgets.QWidget):

    class TableView(QtWidgets.QTableWidget):

        def __init__(self, data, *args):
            QtWidgets.QTableWidget.__init__(self, *args)
            self.setData(data)
            self.resizeColumnsToContents()
            self.resizeRowsToContents()

        def setData(self, data):
            self.data = data
            self.clear()
            horHeaders = []
            for n, key in enumerate(self.data.keys()):
                horHeaders.append(key)
                for m, item in enumerate(self.data[key]):
                    error_cell = False
                    if item.startswith("*") and item.endswith("*"):
                        item = item[1:-1]
                        error_cell = True
                    newitem = QtWidgets.QTableWidgetItem(item)
                    newitem.setFlags(QtCore.Qt.ItemIsEnabled)
                    if error_cell:
                        newitem.setForeground(
                            QtGui.QBrush(QtGui.QColor(255, 0, 0)))
                    self.setItem(m, n, newitem)
            self.setHorizontalHeaderLabels(horHeaders)
            header = self.horizontalHeader()
            # refactored for Python 3.11: was setResizeMode()
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
            header.setStretchLastSection(False)
            # for i in range(len(horHeaders)):
            #     header.setResizeMode(i, QtWidgets.QHeaderView.ResizeToContents
            #               if i < 3 else QtWidgets.QHeaderView.Stretch)

    def __init__(self, parent=None, admin_name=None):
        super(UserProfilesManager, self).__init__(None)
        self.parent = parent
        # self.admin = admin_name
        self.admin_file = UserProfiles.find(admin_name, None)[1]  # filename

        screen = QtWidgets.QDesktopWidget().availableGeometry()
        pct_width = 50
        pct_height = 50
        self.resize(int(screen.width()*pct_width/100),
                    int(screen.height()*pct_height/100))
        self.move(int(screen.width()*(100-pct_width)/200),
                  int(screen.height()*(100-pct_width)/200))

        self.layout = QtWidgets.QHBoxLayout(self)

        self.buttons = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel("User Actions:")
        self.button1 = QtWidgets.QPushButton("&Add User")
        self.button2 = QtWidgets.QPushButton("&Edit User")
        self.button3 = QtWidgets.QPushButton("&Delete User")
        self.button4 = QtWidgets.QPushButton("A&udit User")
        self.button5 = QtWidgets.QPushButton("&Refresh")
        self.bDeleteAll = QtWidgets.QPushButton("Delete All")
        self.buttons.addWidget(self.label)
        self.buttons.addWidget(self.button1)
        self.buttons.addWidget(self.button2)
        self.buttons.addWidget(self.button3)
        self.buttons.addWidget(self.button4)
        self.buttons.addWidget(self.button5)
        self.buttons.addStretch()
        self.buttons.addWidget(QtWidgets.QLabel("Bulk Actions:"))
        self.buttons.addWidget(self.bDeleteAll)
        self.buttons.addWidget(QtWidgets.QLabel(""))
        self.buttons.addWidget(QtWidgets.QLabel(""))

        self.button1.pressed.connect(self.add_user)
        self.button2.pressed.connect(self.edit_user)
        self.button3.pressed.connect(self.remove_user)
        self.button4.pressed.connect(self.audit_user)
        self.button5.pressed.connect(self.update_table_data)
        self.bDeleteAll.pressed.connect(self.delete_all_users)

        data, rows, cols = self.get_user_data()
        self.table = self.TableView(data, rows, cols)
        # self.table.resize(300, 300)

        self.sorting = QtWidgets.QHBoxLayout()
        self.sortLabel = QtWidgets.QLabel("Sort Users By:")
        self.sortBy = QtWidgets.QComboBox()
        self.sortBy.addItems(list(data.keys()))
        self.sortOrder = QtWidgets.QComboBox()
        self.sortOrder.addItems(["Ascending", "Descending"])
        self.sortNow = QtWidgets.QPushButton("Sort")
        self.sorting.addWidget(self.sortLabel)
        self.sorting.addWidget(self.sortBy)
        self.sorting.addWidget(self.sortOrder)
        self.sorting.addWidget(self.sortNow)

        self.developerLayout = QtWidgets.QHBoxLayout()
        self.developerModeChk = QtWidgets.QCheckBox(
            "Enable Developer Mode (stores data unencrypted) - NOT compliant with FDA 21 CFR Part 11")
        self.developerExpires = QtWidgets.QLabel("")

        enabled, error, expires = UserProfiles.checkDevMode()
        self.developerModeChk.setChecked(enabled)
        color = "red" if error else ("green" if enabled else "black")
        self.developerModeChk.setStyleSheet(f"color:{color};")
        self.developerExpires.setStyleSheet("QLabel {color:" + color + ";}")
        if expires == "":
            self.developerExpires.setText("")
        else:
            if enabled:
                self.developerExpires.setText(f"(Expires on {expires})")
            else:
                self.developerExpires.setText(f"(Expired on {expires})")

        self.developerModeChk.stateChanged.connect(self.toggleDevMode)
        self.developerLayout.addWidget(self.developerModeChk)
        self.developerLayout.addWidget(self.developerExpires)
        self.developerLayout.addStretch()

        self.reqAdminUpd_chkbox = QtWidgets.QCheckBox(
            "Require Administrative role to install SW/FW updates")
        self.reqAdminUpd_chkbox.setChecked(UserConstants.REQ_ADMIN_UPDATES)
        self.reqAdminUpd_chkbox.stateChanged.connect(
            self.toggleReqAdminUpdates)

        self.sortNow.pressed.connect(self.sort_user_table)
        self.sort_user_table()  # sort once now at load

        self.layout_r = QtWidgets.QVBoxLayout()
        self.layout_r.addWidget(self.table)
        self.layout_r.addLayout(self.sorting)
        self.layout_r.addLayout(self.developerLayout)
        self.layout_r.addWidget(self.reqAdminUpd_chkbox)

        # Add widgets to layout
        self.layout.addLayout(self.buttons)
        self.layout.addLayout(self.layout_r)
        # self.layout.addLayout(self.sorting)

        self.setLayout(self.layout)
        self.setWindowTitle("Manage Users")

    def toggleDevMode(self, arg):
        try:
            dev_path = os.path.join(Constants.local_app_data_path, ".dev_mode")
            if self.developerModeChk.isChecked():
                with open(dev_path, 'w') as dev:
                    expires_at = str(
                        (dt.datetime.now() + dt.timedelta(days=UserConstants.DEV_EXPIRE_LEN)).date())
                    hexify_str1 = expires_at.encode().hex()
                    hexify_str2 = Constants.app_date.encode().hex()
                    encode_key = "DEADBEEFDEADBEEFDEAD"
                    encode_str1 = bytes(
                        [(ord(a) ^ ord(b)) for a, b in zip(hexify_str1, encode_key)]).decode()
                    encode_str2 = bytes(
                        [(ord(a) ^ ord(b)) for a, b in zip(hexify_str2, encode_key)]).decode()
                    dev.write(encode_str1)  # expiration
                    dev.write('\n')
                    dev.write(encode_str2)  # app_date
                    self.developerModeChk.setStyleSheet("color:green;")
                    self.developerExpires.setStyleSheet(
                        "QLabel {color:green;}")
                    self.developerExpires.setText(f"(Expires on {expires_at})")
                    PopUp.information(self, "Developer Mode Status", "<b>Developer Mode: ENABLED.</b><br/>" +
                                      f"This feature will require renewal in {UserConstants.DEV_EXPIRE_LEN} days<br/>" +
                                      ("and applies only to this build version.<br/>" if not UserConstants.DEV_PERSIST_VER else
                                       "and it will be applied to all build versions<br/>" +
                                       "on this PC unless you explicitly turn it off.<br/>") +
                                      f"<i>Expires on {expires_at}</i>")
            else:
                if os.path.exists(dev_path):
                    os.remove(dev_path)
                    self.developerModeChk.setStyleSheet("color:black;")
                    self.developerExpires.setStyleSheet(
                        "QLabel {color:black;}")
                    self.developerExpires.setText("")
                else:
                    Log.e(f"ERROR: Cannot Delete. File not found: {dev_path}")
                    self.developerModeChk.setStyleSheet("color:red;")
                    self.developerExpires.setStyleSheet("QLabel {color:red;}")
                    self.developerExpires.setText("")
        except:
            Log.e("Error updating Developer Mode")
            self.developerModeChk.setStyleSheet("color:red;")
            self.developerExpires.setStyleSheet("QLabel {color:red;}")
            self.developerExpires.setText("")

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

    def toggleReqAdminUpdates(self, arg):
        try:
            if not os.path.isfile(Constants.user_constants_path):
                os.makedirs(os.path.split(
                    Constants.user_constants_path)[0], exist_ok=True)
            with open(Constants.user_constants_path, 'w') as uc:
                UserConstants.REQ_ADMIN_UPDATES = self.reqAdminUpd_chkbox.isChecked()
                uc_state = str(UserConstants.REQ_ADMIN_UPDATES)
                uc.write(f"REQ_ADMIN_UPDATES = {uc_state}")
        except:
            Log.e("Failed to save user constants settings.")

    def add_user(self):
        action = "Add User"
        roles = [e.name for e in UserRoles]
        roles = roles[1:]  # skip NONE
        roles[roles.index(UserRoles.OPERATE.name)] += " (Capture & Analyze)"
        role, ok = QtWidgets.QInputDialog().getItem(None, action,
                                                    "Role:", roles, 0, False)
        if not ok:
            return
        UserProfiles.create_new_user(UserRoles[role.split()[0]])
        self.update_table_data()

    def edit_user(self):
        # re-validate user creds before change
        # admin_name, _, _ = UserProfiles.change(UserRoles.ADMIN)
        # if admin_name != None:
        action = "Edit User"

        ok = False
        while not ok:
            initials, ok = QtWidgets.QInputDialog.getText(None, action,
                                                          "Initials:", QtWidgets.QLineEdit.Normal)
            if not ok:
                return
            initials = initials.upper()
            if initials.find(" ") >= 0 or len(initials) < 2 or len(initials) > 4:
                Log.w("Please enter user initials.")
                ok = False
        found, filename = UserProfiles.find(None, initials)
        if not found:
            Log.e(f"User {initials} not found. Cannot {action.lower()}.")
            return
        file = os.path.join(UserProfiles.PATH, filename)

        actions = ["Change Role", "Change Password",
                   "Change Initials", "Change Name"]
        action, ok = QtWidgets.QInputDialog().getItem(None, f"{action} {initials}",
                                                      "Action:", actions, 0, False)
        if not ok:
            return

        doc = minidom.parse(file)
        xml = doc.documentElement
        up = doc.getElementsByTagName("secure_user_info")

        for u in up:
            try:
                # allow exception if missing
                role = UserRoles(int(u.getAttribute("role")))
            except:
                role = UserRoles.INVALID
            password = u.getAttribute("password") if u.hasAttribute(
                "password") else "[Missing]"
            initials = u.getAttribute("initials") if u.hasAttribute(
                "password") else "[Missing]"
            name = u.getAttribute("name") if u.hasAttribute(
                "name") else "[Missing]"
            # if not u.hasAttribute("archived"):
            #     u.setAttribute('archived', 'True')
            if u.attributes["password"].value.find("X") < 0:
                # invalidate password hash, but allow recovery for audits
                u.attributes["password"].value += "X"
            # u.attributes["signature"].value = "[redacted]" # retain for audits

        action_id = actions.index(action)

        if action_id == 0:  # role
            if self.admin_file == UserProfiles.find(None, initials)[1]:
                Log.e(f"Cannot change role of active ADMIN user {initials}.")
                Log.e(
                    f"Create another ADMIN user and change this user role using their access.")
                return

            roles = [e.name for e in UserRoles]
            roles = roles[1:]  # skip NONE
            curr_role = roles.index(role.name)
            roles[roles.index(UserRoles.OPERATE.name)
                  ] += " (Capture & Analyze)"
            role, ok = QtWidgets.QInputDialog().getItem(None, action,
                                                        "Role:", roles, curr_role, False)
            if not ok:
                return
            role = UserRoles[role.split()[0]]

        if action_id == 1:  # password
            match = False
            while not match:
                ok = False
                while not ok:
                    pwd1, ok = QtWidgets.QInputDialog.getText(None, action,
                                                              "Password:", QtWidgets.QLineEdit.Password)
                    if not ok:
                        return
                    if len(pwd1) < 8:
                        Log.w(
                            "Passwords must be at least 8 characters. Please try again.")
                        ok = False

                ok = False
                while not ok:
                    pwd2, ok = QtWidgets.QInputDialog.getText(None, action,
                                                              "Confirm Password:", QtWidgets.QLineEdit.Password)
                    if not ok:
                        return
                    if len(pwd2) < 8:
                        Log.w(
                            "Passwords must be at least 8 characters. Please try again.")
                        ok = False

                match = True if pwd1 == pwd2 else False
                if not match:
                    Log.w("Passwords entered do not match. Please try again.")
            pwd = pwd1

        if action_id == 2:  # initials
            ok = False
            while not ok:
                initials, ok = QtWidgets.QInputDialog.getText(None, action,
                                                              "Initials:", QtWidgets.QLineEdit.Normal,
                                                              initials)
                if not ok:
                    return
                initials = initials.upper()
                if initials.find(" ") >= 0 or len(initials) < 2 or len(initials) > 4:
                    Log.w("Please enter your initials.")
                    ok = False

        if action_id == 3:  # name
            ok = False
            while not ok:
                name, ok = QtWidgets.QInputDialog.getText(None, action,
                                                          "Full name:", QtWidgets.QLineEdit.Normal,
                                                          name)
                if not ok:
                    return
                name = name.title()
                if name.find(" ") < 0 or len(name) < 4:
                    Log.w("Please enter your First and Last name.")
                    ok = False

        # recalculate security hashes and put in user profile
        salt = filename[:-4]
        if action_id == 1:
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

        info = doc.createElement('secure_user_info')
        xml.appendChild(info)
        info.setAttribute('name', name)
        info.setAttribute('initials', initials)
        info.setAttribute('role', str(role.value))
        info.setAttribute('password', password)
        info.setAttribute('signature', signature)

        ts_type = "modified"
        ts_val = dt.datetime.now().isoformat()
        hash = hashlib.sha256()
        hash.update(salt.encode())
        hash.update(ts_type.encode())
        hash.update(ts_val.encode())
        signature = hash.hexdigest()

        ts1 = doc.createElement('timestamp')
        xml.appendChild(ts1)
        ts1.setAttribute('type', ts_type)
        ts1.setAttribute('value', ts_val)
        ts1.setAttribute('signature', signature)

        # append new secure_user_info to xml
        xml_str = doc.toxml()  # indent ="\t")
        with open(file, "w") as f:
            f.write(xml_str)
            Log.d(f"Saved XML file: {file}")

        self.update_table_data()

    def remove_user(self, filename=None):
        action = "Delete User"

        if filename == None:
            ok = False
            while not ok:
                initials, ok = QtWidgets.QInputDialog.getText(None, action,
                                                              "Initials:", QtWidgets.QLineEdit.Normal)
                if not ok:
                    return
                initials = initials.upper()
                if initials.find(" ") >= 0 or len(initials) < 2 or len(initials) > 4:
                    Log.w("Please enter your initials.")
                    ok = False
            found, filename = UserProfiles.find(None, initials)
            if not found:
                Log.e(f"User {initials} not found. Cannot {action.lower()}.")
                return
            # found, fileself = UserProfiles.find(self.admin, None)
            if self.admin_file == filename:
                if UserProfiles.count() > 1:
                    Log.e(
                        f"Cannot delete the active ADMIN user {initials} when other accounts exist.")
                    Log.e(
                        "Create another ADMIN user and delete this user using their access.")
                    # Log.e("If multiple admins exist: Change \"Role\" of other admins prior to deleting them.")
                    return
                else:
                    if PopUp.question(self, "Remove Last User Account",
                                      "If you delete the only remaining user account, " +
                                      "anyone will be able to access this application.\n" +
                                      "\nWARNING: This operation cannot be undone!\n" +
                                      "You can add new users again in the future (if desired).\n" +
                                      "\nAre you sure you want to proceed?"):
                        UserProfiles.session_end()
                        name = self.parent.username.text()[6:]
                        Log.i(f"Goodbye, {name}! You have been signed out.")
                        self.parent.username.setText("User: [NONE]")
                        self.parent.userrole = UserRoles.NONE
                        self.parent.signinout.setText("&Sign In")
                        self.parent.manage.setText("&Manage Users...")
                        self.parent.ui1.tool_User.setText(name)
                        self.parent.parent.AnalyzeProc.tool_User.setText(name)
                        self.close()
                    else:
                        return
        else:
            user_info = UserProfiles.get_user_info(filename)
            initials = user_info[1]

        file = os.path.join(UserProfiles.PATH, filename)

        # mark user profile as deleted
        ts_type = "deleted"
        ts_val = dt.datetime.now().isoformat()
        salt = filename[:-4]
        hash = hashlib.sha256()
        hash.update(salt.encode())
        hash.update(ts_type.encode())
        hash.update(ts_val.encode())
        signature = hash.hexdigest()
        with open(file, 'rb+') as f:
            f.seek(-15, 2)
            f.write(
                f'<timestamp type="{ts_type}" value="{ts_val}" signature="{signature}"/></user_profile>'.encode())

        # delete user
        folder_name, file_name = os.path.split(file)
        archive_to = os.path.join(folder_name, "archived")
        FileManager.create_dir(archive_to)
        os.rename(file, os.path.join(archive_to, file_name))
        Log.w(f"User {initials} deleted.")

        if filename == None:
            self.update_table_data()

    def audit_user(self):
        action = "Audit User"

        ok = False
        while not ok:
            initials, ok = QtWidgets.QInputDialog.getText(None, action,
                                                          "Initials:", QtWidgets.QLineEdit.Normal)
            if not ok:
                return
            initials = initials.upper()
            if initials.find(" ") >= 0 or len(initials) < 2 or len(initials) > 4:
                Log.w("Please enter user initials.")
                ok = False
        found, filename = UserProfiles.find(None, initials)
        if not found:
            Log.e(f"User {initials} not found. Cannot {action.lower()}.")
            return
        file = os.path.join(UserProfiles.PATH, filename)

        if os.path.isfile(file):
            audit_failed = False
            xml = minidom.parse(file)
            top = xml.documentElement
            last_role = "NONE"
            last_password = "NONE"
            last_initials = "NONE"
            last_name = "NONE"
            passwords = 0
            col_audit = []
            col_times = []
            col_notes = []
            for child in top.childNodes:
                col_audit.append("[empty]")
                col_times.append("[empty]")
                col_notes.append("[empty]")
                audit_pass = False
                change_record = False
                change_role = False
                change_password = False
                change_initials = False
                change_name = False
                type = "unknown"
                time = "unknown"
                signature = None
                salt = filename[:-4]
                hash = hashlib.sha256()
                hash.update(salt.encode())
                Log.d(f"Processing next audit element... '{salt}'")
                i = child.attributes.items()
                for name, value in i:
                    if name == "signature":
                        signature = hash.hexdigest()
                        if change_record:
                            changed = []
                            if change_role:
                                changed.append(f"ROLE={last_role}")
                            if change_password:
                                changed.append(f"PASSWORD")
                            if change_initials:
                                changed.append(f"INITIALS={last_initials}")
                            if change_name:
                                changed.append(f"NAME={last_name}")
                            if len(changed) == 0:
                                changed.append("NOTHING")
                            audit_text = ", ".join(changed)
                            audit_text = "{}: " + audit_text
                            col_notes[-1] = audit_text
                        else:
                            audit_text = f"{type} @ {time}"
                            col_times[-1] = time
                            if type != "accessed" and col_times[-2] == "[empty]":
                                col_times[-2] = time
                                col_notes[-2] = col_notes[-2].format(
                                    type.upper())
                            col_notes[-1] = f"[{type.upper()}]"
                        if signature == value:
                            # Log.i(f"Signature audit PASS: {audit_text}")
                            audit_pass = True
                        else:
                            # Log.w(f"Signature audit FAIL: {audit_text}") # "'{signature}', '{value}'")
                            audit_pass = False
                        col_audit[-1] = "PASS" if audit_pass else "FAIL"
                    else:
                        try:
                            val = UserRoles(
                                int(value)).name if name == "role" else value
                            if name == "password":
                                if value.find("X") > 0:
                                    val = value[:-1]  # remove "X" for audit
                                else:
                                    # non-archived user profile (only one can exist)
                                    passwords += 1
                            if name == "type":
                                type = val
                            if name == "value":
                                time = val
                            if name in ["role", "password", "initials", "name"]:
                                change_record = True
                                if name == "role":
                                    if val != last_role:
                                        change_role = True
                                    last_role = val
                                if name == "password":
                                    if val != last_password:
                                        change_password = True
                                    last_password = val
                                if name == "initials":
                                    if val != last_initials:
                                        change_initials = True
                                    last_initials = val
                                if name == "name":
                                    if val != last_name:
                                        change_name = True
                                    last_name = val
                        except:
                            Log.w(
                                f"Audit exception for element: '{name}', '{value}'")
                            val = value
                        Log.d(f"Processing audit element: '{name}', '{val}'")
                        hash.update(val.encode())
                if signature == None:
                    # Log.w(f"Signature audit MISSING: {i}") # missing 'signature' attribute")
                    audit_pass = False
                    col_audit[-1] = "PASS" if audit_pass else "FAIL"
                    col_times[-1] = time
                    col_notes[-1] = f"MISSING signature: {i}"
                if not audit_pass:  # set color red for row
                    audit_failed = True
                    col_audit[-1] = f"*{col_audit[-1]}*"
                    col_times[-1] = f"*{col_times[-1]}*"
                    col_notes[-1] = f"*{col_notes[-1]}*"
            if passwords != 1:
                # Log.w(f"User audit FAIL: there can only be 1 valid user info element: found '{passwords}'")
                audit_pass = False
                audit_failed = True
                col_audit.append("*FAIL*")
                col_times.append("*now*")
                col_notes.append("*More than one valid user info element!*")
        else:
            audit_failed = True
            Log.w("User file missing. Cannot perform audit.")

        data = {'Timestamp': col_times,
                'Result': col_audit,
                'Audit Notes': col_notes}
        self.table.setRowCount(len(col_audit))
        self.table.setColumnCount(3)
        self.table.setData(data)
        self.sortBy.clear()
        self.sortBy.addItems(list(data.keys()))
        self.sortOrder.setCurrentIndex(0)  # force ascending
        # self.sort_user_table()

        if audit_failed:
            PopUp.warning(self, "Audit Result",
                          "There are failures in this audit!")  # TODO

        # Log.e("TODO: Show audit results in a QTableView format")
        # Log.e("TODO: Allow admin to acknowledge/ignore audit failures by creating a file: 'audit.ignore'")

    def delete_all_users(self):
        title = "Delete All Users"

        # Confirm user really wants this
        if not PopUp.question(self, "Remove All User Accounts",
                              "If you delete all of the user accounts, " +
                              "anyone will be able to access this application.\n" +
                              "\nWARNING: This operation cannot be undone!\n" +
                              "You can add new users again in the future (if desired).\n" +
                              "\nAre you sure you want to proceed?"):
            Log.w("User does not really want to delete all users. Aborted.")
            return

        # Get their password to allow the action
        admin_info = UserProfiles.get_user_info(self.admin_file)
        admin_initials = admin_info[1]

        ok = False
        while not ok:
            pwd, ok = QtWidgets.QInputDialog.getText(None, title,
                                                     "Confirm Password:", QtWidgets.QLineEdit.Password)
            if not ok:
                return None, None, 0
            if len(pwd) < 8:
                Log.w("Passwords must be at least 8 characters. Please try again.")
                ok = False

        authenticated, filename, params = UserProfiles.auth(
            admin_initials, pwd, UserRoles.ADMIN)
        if not authenticated:
            Log.e("User did not authenticate action to delete all users. Aborted.")
            return

        # Perform the action
        UserProfiles.session_end()
        name = self.parent.username.text()[6:]
        Log.i(f"Goodbye, {name}! You have been signed out.")
        self.parent.username.setText("User: [NONE]")
        self.parent.userrole = UserRoles.NONE
        self.parent.signinout.setText("&Sign In")
        self.parent.manage.setText("&Manage Users...")
        self.parent.ui1.tool_User.setText("Anonymous")
        self.parent.parent.AnalyzeProc.tool_User.setText("Anonymous")
        self.close()

        Log.w("Deleting all user accounts...")
        FileManager.create_dir(UserProfiles.PATH)
        user_files = os.listdir(UserProfiles.PATH)
        user_files = [x for x in user_files if ".xml" in x]  # remove folders
        for filename in user_files:
            self.remove_user(filename)
        Log.w("All user accounts deleted.")

    def update_table_data(self):
        data, rows, cols = self.get_user_data()
        self.table.setRowCount(rows)
        self.table.setColumnCount(cols)
        self.table.setData(data)
        self.sortBy.clear()
        self.sortBy.addItems(list(data.keys()))
        self.sort_user_table()

    def get_user_data(self):
        user_files, user_info = UserProfiles.get_all_user_info()
        Log.d("all_info:", user_info)
        user_names = [i[0] for i in user_info]
        Log.d("names:", user_names)
        user_initials = [i[1] for i in user_info]
        Log.d("inits:", user_initials)
        user_roles = [i[2] for i in user_info]
        Log.d("roles:", user_roles)
        user_created = [i[3] for i in user_info]
        Log.d("created:", user_created)
        user_modified = [i[4] for i in user_info]
        Log.d("modified:", user_modified)
        user_accessed = [i[5] for i in user_info]
        Log.d("accessed:", user_accessed)

        data = {'Initials': user_initials,
                'Name': user_names,
                'Role': user_roles,
                'Created': user_created,
                'Modified': user_modified,
                'Accessed': user_accessed}

        rows = len(user_info)
        cols = len(user_info[0]) if rows else 0
        return data, rows, cols

    def sort_user_table(self):
        sort_order = [QtCore.Qt.AscendingOrder, QtCore.Qt.DescendingOrder]
        self.table.sortItems(self.sortBy.currentIndex(
        ), sort_order[self.sortOrder.currentIndex()])  # sort by initials


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
        if filename == None:
            filename = ""
        file = os.path.join(UserProfiles.PATH, filename)
        if os.path.isfile(file):
            xml = minidom.parse(file)
            up = xml.getElementsByTagName("secure_user_info")
            for u in up:
                name = u.getAttribute("name") if u.hasAttribute(
                    "name") else "[Missing]"
                initials = u.getAttribute("initials") if u.hasAttribute(
                    "initials") else "[Missing]"
                p = u.getAttribute("password") if u.hasAttribute(
                    "password") else "[Missing]"
                s = u.getAttribute("signature") if u.hasAttribute(
                    "signature") else "[Missing]"
                try:
                    # allow exception if missing
                    role = UserRoles(int(u.getAttribute("role")))
                except:
                    role = UserRoles.INVALID

            ts = xml.getElementsByTagName("timestamp")
            now = dt.datetime.now().isoformat()
            today = now[0:now.find('T')+1]
            created = "[NEVER]"
            modified = "[NEVER]"
            accessed = "[NEVER]"
            for t in ts:
                x = t.getAttribute("type") if t.hasAttribute(
                    "type") else "[Missing]"
                y = t.getAttribute("value") if t.hasAttribute(
                    "value") else "[Missing]"
                if y != "[Missing]":
                    y = y[0:y.find('.')]  # remove subseconds
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

    def create_new_user(role):
        Log.w(f"Prompting to create new {role.name.lower()} user...")
        title = f"Create {role.name.title()} User"

        ok = False
        name = QtCore.QDir().home().dirName()
        while not ok:
            name, ok = QtWidgets.QInputDialog.getText(None, title,
                                                      "Full name:", QtWidgets.QLineEdit.Normal,
                                                      name)
            if not ok:
                return
            name = name.title()
            if name.find(" ") < 0 or len(name) < 4:
                Log.w("Please enter your First and Last name.")
                ok = False

        ok = False
        initials = "".join([c for c in name if c.isupper()])
        while not ok:
            initials, ok = QtWidgets.QInputDialog.getText(None, title,
                                                          "Initials:", QtWidgets.QLineEdit.Normal,
                                                          initials)
            if not ok:
                return
            initials = initials.upper()
            if initials.find(" ") >= 0 or len(initials) < 2 or len(initials) > 4:
                Log.w("Please enter your initials.")
                ok = False

        match = False
        while not match:
            ok = False
            while not ok:
                pwd1, ok = QtWidgets.QInputDialog.getText(None, title,
                                                          "Password:", QtWidgets.QLineEdit.Password)
                if not ok:
                    return
                if len(pwd1) < 8:
                    Log.w("Passwords must be at least 8 characters. Please try again.")
                    ok = False

            ok = False
            while not ok:
                pwd2, ok = QtWidgets.QInputDialog.getText(None, title,
                                                          "Confirm Password:", QtWidgets.QLineEdit.Password)
                if not ok:
                    return
                if len(pwd2) < 8:
                    Log.w("Passwords must be at least 8 characters. Please try again.")
                    ok = False

            match = True if pwd1 == pwd2 else False
            if not match:
                Log.w("Passwords entered do not match. Please try again.")

        UserProfiles.create(name, initials, role, pwd1)

    @staticmethod
    def change(requiredRole=UserRoles.ANY):
        if UserProfiles.count() == 0:
            return None, None, 0

        title = "Sign In" if requiredRole.value == UserRoles.ANY.value else requiredRole.name.title()

        ok = False
        while not ok:
            initials, ok = QtWidgets.QInputDialog.getText(None, title,
                                                          "Initials:", QtWidgets.QLineEdit.Normal)
            if not ok:
                return None, None, 0
            initials = initials.upper()
            if initials.find(" ") >= 0 or len(initials) < 2 or len(initials) > 4:
                Log.w("Please enter your initials.")
                ok = False

        ok = False
        while not ok:
            pwd, ok = QtWidgets.QInputDialog.getText(None, title,
                                                     "Password:", QtWidgets.QLineEdit.Password)
            if not ok:
                return None, None, 0
            if len(pwd) < 8:
                Log.w("Passwords must be at least 8 characters. Please try again.")
                ok = False

        authenticated, filename, params = UserProfiles.auth(
            initials, pwd, requiredRole)
        if authenticated:
            Log.i(
                f"Welcome, {params[0]}! Your assigned role is {params[2].name}.")
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
            password = u.getAttribute("password") if u.hasAttribute(
                "password") else "[Missing]"
            initials = u.getAttribute("initials") if u.hasAttribute(
                "password") else "[Missing]"
            name = u.getAttribute("name") if u.hasAttribute(
                "name") else "[Missing]"
            if u.attributes["password"].value.find("X") < 0:
                # invalidate password hash, but allow recovery for audits
                u.attributes["password"].value += "X"

        ok = False
        while not ok:
            current_password, ok = QtWidgets.QInputDialog.getText(None, title,
                                                                  "Current Password:", QtWidgets.QLineEdit.Password)
            if not ok:
                return
            if len(current_password) < 8:
                Log.w("Passwords must be at least 8 characters. Please try again.")
                ok = False

        authenticated, filename, params = UserProfiles.auth(
            initials, current_password, UserRoles.ANY)
        if not authenticated:
            Log.e("User did not authenticate action to change their password. Aborted.")
            return

        match = False
        while not match:
            ok = False
            while not ok:
                pwd1, ok = QtWidgets.QInputDialog.getText(None, title,
                                                          "New Password:", QtWidgets.QLineEdit.Password)
                if not ok:
                    return
                if len(pwd1) < 8:
                    Log.w("Passwords must be at least 8 characters. Please try again.")
                    ok = False

            ok = False
            while not ok:
                pwd2, ok = QtWidgets.QInputDialog.getText(None, title,
                                                          "Confirm Password:", QtWidgets.QLineEdit.Password)
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

        info = doc.createElement('secure_user_info')
        xml.appendChild(info)
        info.setAttribute('name', name)
        info.setAttribute('initials', initials)
        info.setAttribute('role', str(role.value))
        info.setAttribute('password', password)
        info.setAttribute('signature', signature)

        ts_type = "modified"
        ts_val = dt.datetime.now().isoformat()
        hash = hashlib.sha256()
        hash.update(salt.encode())
        hash.update(ts_type.encode())
        hash.update(ts_val.encode())
        signature = hash.hexdigest()

        ts1 = doc.createElement('timestamp')
        xml.appendChild(ts1)
        ts1.setAttribute('type', ts_type)
        ts1.setAttribute('value', ts_val)
        ts1.setAttribute('signature', signature)

        # append new secure_user_info to xml
        xml_str = doc.toxml()  # indent ="\t")
        with open(file, "w") as f:
            f.write(xml_str)
            Log.d(f"Saved XML file: {file}")
        Log.w("Password changed: " + ("*" * len(pwd)))

    @staticmethod
    def session_create(salt):
        file = os.path.join(UserProfiles.PATH, "session.key")
        today = dt.datetime.now().isoformat().split('T')[0]
        hash = hashlib.sha256()
        hash.update(salt.encode())
        hash.update(today.encode())
        session_key = hash.hexdigest()
        with open(file, 'w') as f:
            f.write(session_key)
            Log.d("User session created.")
        UserProfiles.user_preferences = UserPreferences(
            UserProfiles.get_session_file())
        UserProfiles.user_preferences.set_preferences()
        # print(UserProfiles.user_preferences.get_preferences())
        # print(UserProfiles.user_preferences.get_folder_save_path(
        #     device_id=12345678, port_id=1))
        # print(UserProfiles.user_preferences.get_file_save_path(
        #     device_id=12345678, port_id=1))

    @staticmethod
    def session_info():
        file = os.path.join(UserProfiles.PATH, "session.key")
        today = dt.datetime.now().isoformat().split('T')[0]
        if os.path.exists(file):
            files, infos = UserProfiles.get_all_user_info()
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
        today = dt.datetime.now().isoformat().split('T')[0]
        if os.path.exists(file):
            files, infos = UserProfiles.get_all_user_info()
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
                    n = u.getAttribute("name") if u.hasAttribute(
                        "name") else "[Missing]"
                    i = u.getAttribute("initials") if u.hasAttribute(
                        "initials") else "[Missing]"
                if n == name or i == initials:  # only check most recent secure_user_info record
                    return True, filename
        return False, None

    @staticmethod
    def create(name, initials, role, pwd):
        found, filename = UserProfiles.find(name, initials)
        sign_in_user = UserProfiles.count() == 0
        if not found:
            Log.i(
                f"Create user, Name: {name}, initials: {initials}, role: {role.name} password: {'*'*len(pwd)}")
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
            xml = doc.createElement('user_profile')
            doc.appendChild(xml)

            info = doc.createElement('secure_user_info')
            xml.appendChild(info)
            info.setAttribute('name', name)
            info.setAttribute('initials', initials)
            info.setAttribute('role', str(role.value))
            info.setAttribute('password', password)
            info.setAttribute('signature', signature)

            ts_type = "created"
            ts_val = dt.datetime.now().isoformat()
            hash = hashlib.sha256()
            hash.update(salt.encode())
            hash.update(ts_type.encode())
            hash.update(ts_val.encode())
            signature = hash.hexdigest()

            ts1 = doc.createElement('timestamp')
            xml.appendChild(ts1)
            ts1.setAttribute('type', ts_type)
            ts1.setAttribute('value', ts_val)
            ts1.setAttribute('signature', signature)
            # ts2 = doc.createElement('timestamp')
            # xml.appendChild(ts2)
            # ts2.setAttribute('name', "modified")
            # ts2.setAttribute('value', ts)

            # Log.d(doc)
            xml_str = doc.toxml()  # indent ="\t")
            with open(file, "w") as f:
                f.write(xml_str)
                Log.d(f"Saved XML file: {file}")

            if sign_in_user:
                # create session
                UserProfiles.auth(initials, pwd, UserRoles.ADMIN)

        else:
            Log.e(
                f"Failed to create user. User info conflicts with user profile '{filename[:-4]}'.")

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
            f"User role check: user={userRole.name}, required={requiredRole.name}, result={result}")
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
                    name = u.getAttribute("name") if u.hasAttribute(
                        "name") else "[Missing]"
                    initials = u.getAttribute("initials") if u.hasAttribute(
                        "initials") else "[Missing]"
                    p = u.getAttribute("password") if u.hasAttribute(
                        "password") else "[Missing]"
                    s = u.getAttribute("signature") if u.hasAttribute(
                        "signature") else "[Missing]"

                    try:
                        # allow exception if missing
                        role = UserRoles(int(u.getAttribute("role")))
                    except:
                        role = UserRoles.INVALID

                UserProfiles.session_create(filename[:-4])
                # only check most recent secure_user_info record
                if not UserProfiles.check(role, requiredRole):
                    Log.e(
                        f"User {initials} does not have the required {requiredRole.name} role privileges.")
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
                    with open(file, 'rb+') as f:
                        f.seek(-15, 2)
                        f.write(
                            f'<timestamp type="{ts_type}" value="{ts_val}" signature="{signature}"/></user_profile>'.encode())
                        # f.write(f'</user_profile>\r\n'.encode())
                    Log.d(f"User {initials} authenticated successfully.")
                    return True, filename, [name, initials, role]
                elif p != password and s == signature:
                    Log.e(f"Auth failure: Invalid credentials.")
                    Log.d(
                        f"Auth failure details: user={initials}, p={p == password}, s={s == signature}")
                    UserProfiles.session_end()
                    return False, filename, None
                else:  # signature not valid
                    Log.e(
                        f"Auth failure: Corrupt user profile {initials}. See an administrator.")
                    Log.e(
                        f"File security checks indicate your profile is invalid or had unauthorized changes made to it.")
                    Log.d(
                        f"Auth failure details: file={filename}, p={p == password}, s={s == signature}")
                    UserProfiles.session_end()
                    return False, filename, None
            else:
                Log.e(f"Auth failure: Invalid user account.")
                Log.d(
                    f"Auth failure details: User profile is not a file! Not found: {file}")
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
                with open(dev_path, 'r') as dev:
                    try:
                        encode_str1 = dev.readline()
                        encode_str2 = dev.readline()
                        encode_key = "DEADBEEFDEADBEEFDEAD"
                        hexify_str1 = bytes(
                            [(ord(a) ^ ord(b)) for a, b in zip(encode_str1, encode_key)]).decode()
                        hexify_str2 = bytes(
                            [(ord(a) ^ ord(b)) for a, b in zip(encode_str2, encode_key)]).decode()
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
                            f"Developer Mode encoded expiration value could not be parsed ({expires_on})! Please renew or disable.")
                        is_error = True
                    elif build_date != Constants.app_date and not UserConstants.DEV_PERSIST_VER:
                        Log.e(
                            "Developer Mode was enabled for another SW version and is invalid here. Please renew or disable.")
                        is_error = True
                        expires_at = ""
                    elif expires_on > now - dt.timedelta(days=1):
                        if expires_on < now + dt.timedelta(days=UserConstants.DEV_EXPIRE_LEN):
                            enabled = True
                        else:
                            Log.e(
                                f"Developer Mode expiration date ({expires_at}) is invalid! Please renew or disable.")
                            is_error = True
                            expires_at = ""
                    else:
                        Log.e(
                            f"Developer Mode expired on {expires_at}. Please renew or disable.")
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
            a_list = ['Traceback (most recent call last):']
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        return enabled, is_error, expires_at


class UserPreferences:
    def __init__(self, user_session_key: str):
        self._set_user_session(user_session_key)
        self.setup()

    def setup(self) -> None:
        user_info = self._get_user_session()
        user_preferences_path = os.path.join(Constants.local_app_data_path, "profiles/users",
                                             f"{user_info}-preferences.json")
        global_preferences_path = os.path.join(
            Constants.local_app_data_path, "global_preferences.json")

        self.set_use_global(True)
        if os.path.exists(global_preferences_path):
            self._set_global_preferences_path(global_preferences_path)
        else:
            Log.e(
                tag=TAG, msg="No global file format preferences found. Writing global and using.")
            FileStorage.DEV_write_default_preferences(global_preferences_path)
            self._set_global_preferences_path(global_preferences_path)

        if os.path.exists(user_preferences_path):
            Log.d(TAG, 'Using User Preferences')
            self._set_user_preferences_path(user_preferences_path)
            self.set_use_global(False)
        elif user_info is not None:
            Log.w(TAG, 'Creating User Preferences from default format and using global.')
            FileStorage.DEV_write_default_preferences(user_preferences_path)
            self._set_user_preferences_path(user_preferences_path)

    def set_preferences(self):
        import json
        # Load and parse preferences file.
        preferences_data = None
        if self.get_use_global():
            with open(self._get_global_preferences_path(), "r") as preferences_file:
                preferences_data = json.load(preferences_file)
        else:
            with open(self._get_user_preferences_path(), 'r') as preferences_file:
                preferences_data = json.load(preferences_file)

        # Temporary variables to hold parsed contents
        load_data_path = str(preferences_data["load_data_path"])
        write_data_path = str(preferences_data["write_data_path"])
        folder_tag_format = str(preferences_data["folder_format"])
        file_tag_format = str(preferences_data["filename_format"])
        folder_delimiter = str(
            preferences_data["folder_format_delimiter"])
        filename_delimiter = str(
            preferences_data["filename_format_delimiter"])
        date_format = str(preferences_data["date_format"])
        time_format = str(preferences_data["time_format"])

        # Set user preferences using accessor and mutators.
        self._set_load_data_path(load_data_path)
        self._set_write_data_path(write_data_path)
        self._set_folder_format_pattern(folder_tag_format)
        self._set_file_format_pattern(file_tag_format)
        self._set_folder_delimiter(folder_delimiter)
        self._set_file_delimiter(filename_delimiter)
        self._set_date_format(date_format)
        self._set_time_format(time_format)

    def get_preferences(self) -> dict:
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
        return preferences_dict

    def load_user_preferences(self) -> dict:
        return FileStorage.DEV_load_preferences(self._get_user_preferences_path())

    def load_global_preferences(self) -> dict:
        return FileStorage.DEV_load_preferences(self._get_global_preferences_path())

    def reset_global_preferences(self):
        FileStorage.DEV_write_default_preferences(
            save_path=self._get_global_preferences_path())

    def write_global_preferences(self):
        FileStorage.DEV_write_preferences(
            save_path=self._get_global_preferences_path(), preferences=self.get_preferences())

    def reset_user_preferences(self):
        FileStorage.DEV_write_default_preferences(
            save_path=self._get_user_preferences_path())

    def write_user_preferences(self):
        FileStorage.DEV_write_preferences(
            save_path=self._get_user_preferences_path(), preferences=self.get_preferences())

    def get_folder_save_path(self, runname: str, device_id: int, port_id: int) -> str:
        return self._build_save_path(
            pattern=self._get_folder_format_pattern().split(self._get_folder_delimiter()), delimiter=self._get_folder_delimiter(), runname=runname, device_id=device_id, port_id=port_id)

    def get_file_save_path(self, runname: str, device_id: int, port_id: int) -> str:
        return self._build_save_path(
            pattern=self._get_file_format_pattern().split(self._get_file_delimiter()), delimiter=self._get_file_delimiter(),  runname=runname, device_id=device_id, port_id=port_id)

    def set_use_global(self, use_global) -> None:
        self.use_global = use_global

    def get_use_global(self) -> bool:
        return self.use_global
    # -- Private Utilities -- #

    def _build_save_path(self, pattern: list, runname: str, delimiter: str, device_id: int, port_id: int) -> str:
        save_path = ""

        for i, tag in enumerate(pattern):
            print(save_path)
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
            elif tag == Constants.select_tag_prompt:
                Log.w(TAG, 'Ignoring empty folder format tag pattern')
                continue  # skip adding another delimeter
            else:
                Log.e(TAG, f'Invalid folder format tag pattern: {tag}')
                raise ValueError('Invalid folder format tag pattern')
            if i < len(pattern) - 1:
                save_path = save_path + delimiter

        return save_path

    def _on_username(self) -> str:
        is_valid, user_info = UserProfiles.session_info()
        if is_valid:
            username = user_info[0]
            return username
        else:
            Log.e(TAG, 'Invalid user session')
            raise ValueError("Invalid user session")

    def _on_initials(self) -> str:
        is_valid, user_info = UserProfiles.session_info()
        if is_valid:
            initials = user_info[1]
            return initials
        else:
            Log.e(TAG, 'Invalid user session')
            raise ValueError("Invalid user session")

    def _on_device(self, device_id: int) -> str:
        return str(device_id)

    def _on_runname(self, runname: str) -> str:
        return str(runname)

    def _on_date(self) -> str:
        from datetime import datetime
        format_mapping = {
            "YYYY": "%Y",  # Year with century as a decimal number
            "MM": "%m",    # Month as a zero-padded decimal number
            "DD": "%d",    # Day of the month as a zero-padded decimal number
            # Hour (24-hour clock) as a zero-padded decimal number
            "hh": "%H",
            "mm": "%M",    # Minute as a zero-padded decimal number
            "ss": "%S",    # Second as a zero-padded decimal number
            "A": "%p",     # AM/PM (12-hour clock)
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
        for delimiter in Constants.path_delimiters:
            folder_format_pattern = folder_format_pattern.replace(
                delimiter, '|')
        split_parts = folder_format_pattern.split('|')
        tokens = [part for part in split_parts]
        self._folder_format_pattern = tokens

    def _set_file_format_pattern(self, file_format_pattern: str) -> None:
        for delimiter in Constants.path_delimiters:
            file_format_pattern = file_format_pattern.replace(
                delimiter, '|')
        split_parts = file_format_pattern.split('|')
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

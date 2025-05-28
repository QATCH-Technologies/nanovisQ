import datetime
import os
import hashlib
import pyzipper

TAG = ""  # "[FileStorage]"


###########################################################################
# Get an IO handle to read/write/append to a secured ZIP archive record
###########################################################################

class secure_open:

    def __init__(self, file, mode='r', zipname=None, insecure=False):
        self.file = file
        self.mode = mode
        self.zipname = zipname
        self.insecure = insecure
        self.zf = None
        self.fh = None

    def __enter__(self):
        file = self.file
        mode = self.mode
        zipname = self.zipname
        zf = self.zf
        fh = self.fh

        # NOTE: Writing a non-existent file will always use encryption... is that fine?
        # It should be fine. Tries to access record without 'pwd' and only encrypts if needed
        if os.path.isfile(file):
            self.fh = open(file, mode)
            return self.fh
        else:
            archive, record = os.path.split(file)
            folder, subDir = os.path.split(archive)
            if zipname == None:
                zipname = subDir
            zn = os.path.join(archive, f"{zipname}.zip")
            if 'w' in mode and os.path.isfile(zn):
                # if archive exists, upgrade 'w' to 'a' to keep other records already in ZIP
                zm = mode.replace('w', 'a')
            else:
                zm = mode
            zf = pyzipper.AESZipFile(zn, zm,
                                     compression=pyzipper.ZIP_DEFLATED,
                                     allowZip64=True,
                                     encryption=pyzipper.WZ_AES)
            if True:
                i = 0
                password_protected = False
                while True:
                    i += 1
                    if i > 3:
                        print(
                            "This ZIP has encrypted files: Try again with a valid password!")
                        break
                    try:
                        zf.testzip()  # will fail if encrypted and no password set
                        break  # test pass
                    except RuntimeError as e:
                        if 'encrypted' in str(e):
                            print('Accessing secured records...')
                            zf.setpassword(hashlib.sha256(
                                zf.comment).hexdigest().encode())
                            password_protected = True
                        else:
                            # RuntimeError for other reasons....
                            print("ZIP RuntimeError: " + str(e))
                            break
                    except Exception as e:
                        # other Exception for any reason...
                        print("ZIP Exception: " + str(e))
                        break

                # UserProfiles.count() > 0 and password_protected == False and UserProfiles.checkDevMode()[0] == False:
                if False:
                    # create a protected archive
                    friendly_name = f"{subDir} ({datetime.date.today()})"
                    zf.comment = friendly_name.encode()  # run name
                    zf.setpassword(hashlib.sha256(
                        zf.comment).hexdigest().encode())
                else:
                    zf.setencryption(None)
                    if True:  # UserProfiles.checkDevMode()[0]:
                        print("Developer Mode is ENABLED - NOT encrypting ZIP file")

                proceed = True
                archive_file = record
                namelist = zf.namelist()

                if not 'w' in mode:  # reading or appending
                    if record in namelist:
                        crc_file = archive_file[:-4] + ".crc"

                        if crc_file in namelist and crc_file != record:
                            archive_CRC = str(
                                hex(zf.getinfo(archive_file).CRC))
                            compare_CRC = zf.read(crc_file).decode()

                            print(f"Archive CRC: {archive_CRC}")
                            print(f"Compare CRC: {compare_CRC}")

                            if not archive_CRC == compare_CRC:
                                print(f"Record {record} CRC mismatch!")
                                proceed = False
                        elif archive_file.endswith(".csv"):
                            print(f"Record {record} missing CRC file!")
                            proceed = False
                        else:
                            print(
                                f"Record {record} has no CRC file, but it's not a CSV, so allow it to proceed...")
                            # proceed = False
                    else:
                        print(f"Record {record} not found!")
                        # proceed = False

                if proceed or self.insecure:
                    self.zf = zf  # export to global
                    self.fh = zf.open(archive_file, mode, force_zip64=True)
                    return self.fh
                else:
                    zf.close()
                    raise Exception(
                        f"Security checks failed. Cannot open secured file {file}.")

    def __exit__(self, type, value, traceback):
        file = self.file
        mode = self.mode
        zipname = self.zipname
        zf = self.zf
        fh = self.fh

        if fh != None:
            fh.close()

        if zf != None:
            _, record = os.path.split(file)

            archive_file = record
            if not 'r' in mode:  # writing or appending
                namelist = zf.namelist()
                if record in namelist:
                    if archive_file.endswith(".csv"):
                        crc_file = archive_file[:-4] + ".crc"
                        archive_CRC = str(hex(zf.getinfo(archive_file).CRC))
                        # zf.writestr(crc_file, archive_CRC)
                        with zf.open(crc_file, 'w') as crc_fh:  # no append, must 'w'
                            crc_fh.write(archive_CRC.encode())

            zf.close()

    ###########################################################################
    # Get an IO handle to read/write/append to a secured ZIP archive record
    ###########################################################################

    @staticmethod
    def file_exists(file, zipname=None):
        if os.path.isfile(file):
            return True
        else:

            archive, record = os.path.split(file)
            folder, subDir = os.path.split(archive)
            if zipname == None:
                zipname = subDir
            zn = os.path.join(archive, f"{zipname}.zip")

            if not os.path.isfile(zn):
                return False

            zf = pyzipper.AESZipFile(zn, 'r',
                                     compression=pyzipper.ZIP_DEFLATED,
                                     allowZip64=True,
                                     encryption=pyzipper.WZ_AES)
            namelist = zf.namelist()

            return True if record in namelist else False

    @staticmethod
    def get_namelist(zip_path: str, zip_name: str = "capture"):
        """
        Retrieves the list of file names contained in a zip archive.

        Args:
            zip_path (str): The full path to the directory containing the zip archive.
            zip_name (str, optional): The base name of the zip file (without extension). 
                                    Defaults to "capture".

        Returns:
            list: A list of file names in the zip archive.

        Raises:
            FileNotFoundError: If the zip file does not exist.

        Notes:
            The method assumes the zip file is encrypted using AES and handles it 
            accordingly with `pyzipper.AESZipFile`.

        Example:
            namelist = FileManager.get_namelist("/path/to/zip/folder")
        """
        # Split the provided path into archive and record names
        # 'archive' is the directory, 'record' is the file/leaf
        archive, record = os.path.split(zip_path)
        # Extract parent directory and its leaf name
        folder, subDir = os.path.split(archive)

        # Use `subDir` as the zip file name if `zip_name` is None
        if zip_name is None:
            zip_name = subDir

        # Construct the full path to the zip file using the determined zip name
        zn = os.path.join(archive, f"{zip_name}.zip")

        # Check if the zip file exists. If not, raise FileNotFoundError
        if not os.path.isfile(zn):
            raise FileNotFoundError(f"The zip file {zn} does not exist.")

        # Open the zip file using pyzipper for AES encryption handling
        zf = pyzipper.AESZipFile(
            zn, 'r',  # Open in read mode
            compression=pyzipper.ZIP_DEFLATED,  # Use ZIP_DEFLATED compression
            allowZip64=True,                   # Support for files larger than 4GB
            encryption=pyzipper.WZ_AES         # AES encryption support
        )

        # Extract and return the list of file names in the zip archive
        namelist = zf.namelist()
        return namelist

from QATCH.core.constants import Constants
from subprocess import Popen, PIPE
from datetime import date
import logging
import os
import shutil
import argparse


class QatchTagger():

    def __init__(self):

        log = logging.getLogger()
        log.setLevel(logging.DEBUG)

        # set up argument parser
        parser = argparse.ArgumentParser(
            description='QATCH nanovisQ software tagger utility')
        parser.add_argument("--nightly", action="store_true",
                            help="Set to build a nightly build")

        # get arguments
        args = parser.parse_args()
        self.args = args

    def run(self):

        if self.args.nightly:
            logging.info("Creating a nightly build...")
            Constants.app_version = f"{Constants.app_version}_nightly"
            Constants.app_date = date.today().strftime("%Y-%m-%d")

        tag_name = f"nanovisQ_SW_{Constants.app_version} ({Constants.app_date})"

        os.chdir(os.path.dirname(__file__))  # change cwd to location of file
        path_to_trunk = os.path.join(os.path.dirname(os.getcwd()), "trunk")
        path_to_tag = os.path.join(
            os.path.dirname(os.getcwd()), "tags", tag_name)
        path_to_dev = os.path.join(os.path.dirname(os.getcwd()), "dev")
        installer_dst = os.path.join(path_to_tag, "dist")

        if self.args.nightly:
            dirname, basename = os.path.split(path_to_tag)
            path_to_tag = os.path.join(dirname, "nightly", basename)

        logging.info(f"Tag name: {tag_name}")
        logging.info(f"Path to tag: {path_to_tag}")

        # TEST CODE, do not keep
        # path_to_tag += " test"
        # if os.path.exists(path_to_tag):
        #   shutil.rmtree(path_to_tag)

        # MOVE TRUNK TO TAG (IF NOT EXISTS)
        try:
            os.makedirs(path_to_tag)  # may raise OSError
            for f in os.listdir(path_to_trunk):
                if f.startswith(".git"):
                    continue
                src = os.path.join(path_to_trunk, f)
                dst = os.path.join(path_to_tag, f)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

        except OSError as e:
            logging.error(f"Tag already exists: {tag_name}. Cannot continue.")
            try:
                contents = os.listdir(path_to_tag)
                logging.error("Existing folder contents:")
                for file in contents:
                    logging.error(f"> {file}")
            except:
                raise e
            return

        except Exception as e:
            logging.error(e)
            return

        # ONLY KEEP THE KEEPERS IN TAG FOR PYTHON CODE
        try:
            keepers = ["docs", "QATCH", f"QATCH_Q-1_FW_py_{Constants.best_fw_version}",
                       "tools", "app.py", "launch.bat", "requirements.txt"]
            for f in os.listdir(path_to_tag):
                f = os.path.join(path_to_tag, f)
                keep = False
                for k in keepers:
                    if f.endswith(k):
                        keep = True
                if not keep:
                    logging.debug(f"Removing: {f}")
                    if os.path.isdir(f):
                        shutil.rmtree(f)
                    elif os.path.isfile(f):
                        os.remove(f)
                    else:
                        raise Exception(
                            f"Unknown file type: '{os.path.basename(f)}' is not a file or folder.")

            fw_build_path = os.path.join(path_to_tag, keepers[2], "build")
            if os.path.exists(fw_build_path):
                logging.debug(f"Removing FW Build path: {fw_build_path}")
                shutil.rmtree(fw_build_path)

        except Exception as e:
            logging.error(e)
            return

        # CREATE THE PYTHON CODE ZIP
        try:
            logging.info("Making ZIP of python code... (may take a while)")
            archive_name = os.path.join(os.path.dirname(
                path_to_tag), os.path.basename(path_to_tag).split()[0]) + "_py"
            zip_py = shutil.make_archive(archive_name, "zip", path_to_tag)
            logging.debug(f"Created: {zip_py}")
            logging.debug("Moving to 'dist' folder...")
            move_to = os.path.join(path_to_tag, "dist",
                                   os.path.basename(zip_py))
            os.makedirs(os.path.dirname(move_to))
            os.rename(zip_py, move_to)
            zip_py = move_to
            logging.info(f"Python ZIP: {zip_py}")

        except Exception as e:
            logging.error(e)
            return

        if not self.args.nightly:

            # MOVE BACK IN THE PRE-COMPILED EXE TO DIST FOLDER
            try:
                logging.debug("Moving EXE dist to tag...")
                path_to_dist_src = os.path.join(
                    path_to_trunk, "dist", "QATCH nanovisQ")
                path_to_dist_dst = os.path.join(
                    path_to_tag, "dist", "QATCH nanovisQ")
                shutil.copytree(path_to_dist_src, path_to_dist_dst)
                logging.debug("Moved successfully.")

            except Exception as e:
                logging.error(e)
                return

            # REMOVE FIRMWARE SOURCE FROM EXE BUILD
            # try:
            #   if 'r' in Constants.best_fw_version:
            #     logging.debug("Removing FW source files from EXE build...")
            #     fw_main_path = os.path.join(path_to_tag, f"QATCH_Q-1_FW_py_{Constants.best_fw_version}")
            #     for f in os.listdir(fw_main_path):
            #       f = os.path.join(fw_main_path, f)
            #       keep = False
            #       if f.endswith(".hex") or f.endswith(".md") or f.endswith(".pdf"):
            #         keep = True
            #       if not keep:
            #         logging.debug(f"Removing: {f}")
            #         if os.path.isdir(f):
            #           shutil.rmtree(f)
            #         elif os.path.isfile(f):
            #           os.remove(f)
            #         else:
            #           raise Exception(f"Unknown file type: '{os.path.basename(f)}' is not a file or folder.")
            #     logging.debug("Removed successfully.")
            #   else:
            #     logging.debug("Skipping FW source removal for 'b' build.")
            #
            # except Exception as e:
            #   logging.error(e)
            #   return

            # CREATE THE EXE CODE ZIP
            try:
                logging.info(
                    "Making ZIP of bundled code... (may take a while)")
                archive_name = os.path.join(
                    path_to_tag, "dist", os.path.basename(path_to_tag).split()[0]) + "_exe"
                zip_exe = shutil.make_archive(
                    archive_name, "zip", path_to_dist_dst)
                logging.debug(f"Removing {path_to_dist_dst}")
                shutil.rmtree(path_to_dist_dst)
                logging.info(f"Bundled ZIP: {zip_exe}")

            except Exception as e:
                logging.error(e)
                return

            # CREATE THE INSTALLER CODE ZIP USING INSTALLER FROM DEV TOOLS
            try:
                logging.info(
                    "Making ZIP of installer code... (may take a while)")
                installer_version = "1.0.0.2"
                # archive_name = os.path.join(path_to_tag, "dist", os.path.basename(path_to_tag).split()[0]) + "_installer"
                installer_src = os.path.join(
                    path_to_dev, "tools", "installer", "tags", installer_version, "dist", "QATCH installer.exe")
                installer_crc = os.path.join(
                    path_to_dev, "tools", "installer", "tags", installer_version, "dist", "installer.checksum")
                # installer_dst = os.path.join(path_to_tag, "dist")  # , "installer")
                os.makedirs(installer_dst, exist_ok=True)
                shutil.copy2(installer_src, installer_dst)
                shutil.copy2(installer_crc, installer_dst)
                # shutil.copy2(zip_exe, installer_dst)
                # zip_installer = shutil.make_archive(archive_name, "zip", installer_dst)
                # logging.debug(f"Removing {installer_dst}")
                # shutil.rmtree(installer_dst)
                # logging.info(f"Installer ZIP: {zip_installer}")
                logging.info(
                    f"Installer EXE: {os.path.join(installer_dst, 'QATCH installer.exe')}")

            except Exception as e:
                logging.error(e)
                return

        # CREATE DEFAULT TARGETS FILE
        targets_path = os.path.join(installer_dst, 'targets.csv')
        os.makedirs(installer_dst, exist_ok=True)
        with open(targets_path, 'w') as f:
            # change file contents to "ALL" after tag verification tests PASS
            f.write("WINDOWS-AN4Q851")

        # PUSH THE NEW TAG TO THE REPO
        try:
            # installer_dst = os.path.join(path_to_tag, "dist")
            script_path = os.path.join(os.getcwd(), 'push_tag.bat')
            logging.info(
                f"Updating '{os.path.basename(script_path)}' script...")
            with open(script_path, 'w') as f:
                f.write(
                    f"git tag -a {Constants.app_version} -m \"{tag_name}\"\n")
                f.write(
                    f"git push origin {Constants.app_version}\n")
                f.write(
                    f"git tag -l --sort=taggerdate > tags.txt\n")
                f.write(
                    f"REM move 'tags.txt' to 'dist' folder\n")
                f.write(
                    f"pause"
                )

            if self.args.nightly:
                push = "push"
            else:
                push = input("Enter 'push' to tag now: ").lower()
            if push == "push":
                logging.info("Pushing tag to origin...")
                p = Popen(script_path, cwd=os.path.dirname(script_path),
                          shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                stdout, stderr = p.communicate(b'exit\n')
                if stdout is not None:
                    for line in stdout.decode().splitlines():
                        line = line.strip()
                        if len(line):
                            logging.debug(line)
                if stderr is not None:
                    for line in stderr.decode().splitlines():
                        line = line.strip()
                        if len(line):
                            logging.error(line)
                if p.returncode == 0:
                    logging.info("Successfully pushed tag to origin.")
                else:
                    logging.error(
                        f"Failed to push tag. Return code: {p.returncode}. Check debug script output above.")

                logging.info("Moving 'tags.txt' to 'dist' folder...")
                tags_path = os.path.join(os.getcwd(), 'tags.txt')
                move_to = os.path.join(
                    installer_dst, os.path.basename(tags_path))
                if not os.path.exists(tags_path):
                    raise FileNotFoundError(
                        f"Missing file: \"{tags_path}\"")
                if os.path.exists(move_to):
                    raise PermissionError(
                        f"File already exists: \"{move_to}\"")
                shutil.copyfile(tags_path, move_to)

            else:
                logging.warning(
                    "User declined tagging. Not pushing tag to repo. Run 'push_tag.bat' when ready.")

        except Exception as e:
            logging.error(e)
            return


if __name__ == '__main__':
    QatchTagger().run()

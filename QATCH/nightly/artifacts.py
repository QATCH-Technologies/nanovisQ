import json as j
import os as o
import requests as r
import pyzipper as z

try:
    from QATCH.nightly.security import GH_Security
except:
    from security import GH_Security

from PyQt5.QtCore import pyqtBoundSignal
from threading import Event as event


class GH_Artifacts:

    headers = \
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
    latest_build_file = o.path.join(
        o.path.dirname(__file__), "latest_build.json")

    def __init__(self) -> None:
        self.security = GH_Security()

    def raw(self) -> dict:
        resp = r.get(
            self.security.caesar_decipher("*".join(self.security.GHAPIURL)),
            headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def all(self) -> tuple[dict, str]:
        resp = self.raw()
        keys = []
        for i in range(resp['total_count']):
            for j in list(resp['artifacts'][i].keys()).copy():
                k: str = j
                if any(list([
                    k.startswith('a'),
                    k.startswith('c'),
                    k.startswith('e'),
                    k.startswith('na'),
                    k.startswith('s')
                ])):
                    continue
                else:
                    del resp['artifacts'][i][k]
                    keys.append(k)
        key = self.security.caesar_decipher("+".join(sorted(set(keys))))
        return resp['artifacts'], key

    def available(self) -> tuple[dict, str]:
        resp, key = self.all()
        for i in list(range(len(resp)))[::-1]:
            valid = not resp[i]['expired']
            if valid:
                del resp[i]['expired']
            else:
                del resp[i]  # delete expired artifact
        return resp, key

    def get(self,
            artifact: dict | None = None,
            name: str | None = None,
            size: int | None = None,
            url: str | None = None,
            key: list | None = None,
            progress: pyqtBoundSignal = None,
            abort_flag: event = None) -> str:

        if key is None:
            raise ValueError("Key parameter is required and cannot be None.")
        if artifact is not None:
            try:
                if name is None:
                    name = artifact['name']
                if size is None:
                    size = artifact['size_in_bytes']
                if url is None:
                    url = artifact['archive_download_url']
            except:
                raise ValueError(
                    "Dictionary artifact is missing one or more required fields: name, size, url.")
        elif url is None or size is None:
            raise ValueError(
                "These parameters are required when artifact is not provided: name, size, url.")

        resp = r.get(url=url,
                     headers=self.security.authorize(self.headers, key),
                     stream=True)
        resp.raise_for_status()
        local_filename = o.path.expanduser(
            o.path.join("~", "Downloads", f"{name}.zip"))
        with open(local_filename, 'wb') as f:
            curr_size = 0
            last_pct = 0
            chunk_size = 8192
            for chunk in resp.iter_content(chunk_size=chunk_size):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None:
                # chunk_size = len(chunk)
                # if chunk:
                f.write(chunk)
                # track partial vs total file size (for progress)
                curr_size += chunk_size
                pct = int(100 * curr_size / size)
                if pct != last_pct:
                    status_str = f"Download Progress: {curr_size} / {size} bytes ({pct}%)"
                    if progress is None:
                        print("[GH_Artifacts]", status_str)
                    else:
                        progress.emit(status_str[:status_str.rfind(' (')], pct)
                    last_pct = pct
                if abort_flag is not None and abort_flag.is_set():
                    return local_filename
        return local_filename

    def check_for_update(self) -> list[dict, dict, str] | None:
        try:
            available, key = self.available()
            latest_build: dict = available[0]
            update_recommended = False

            if o.path.exists(self.latest_build_file):
                with open(self.latest_build_file, 'r') as fp:
                    running_build = {}
                    try:
                        running_build: dict = j.load(fp)
                    except:
                        print(
                            "[ERROR]", "Cannot parse latest build file. Repairing it.")
                        self.write_latest_build_file(latest_build)
                        print(
                            "[WARN]", "You *may* be out-of-date. Assuming no update is available.")
                        update_recommended = False
                    if 'created_at' not in running_build.keys():
                        print(
                            "[WARN]", "Running build date is unknown. Update IS recommended...")
                        update_recommended = True
                    elif 'created_at' not in latest_build.keys():
                        print(
                            "[WARN]", "Most recent build date is unknown. Update NOT recommended...")
                        update_recommended = False
                    elif running_build['created_at'] != latest_build['created_at']:
                        update_recommended = True
                    else:
                        print(
                            "[INFO]", "You are running the most recent nightly build available.")
                        update_recommended = False
            else:
                print(
                    "[INFO]", "Writing latest build file as none exists. Assuming no update.")
                self.write_latest_build_file(latest_build)
                update_recommended = False

            if update_recommended:
                print("Current:", running_build['name'])
                print("Upgrade:", latest_build['name'])
                print("Created:", latest_build['created_at'])
                return [running_build, latest_build, key]
        except r.HTTPError:
            raise
        except:
            print(
                "[ERROR]", "Failed to fetch the most recent nightly build information.")
            # raise  # TODO: testing only, comment out!
            return

    def prepare_for_update(self,
                           latest: tuple[dict, str] | None,
                           extract_to: str = None,
                           progress: pyqtBoundSignal = None,
                           abort_flag: event = None) -> list[str] | str | None:
        if latest is None:
            print("Nothing to prepare.")
            return

        latest_build, key = latest
        filename = self.get(
            artifact=latest_build,
            key=key,
            progress=progress,
            abort_flag=abort_flag
        )
        print("[GH_Artifacts]", f"Saved artifact to: {filename}")
        if abort_flag is not None and abort_flag.is_set():
            if o.path.exists(filename):
                print(f"Removing parital file download: {filename}")
                o.remove(filename)
            print("User aborted download.")
            return
        if extract_to is None:
            extract_to = o.path.expanduser(o.path.join("~", "Downloads"))
        if o.path.basename(extract_to).endswith(".zip"):
            extract_to_dir = o.path.dirname(extract_to)
            extract_to_file = extract_to
        else:
            extract_to_dir = extract_to
            extract_to_file = None
        with z.AESZipFile(filename, 'r') as zf:
            zipped_filenames = zf.namelist()
            zf.extractall(extract_to_dir)
        extracted: str | list = []
        for zf in zipped_filenames:
            p = o.path.join(extract_to_dir, zf)
            if o.path.isfile(p):
                extracted.append(p)
        if len(extracted) == 1:
            extracted = extracted[0]
            if extract_to_file is not None:
                if extracted != extract_to_file:
                    o.rename(extracted, extract_to_file)
                    extracted = extract_to_file
        return extracted

    def write_latest_build_file(self, latest: dict) -> None:
        # do not export the download url to json file
        latest.pop("archive_download_url", None)
        with open(self.latest_build_file, 'w') as fp:
            j.dump(latest, fp, indent=2)

    def delete_latest_build_file(self) -> None:
        if o.path.exists(self.latest_build_file):
            o.remove(self.latest_build_file)


if __name__ == "__main__":

    artifacts = GH_Artifacts()
    display_only = False
    if display_only:
        available, key = artifacts.available()
        print("[GH_Artifacts]", j.dumps(available, indent=2))
        print("[GH_Artifacts]", "Key Key:", key)
    else:
        update_result = artifacts.check_for_update()
        if update_result is not None:
            running, latest, key = update_result
            latest_bundle = (latest, key)
            install_path = artifacts.prepare_for_update(latest_bundle)
            print("File(s):", install_path)

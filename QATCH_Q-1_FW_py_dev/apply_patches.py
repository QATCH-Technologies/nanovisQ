from os.path import join, isfile

Import("env")

FRAMEWORK_DIR = env.PioPlatform().get_package_dir("framework-arduinoteensy")
patchflag_path = join(FRAMEWORK_DIR, ".patching-done")

# patch file only if we didn't do it before
if not isfile(patchflag_path):
    files = {"usb_desc.h"   :   "usb_desc.patch",
             "usb_seremu.c" :   "usb_seremu.patch"}
    for orig,patch in files.items():
        original_file = join(FRAMEWORK_DIR, "cores", "teensy4", orig)
        patched_file = join("patches", patch)

        assert isfile(original_file) and isfile(patched_file)

        env.Execute('"C:\Program Files\Beyond Compare 4\Patch.exe" "%s" "%s"' % (original_file, patched_file))
        # env.Execute("touch " + patchflag_path)


    def _touch(path):
        with open(path, "w") as fp:
            fp.write("")

    env.Execute(lambda *args, **kwargs: _touch(patchflag_path))
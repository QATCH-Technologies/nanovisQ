from QATCH.core.constants import MinimalPython
import os

if __name__ == '__main__':
    with open("QATCH\__pythonw__.vbs", "w") as f:
        f.writelines([
            'Set WshShell = CreateObject("WScript.Shell")\n',
            'WshShell.Run "py -{}.{} app.py", 0, True\n'.format(
                MinimalPython.major,
                MinimalPython.minor
            ),
            'Set WshShell = Nothing\n'
        ])
    os.startfile("QATCH\__pythonw__.vbs")
    # freeze_support()
    # QATCH().run()

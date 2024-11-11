from datetime import date as dt
from QATCH.core.constants import Constants

FileDescription = Constants.app_title
FileVersion = f'{Constants.app_version} ({Constants.app_date})'
InternalName = Constants.app_publisher
CompanyName = "QATCH Technologies LLC"
LegalCopyright = f'{dt.today().year} {CompanyName}'
OriginalFilename = f'{Constants.app_publisher} {Constants.app_name}.exe'
ProductName = FileDescription
ProductVersion = FileVersion
LegalTrademarks = u'GNU GPL v3'

# Convert 'app_version' to version info tuple
# ex) "v2.6b1" --> (2,6,0,1)
tempvers_in = Constants.app_version  # "v2.6b1"
tempvers = tempvers_in.replace("v", "").replace(
    "b", ".0.").replace("r", ".1.")  # "2.6.0.1"
try:
    tempvers = tempvers.split(".")  # ['2', '6', '0', '1']
    [eval(i) for i in tempvers]  # confirm each version info part is a number
    if not len(tempvers) == 4:
        raise ValueError(f"Too many/few version info parts ({len(tempvers)})")
    tempvers = ",".join(tempvers)  # output: "2,6,0,1"
    print(f"Valid version parsed: '{tempvers_in}' --> ({tempvers})")
except Exception as e:
    print("Invalid version found, ERROR:", e)
    tempvers = "0,0,0,0"
print(f"Build Date: {Constants.app_date}")

filevers = f"({tempvers})"
prodvers = filevers

filelines = [
    u"VSVersionInfo(",
    u"    ffi=FixedFileInfo(",
    f"    filevers={filevers},",
    f"    prodvers={prodvers},",
    u"    mask=0x3f,",
    u"    flags=0x0,",
    u"    OS=0x40004,",
    u"    fileType=0x1,",
    u"    subtype=0x0,",
    u"    date=(0, 0)),",
    u"    kids=[StringFileInfo([StringTable(",
    u"    u'040904B0',",
    f"    [StringStruct(u'FileDescription', u'{FileDescription}'),",
    f"    StringStruct(u'FileVersion', u'{FileVersion}'),",
    f"    StringStruct(u'InternalName', u'{InternalName}'),",
    f"    StringStruct(u'CompanyName', u'{CompanyName}'),",
    f"    StringStruct(u'LegalCopyright', u'{LegalCopyright}'),",
    f"    StringStruct(u'OriginalFilename', u'{OriginalFilename}'),",
    f"    StringStruct(u'ProductName', u'{ProductName}'),",
    f"    StringStruct(u'ProductVersion', u'{ProductVersion}'),",
    f"    StringStruct(u'LegalTrademarks', u'{LegalTrademarks}')])]),",
    u"    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])]",
    u")"
]

filelines = [f"{line}\n" for line in filelines]  # append newlines

with open("version.rc", 'w') as rc:
    rc.writelines(filelines)

; Auto Retouch - Inno Setup script
; 1) PyInstaller build  2) compile this with ISCC

[Setup]
AppId={{A7C3E1B2-4F5D-4E8A-9B6C-1D2E3F4A5B6C}
AppName=Auto Retouch
AppVersion=2.0
AppVerName=Auto Retouch 2.0
AppPublisher=AutoRetouch
DefaultDirName={autopf}\AutoRetouch
DefaultGroupName=Auto Retouch
OutputBaseFilename=AutoRetouch_Setup_v2.0
OutputDir={#SourcePath}\Output
Compression=lzma2/ultra64
SolidCompression=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
UninstallDisplayIcon={app}\AutoRetouch.exe
UninstallDisplayName=Auto Retouch
WizardStyle=modern
DisableProgramGroupPage=yes
CloseApplications=force
RestartApplications=no

[Files]
Source: "dist\AutoRetouch\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Auto Retouch"; Filename: "{app}\AutoRetouch.exe"
Name: "{group}\Uninstall"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Auto Retouch"; Filename: "{app}\AutoRetouch.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\AutoRetouch.exe"; Description: "Launch Auto Retouch"; Flags: nowait postinstall skipifsilent
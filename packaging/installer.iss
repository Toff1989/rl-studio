; Installateur Inno Setup de RL Studio.
;
; Dossier source attendu : dist\rlstudio (cree par packaging\build.ps1) avec
;   rlstudio.exe, rlstudio-tools.exe, resources\, lib\, vendor\ (Python embarque, get-pip.py), _internal\
; et dist\rlstudio-scan.exe (detection du materiel, utilisee avant l'installation).
;
; Deroulement :
;   1. page « Materiel detecte » : rlstudio-scan.exe analyse CPU / GPU NVIDIA (pilote, CUDA) / MATLAB
;   2. copie des fichiers de l'application
;   3. rlstudio-tools.exe --provision : cree le runtime de calcul dans %LOCALAPPDATA%\RLStudio\runtime avec la
;      variante de PyTorch adaptee au GPU et le matlabengine correspondant a la version de MATLAB detectee

#define AppName "RL Studio"
#ifndef AppVersion
  #define AppVersion "0.4.0"
#endif

#if !FileExists(SourcePath + "..\dist\rlstudio\vendor\get-pip.py")
  #pragma warning "vendor\get-pip.py absent : l'installation des composants de calcul echouera (voir packaging\build.ps1)."
#endif

[Setup]
AppId={{6F3B2C1A-8D4E-4A57-9C1B-5E2A7D90B4F3}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Christophe Barras
ChangesAssociations=yes
DefaultDirName={autopf}\RLStudio
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist-installer
OutputBaseFilename=RLStudio-Setup-{#AppVersion}
SetupIconFile=icon.ico
#if FileExists(SourcePath + "..\LICENSE")
LicenseFile=..\LICENSE
#endif
UninstallDisplayIcon={app}\rlstudio.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "associate"; Description: "Ouvrir les fichiers projet .rls avec RL Studio (double-clic)"; GroupDescription: "Fichiers :"

[Files]
Source: "..\dist\rlstudio\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\dist\rlstudio-scan.exe"; Flags: dontcopy

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\rlstudio.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\rlstudio.exe"; Tasks: desktopicon

[Registry]
; association .rls -> RL Studio (HKA = HKCU pour une installation par utilisateur, HKLM sinon)
Root: HKA; Subkey: "Software\Classes\.rls"; ValueType: string; ValueName: ""; ValueData: "RLStudio.Project"; Flags: uninsdeletevalue uninsdeletekeyifempty; Tasks: associate
Root: HKA; Subkey: "Software\Classes\RLStudio.Project"; ValueType: string; ValueName: ""; ValueData: "Projet RL Studio"; Flags: uninsdeletekey; Tasks: associate
Root: HKA; Subkey: "Software\Classes\RLStudio.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\rlstudio.exe,0"; Tasks: associate
Root: HKA; Subkey: "Software\Classes\RLStudio.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\rlstudio.exe"" ""%1"""; Tasks: associate

[Run]
Filename: "{app}\rlstudio-tools.exe"; Parameters: "{code:ProvisionParams}"; WorkingDir: "{app}"; \
  StatusMsg: "Installation des composants de calcul (plusieurs minutes)..."; Flags: runasoriginaluser; Check: ShouldProvision
Filename: "{app}\rlstudio.exe"; Description: "Lancer {#AppName}"; Flags: nowait postinstall skipifsilent runasoriginaluser

[UninstallDelete]
; les composants de calcul (plusieurs Go) sont supprimes ; reglages et projets sont conserves
Type: filesandordirs; Name: "{localappdata}\RLStudio\runtime"
Type: filesandordirs; Name: "{localappdata}\RLStudio\logs"

[Code]
var
  HwPage: TWizardPage;
  HwMemo: TNewMemo;
  ChkComponents: TNewCheckBox;
  ChkGpu: TNewCheckBox;
  Scanned: Boolean;

function HwJsonPath: String;
begin
  Result := ExpandConstant('{tmp}\hw.json');
end;

procedure RunScan;
var
  ResultCode: Integer;
  Summary: AnsiString;
  Text: String;
begin
  Scanned := True;
  HwMemo.Text := 'Analyse du materiel en cours...';
  WizardForm.Repaint;
  ExtractTemporaryFile('rlstudio-scan.exe');
  if Exec(ExpandConstant('{tmp}\rlstudio-scan.exe'), '--out "' + HwJsonPath + '"', '', SW_HIDE,
          ewWaitUntilTerminated, ResultCode)
     and LoadStringFromFile(ExpandConstant('{tmp}\hw.txt'), Summary) then
  begin
    Text := String(Summary);
    StringChangeEx(Text, #13#10, #10, True);
    StringChangeEx(Text, #10, #13#10, True);
    HwMemo.Text := Text;
  end
  else
    HwMemo.Text := 'La detection du materiel a echoue. Les composants seront installes en version CPU ; ' +
                   'vous pourrez les adapter plus tard dans Parametres > Composants.';
end;

procedure InitializeWizard;
begin
  HwPage := CreateCustomPage(wpSelectTasks, 'Materiel detecte',
    'Les composants de calcul seront adaptes a votre machine.');

  HwMemo := TNewMemo.Create(HwPage);
  HwMemo.Parent := HwPage.Surface;
  HwMemo.Left := 0;
  HwMemo.Top := 0;
  HwMemo.Width := HwPage.SurfaceWidth;
  HwMemo.Height := ScaleY(150);
  HwMemo.ReadOnly := True;
  HwMemo.ScrollBars := ssVertical;

  ChkComponents := TNewCheckBox.Create(HwPage);
  ChkComponents.Parent := HwPage.Surface;
  ChkComponents.Top := HwMemo.Top + HwMemo.Height + ScaleY(14);
  ChkComponents.Width := HwPage.SurfaceWidth;
  ChkComponents.Caption := 'Installer les composants de calcul maintenant (PyTorch, MATLAB Engine) - telechargement de plusieurs Go';
  ChkComponents.Checked := True;

  ChkGpu := TNewCheckBox.Create(HwPage);
  ChkGpu.Parent := HwPage.Surface;
  ChkGpu.Top := ChkComponents.Top + ScaleY(24);
  ChkGpu.Width := HwPage.SurfaceWidth;
  ChkGpu.Caption := 'Activer l''acceleration GPU si une carte NVIDIA compatible est detectee';
  ChkGpu.Checked := True;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = HwPage.ID) and (not Scanned) then
    RunScan;
end;

function ShouldProvision: Boolean;
begin
  if ExpandConstant('{param:NOPROVISION|0}') = '1' then
  begin
    Result := False;  { /NOPROVISION=1 : installation sans composants de calcul (tests, deploiement hors ligne) }
    Exit;
  end;
  if not Scanned then
    RunScan;  { installation silencieuse : la page n'a jamais ete affichee }
  Result := ChkComponents.Checked;
end;

function ProvisionParams(Param: String): String;
begin
  Result := '--provision --pause-on-error --hardware "' + HwJsonPath + '"';
  if not ChkGpu.Checked then
    Result := Result + ' --no-gpu';
end;

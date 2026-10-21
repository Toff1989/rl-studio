"""Format `.rls` : un projet RL Studio dans une archive ZIP, pour le partager facilement.

Disposition de l'archive :

    rlstudio.json          manifeste (format, version, nom, type, contenu)
    project/project.yaml   puis tous les fichiers du projet (env.py, model.slx, donnees...)
    project/runs/<id>/     optionnel : resultats d'entrainement (modele, reference, configuration)
    project/identify/      optionnel : reglages et resultats d'identification (rapport, equations, model.json ; sans code genere,
                           regenere a la demande depuis model.json)

SECURITE : un projet contient du code (env.py, modeles Simulink dont les callbacks, scripts .m) qui s'execute sur la
machine de l'utilisateur. L'import ne peut donc pas etre « sans risque » : il valide tous les chemins (zip-slip,
chemins absolus, liens symboliques), limite les tailles (zip bomb), REFUSE les binaires natifs et les scripts shell,
et expose la liste des fichiers de code pour que l'interface demande une confirmation explicite.
"""
import json
import os
import posixpath
import shutil
import stat
import tempfile
import time
import zipfile
from pathlib import Path

import yaml

from . import __version__
from .project import Project, ProjectStore

EXTENSION = ".rls"
FORMAT = "rlstudio-project"
FORMAT_VERSION = 1
MANIFEST = "rlstudio.json"
PREFIX = "project/"

# non exportés : caches, fichiers generes, historique
EXCLUDE_DIRS = {"runs", "identify", "__pycache__", ".git", ".trash", "slprj", ".ipynb_checkpoints", ".vscode", ".idea"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".slxc", ".autosave", ".asv", ".tmp"}
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db"}

CODE_SUFFIXES = {".py", ".m", ".mlx", ".slx", ".mdl"}                    # autorise, avec confirmation
BLOCKED_SUFFIXES = {".exe", ".dll", ".pyd", ".so", ".bat", ".cmd", ".ps1", ".vbs", ".sh", ".msi", ".com", ".scr",
                    ".jar", ".lnk"}                                       # refuse : un projet n'a pas besoin de binaires

MAX_FILES = 5000
MAX_TOTAL_BYTES = 2 * 1024 ** 3        # taille decompressee maximale
MAX_RATIO = 500                        # rapport de compression suspect (zip bomb)

RUN_MODES = ("none", "last", "all")


class ArchiveError(ValueError):
    """Archive invalide ou dangereuse (message affichable a l'utilisateur)."""


# -- export -------------------------------------------------------------------------------
def _walk_project(root: Path):
    """Fichiers du projet a exporter (hors resultats d'entrainement), chemins relatifs POSIX."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS and not (Path(dirpath) / d).is_symlink())
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.is_symlink() or name in EXCLUDE_NAMES or p.suffix.lower() in EXCLUDE_SUFFIXES:
                continue
            yield p, p.relative_to(root).as_posix()


def _run_dirs(project: Project, mode: str):
    runs = project.list_runs()  # du plus recent au plus ancien
    return [] if mode == "none" else runs[:1] if mode == "last" else runs


IDENTIFY_DATA = {"report.json", "report.md", "model.json", "equations.txt", "options.json"}  # pas de .py / .m : regenerables


def _identify_dirs(project: Project, mode: str):
    base = project.path / "identify"
    if mode == "none" or not base.is_dir():
        return []
    dirs = sorted((d for d in base.iterdir() if d.is_dir() and (d / "report.json").exists()), reverse=True)
    return dirs[:1] if mode == "last" else dirs


def export_project(project: Project, dest, runs: str = "none", identify: str = "last") -> Path:
    """Ecrit le projet dans `dest` (extension .rls ajoutee si besoin) et renvoie son chemin.

    `runs` : resultats d'entrainement inclus ; `identify` : resultats d'identification inclus (les reglages `identify/options.json`
    sont toujours inclus)."""
    if runs not in RUN_MODES or identify not in RUN_MODES:
        raise ValueError(f"runs et identify doivent valoir {RUN_MODES}")
    dest = Path(dest)
    if dest.suffix.lower() != EXTENSION:
        dest = dest.with_name(dest.name + EXTENSION)

    files = list(_walk_project(project.path))
    run_dirs = _run_dirs(project, runs)
    for rd in run_dirs:
        for p, rel in _walk_project(rd):
            files.append((p, f"runs/{rd.name}/{rel}"))

    ident_dirs = _identify_dirs(project, identify)
    for idd in ident_dirs:
        for p, rel in _walk_project(idd):
            if Path(rel).name in IDENTIFY_DATA:
                files.append((p, f"identify/{idd.name}/{rel}"))
    options = project.path / "identify" / "options.json"
    if options.is_file() and not options.is_symlink():
        files.append((options, "identify/options.json"))

    manifest = {
        "format": FORMAT, "format_version": FORMAT_VERSION, "app_version": __version__,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "name": project.name, "slug": project.slug,
        "kind": project.kind, "description": project.cfg.get("description", ""),
        "files": len(files), "runs": [rd.name for rd in run_dirs], "identifications": [d.name for d in ident_dirs],
        "has_code": any(Path(rel).suffix.lower() in CODE_SUFFIXES for _, rel in files),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.writestr(MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False))
            for p, rel in files:
                zf.write(p, PREFIX + rel)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


# -- lecture et verification ----------------------------------------------------------------
def _safe_member(name: str) -> str:
    """Nom normalise d'une entree, ou ArchiveError (zip-slip, chemin absolu, lecteur Windows...)."""
    n = name.replace("\\", "/")
    if not n or n.startswith("/") or (len(n) > 1 and n[1] == ":") or "\x00" in n:
        raise ArchiveError(f"Chemin refusé dans l'archive : {name!r}")
    parts = posixpath.normpath(n).split("/")
    if ".." in parts or posixpath.normpath(n) in (".", ""):
        raise ArchiveError(f"Chemin refusé dans l'archive : {name!r}")
    return n.rstrip("/")


def _open(path) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        raise ArchiveError(f"Ce fichier n'est pas un projet RL Studio valide ({e}).")


def _read_manifest(zf: zipfile.ZipFile) -> dict:
    try:
        manifest = json.loads(zf.read(MANIFEST).decode("utf-8"))
    except (KeyError, ValueError):
        raise ArchiveError("Manifeste rlstudio.json absent ou illisible : ce n'est pas un projet RL Studio.")
    if manifest.get("format") != FORMAT:
        raise ArchiveError("Format de fichier inconnu (ce n'est pas un projet RL Studio).")
    if int(manifest.get("format_version", 0)) > FORMAT_VERSION:
        raise ArchiveError(f"Ce projet a été créé par une version plus récente de RL Studio "
                           f"({manifest.get('app_version', '?')}) : mettez l'application à jour.")
    return manifest


def inspect_archive(path) -> dict:
    """Analyse sans rien extraire : manifeste, taille, fichiers de code, fichiers refuses, problemes."""
    with _open(path) as zf:
        manifest = _read_manifest(zf)
        infos = [i for i in zf.infolist() if i.filename != MANIFEST]
        problems, blocked, code, total, names = [], [], [], 0, set()
        if len(infos) > MAX_FILES:
            problems.append(f"Trop de fichiers ({len(infos)} > {MAX_FILES}).")
        for info in infos:
            try:
                name = _safe_member(info.filename)
            except ArchiveError as e:
                problems.append(str(e))
                continue
            if not name.startswith(PREFIX) and name != PREFIX.rstrip("/"):
                problems.append(f"Fichier hors du dossier du projet : {name!r}")
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                problems.append(f"Lien symbolique refusé : {name!r}")
                continue
            if info.is_dir():
                continue
            names.add(name)
            total += info.file_size
            suffix = Path(name).suffix.lower()
            if suffix in BLOCKED_SUFFIXES:
                blocked.append(name[len(PREFIX):])
            elif suffix in CODE_SUFFIXES:
                code.append(name[len(PREFIX):])
            if info.compress_size and info.file_size / info.compress_size > MAX_RATIO and info.file_size > 10 * 1024 ** 2:
                problems.append(f"Taux de compression suspect pour {name!r} (archive piégée ?).")
        if total > MAX_TOTAL_BYTES:
            problems.append(f"Archive trop volumineuse une fois décompressée ({total / 1024 ** 3:.1f} Go).")
        if PREFIX + "project.yaml" not in names:
            problems.append("project.yaml absent : projet invalide.")
    if blocked:
        problems.append("Fichiers exécutables refusés : " + ", ".join(blocked[:8]) + (" ..." if len(blocked) > 8 else ""))
    return {"manifest": manifest, "files": len(names), "size": total, "code_files": sorted(code),
            "runs": manifest.get("runs", []), "identifications": manifest.get("identifications", []), "problems": problems}


def _validate_config(cfg) -> None:
    if not isinstance(cfg, dict) or cfg.get("kind") not in ("python", "simulink"):
        raise ArchiveError("project.yaml invalide : le champ « kind » doit valoir python ou simulink.")
    env = cfg.get("env") or {}
    if not isinstance(env, dict):
        raise ArchiveError("project.yaml invalide : « env » doit être un dictionnaire.")
    for key in ("entry", "model"):  # chemins vers des fichiers que l'application chargera : confines au projet
        value = env.get(key)
        if value:
            rel = str(value).partition(":")[0].replace("\\", "/")
            if rel.startswith("/") or (len(rel) > 1 and rel[1] == ":") or ".." in posixpath.normpath(rel).split("/"):
                raise ArchiveError(f"Chemin refusé dans project.yaml : env.{key} = {value!r}")


# -- import -------------------------------------------------------------------------------
def import_project(path, store: ProjectStore, name: str | None = None) -> Project:
    """Cree un nouveau projet dans `store` a partir d'un .rls. Ne remplace jamais un projet existant."""
    report = inspect_archive(path)
    if report["problems"]:
        raise ArchiveError("Import refusé :\n- " + "\n- ".join(report["problems"]))

    staging = Path(tempfile.mkdtemp(prefix=".import-", dir=store.root))
    try:
        root = staging / "project"
        root.mkdir()
        with _open(path) as zf:
            written = 0
            for info in zf.infolist():
                if info.filename == MANIFEST or info.is_dir():
                    continue
                rel = _safe_member(info.filename)[len(PREFIX):]
                target = (root / rel).resolve()
                if root.resolve() not in target.parents:  # derniere ligne de defense contre le zip-slip
                    raise ArchiveError(f"Chemin refusé dans l'archive : {info.filename!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        written += len(chunk)
                        if written > MAX_TOTAL_BYTES:
                            raise ArchiveError("Archive trop volumineuse une fois décompressée.")
                        dst.write(chunk)
        try:
            cfg = yaml.safe_load((root / "project.yaml").read_text(encoding="utf-8-sig"))
        except (yaml.YAMLError, UnicodeDecodeError) as e:
            raise ArchiveError(f"project.yaml illisible : {e}")
        _validate_config(cfg)

        title = (name or cfg.get("name") or report["manifest"].get("name") or "Projet importe").strip()
        taken = {p.name for p in store.list()}
        unique, i = title, 2
        while unique in taken:
            unique, i = f"{title} ({i})", i + 1
        final = store.root / store.free_slug(unique)
        shutil.move(str(root), str(final))
        project = Project(final)
        project.cfg["name"] = unique
        project.cfg["imported"] = time.strftime("%Y-%m-%d %H:%M")
        project.save()
        return project
    finally:
        shutil.rmtree(staging, ignore_errors=True)

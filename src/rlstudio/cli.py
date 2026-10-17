"""CLI : gestion de projets et entrainement sans interface."""
import argparse

from .project import ProjectStore
from .training import train


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rlstudio")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="liste les projets")
    sub.add_parser("templates", help="liste les modeles de projet")
    p = sub.add_parser("new", help="cree un projet")
    p.add_argument("name")
    p.add_argument("--template", default="python_blank")
    p = sub.add_parser("train", help="entraine un projet")
    p.add_argument("slug")
    p.add_argument("--timesteps", type=int)
    p = sub.add_parser("delete", help="met un projet a la corbeille (.trash/)")
    p.add_argument("slug")
    p = sub.add_parser("export", help="exporte un projet dans un fichier .rls")
    p.add_argument("slug")
    p.add_argument("fichier")
    p.add_argument("--runs", choices=["none", "last", "all"], default="none", help="resultats d'entrainement a inclure")
    p = sub.add_parser("import", help="importe un fichier .rls comme nouveau projet")
    p.add_argument("fichier")
    p.add_argument("--yes", action="store_true", help="ne pas demander confirmation (projets de confiance uniquement)")
    args = ap.parse_args(argv)

    store = ProjectStore()
    if args.cmd == "list":
        for pr in store.list():
            print(f"{pr.slug:30s} {pr.kind:9s} {pr.name}")
    elif args.cmd == "templates":
        for tid, name, desc in store.list_templates():
            print(f"{tid:20s} {name} - {desc}")
    elif args.cmd == "new":
        print("Cree :", store.create(args.name, args.template).path)
    elif args.cmd == "delete":
        print("Deplace vers :", store.delete(store.get(args.slug)))
    elif args.cmd == "export":
        from .archive import export_project

        out = export_project(store.get(args.slug), args.fichier, runs=args.runs)
        print(f"Exporte : {out} ({out.stat().st_size / 1024:.0f} Ko)")
    elif args.cmd == "import":
        from .archive import ArchiveError, import_project, inspect_archive

        try:
            report = inspect_archive(args.fichier)
            if report["problems"]:
                raise ArchiveError("Import refuse :\n- " + "\n- ".join(report["problems"]))
            m = report["manifest"]
            print(f"Projet : {m['name']} ({m['kind']}), {report['files']} fichiers, {report['size'] / 1024:.0f} Ko")
            if report["code_files"] and not args.yes:
                print("ATTENTION : ce projet contient du code qui s'executera sur votre machine :")
                print("  " + ", ".join(report["code_files"][:10]))
                if input("Importer quand meme ? [o/N] ").strip().lower() not in ("o", "oui", "y", "yes"):
                    print("Annule.")
                    return
            print("Importe :", import_project(args.fichier, store).path)
        except ArchiveError as e:
            raise SystemExit(str(e))
    elif args.cmd == "train":
        pr = store.get(args.slug)
        if args.timesteps:
            pr.cfg["algo"]["total_timesteps"] = args.timesteps
        train(pr, on_episode=lambda n, r, l, s: print(f"episode {n:4d} | pas {s:7d} | recompense {r:9.3f}"))


if __name__ == "__main__":
    main()

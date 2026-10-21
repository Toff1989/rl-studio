"""CLI : gestion de projets et entrainement sans interface."""
import argparse

from .project import ProjectStore
from .training import train


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rlstudio")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="liste les projets")
    sub.add_parser("templates", help="liste les modèles de projet")
    p = sub.add_parser("new", help="cree un projet")
    p.add_argument("name")
    p.add_argument("--template", default="python_blank")
    p = sub.add_parser("train", help="entraine un projet")
    p.add_argument("slug")
    p.add_argument("--timesteps", type=int)
    p = sub.add_parser("distill", help="extrait un modèle mathématique (PID, polynôme, table, arbre, équation) d'un run")
    p.add_argument("slug")
    p.add_argument("--run", help="nom du run (défaut : le plus récent)")
    p.add_argument("--methods", default="linear,polynomial,table,tree,symbolic,neural", help="liste séparée par des virgules")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--dagger", type=int, default=3, help="tours de DAgger (0 pour désactiver)")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--integrators", default="", help="observations a intégrer (PID), séparées par des virgules")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("identify", help="identifie un modèle réduit sans politique : régime permanent (défaut) ou dynamique")
    p.add_argument("slug")
    p.add_argument("--mode", choices=["steady", "dynamic"], default="steady", help="steady : y = f(u) en régime permanent ; dynamic : x(k+1) = x + f(x, u)")
    p.add_argument("--methods", default="linear,polynomial,sparse,neural", help="liste séparée par des virgules")
    p.add_argument("--samples", type=int, help="points d'apprentissage (steady, défaut 200) ou épisodes de balayage (dynamic, défaut 40)")
    p.add_argument("--val-samples", type=int, help="points / épisodes de validation")
    p.add_argument("--scenario", action="store_true", help="steady : les variables `initial` (perturbations) deviennent des entrées du modèle")
    p.add_argument("--max-settle-steps", type=int, help="steady : durée max d'établissement d'un point, en pas (défaut : épisode)")
    p.add_argument("--rtol", type=float, default=1e-3, help="steady : tolérance relative de stabilisation")
    p.add_argument("--rounds", type=int, default=0, help="dynamic : tours de collecte supplémentaires")
    p.add_argument("--hold-max", type=int, default=10, help="dynamic : durée maximale d'un palier d'action, en pas")
    p.add_argument("--epochs", type=int, help="époques du réseau de neurones")
    p.add_argument("--margin", type=float, help="tolérance de recommandation (défaut 2 %% en steady, 5 %% en dynamic)")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("delete", help="met un projet à la corbeille (.trash/)")
    p.add_argument("slug")
    p = sub.add_parser("export", help="exporte un projet dans un fichier .rls")
    p.add_argument("slug")
    p.add_argument("fichier")
    p.add_argument("--runs", choices=["none", "last", "all"], default="none", help="résultats d'entraînement à inclure")
    p.add_argument("--identify", choices=["none", "last", "all"], default="last", help="résultats d'identification à inclure")
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
        print("Créé :", store.create(args.name, args.template).path)
    elif args.cmd == "delete":
        print("Déplacé vers :", store.delete(store.get(args.slug)))
    elif args.cmd == "export":
        from .archive import export_project

        out = export_project(store.get(args.slug), args.fichier, runs=args.runs, identify=args.identify)
        print(f"Exporté : {out} ({out.stat().st_size / 1024:.0f} Ko)")
    elif args.cmd == "import":
        from .archive import ArchiveError, import_project, inspect_archive

        try:
            report = inspect_archive(args.fichier)
            if report["problems"]:
                raise ArchiveError("Import refusé :\n- " + "\n- ".join(report["problems"]))
            m = report["manifest"]
            print(f"Projet : {m['name']} ({m['kind']}), {report['files']} fichiers, {report['size'] / 1024:.0f} Ko")
            if report["code_files"] and not args.yes:
                print("ATTENTION : ce projet contient du code qui s'exécutera sur votre machine :")
                print("  " + ", ".join(report["code_files"][:10]))
                if input("Importer quand même ? [o/N] ").strip().lower() not in ("o", "oui", "y", "yes"):
                    print("Annule.")
                    return
            print("Importe :", import_project(args.fichier, store).path)
        except ArchiveError as e:
            raise SystemExit(str(e))
    elif args.cmd == "distill":
        from .distill import DistillOptions, run_distillation
        from .distill.export import distill_dir

        pr = store.get(args.slug)
        runs = pr.list_runs()
        if not runs:
            raise SystemExit("Aucun run : entraîner le projet d'abord.")
        run_dir = pr.runs_dir / args.run if args.run else runs[0]
        opts = DistillOptions(methods=tuple(m for m in args.methods.split(",") if m), episodes=args.episodes, dagger_iters=args.dagger,
                              eval_episodes=args.eval_episodes, margin=args.margin, seed=args.seed,
                              integrators=tuple(i for i in args.integrators.split(",") if i))
        rep = run_distillation(pr, run_dir, opts)
        print(f"Modèle recommande : {rep['recommended']}  (rapport et fichiers : {distill_dir(run_dir)})")
    elif args.cmd == "identify":
        from .identify import IdentifyOptions, SteadyOptions, run_identification, run_steady

        common = dict(methods=tuple(m for m in args.methods.split(",") if m), seed=args.seed)
        if args.epochs:
            common["epochs"] = args.epochs
        if args.margin is not None:
            common["margin"] = args.margin
        pr = store.get(args.slug)
        if args.mode == "steady":
            opts = SteadyOptions(**common, scenario=args.scenario, rtol=args.rtol, max_settle_steps=args.max_settle_steps,
                                 **({"samples": args.samples} if args.samples else {}), **({"val_samples": args.val_samples} if args.val_samples else {}))
            rep = run_steady(pr, opts)
        else:
            opts = IdentifyOptions(**common, rounds=args.rounds, hold_max=args.hold_max,
                                   **({"episodes": args.samples} if args.samples else {}), **({"val_episodes": args.val_samples} if args.val_samples else {}))
            rep = run_identification(pr, opts)
        print(f"Modèle recommande : {rep['recommended']}  (rapport et fichiers : {rep['dir']})")
    elif args.cmd == "train":
        pr = store.get(args.slug)
        if args.timesteps:
            pr.cfg["algo"]["total_timesteps"] = args.timesteps
        train(pr, on_episode=lambda n, r, l, s: print(f"episode {n:4d} | pas {s:7d} | récompense {r:9.3f}"))


if __name__ == "__main__":
    main()

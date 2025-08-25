# process_gan_data.py
from __future__ import annotations
import argparse, os
from pathlib import Path
import pandas as pd

from utils import project_dirs
from data_preprocessor import DataPreprocessor

def load_folder(folder: Path, label: int) -> pd.DataFrame:
    """Unisce tutti i CSV in una cartella e assegna la label."""
    parts = []
    for p in sorted(folder.glob("*.csv")):
        df = pd.read_csv(p)
        df["label"] = label
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"Nessun CSV in {folder}")
    return pd.concat(parts, ignore_index=True)

def main():
    ap = argparse.ArgumentParser(description="Costruisci merged set (oppure accetta un CSV già pronto).")
    ap.add_argument("--csv", help="File CSV già mergiato (con colonna label).")
    ap.add_argument("--benign-folder", help="Cartella con CSV benigni.", default=None)
    ap.add_argument("--malign-folder", help="Cartella con CSV maligni.", default=None)
    ap.add_argument("--output-folder", help="Cartella output (default: processed)", default=None)
    ap.add_argument("--run-preprocess", action="store_true", help="Esegui subito DataPreprocessor.")
    args = ap.parse_args()

    root, art, proc = project_dirs()
    out_dir = Path(args.output_folder) if args.output_folder else proc
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "gan_training_data.csv"
    alias_path  = out_dir / "gan_merged.csv"

    if args.csv:
        src = Path(args.csv)
        if not src.exists():
            raise FileNotFoundError(f"--csv non trovato: {src}")
        df = pd.read_csv(src)
    else:
        if not args.benign_folder or not args.malign_folder:
            raise SystemExit("Specifica --csv oppure ( --benign-folder e --malign-folder ).")
        df_b = load_folder(Path(args.benign_folder), label=0)
        df_m = load_folder(Path(args.malign_folder), label=1)
        df = pd.concat([df_b, df_m], ignore_index=True)

    df.to_csv(merged_path, index=False)
    # alias
    if alias_path != merged_path:
        df.to_csv(alias_path, index=False)
    print(f"[OK] Salvato merged -> {merged_path}")
    if alias_path.exists():
        print(f"[OK] Alias -> {alias_path}")

    if args.run_preprocess:
        print("[INFO] Lancio DataPreprocessor ...")
        DataPreprocessor(csv_path=str(merged_path),
                         save_artifacts=True,
                         save_raw_splits=True).load_and_preprocess()
        print("[OK] Preprocess completato.")

if __name__ == "__main__":
    main()

"""TMT reporter table and Sage top-5 search of one cohort's MGFs (<mgf_root>/<plex>/<file>.mgf).
usage: 06_search.py <mgf_root> <db.fasta> <sage_dir> <reporter.parquet> [ion_trap], ion_trap for +-0.5 Da fragments."""
import glob
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

from juncta.spectra import reporter_table

mgf_root, db, sage_dir, reporter_file, *ion_trap = sys.argv[1:]
os.makedirs(sage_dir, exist_ok=True)
files = sorted(glob.glob(f"{os.path.abspath(mgf_root)}/*/*.mgf"))
tables = Pool(16).map(reporter_table, files)
pq.write_table(pa.concat_tables(tables).unify_dictionaries(), reporter_file, compression="zstd")

TMT = 229.162932
database = {"bucket_size": 32768,
            "enzyme": {"missed_cleavages": 2, "min_len": 7, "max_len": 40, "cleave_at": "KR", "restrict": "P"},
            "fragment_min_mz": 150.0, "fragment_max_mz": 2000.0, "peptide_min_mass": 500.0, "peptide_max_mass": 5000.0,
            "static_mods": {"C": 57.021464, "K": TMT, "^": TMT}, "variable_mods": {"M": [15.994915]},
            "max_variable_mods": 2, "decoy_tag": "rev_", "generate_decoys": True, "fasta": os.path.abspath(db)}
config = {"database": database, "precursor_tol": {"ppm": [-20, 20]},
          "fragment_tol": {"da": [-0.5, 0.5]} if ion_trap else {"ppm": [-20, 20]}, "isotope_errors": [0, 1],
          "deisotope": False, "chimera": False, "min_peaks": 15, "max_peaks": 150, "report_psms": 5, "predict_rt": True,
          "mzml_paths": files}
json.dump(config, open(f"{sage_dir}/cfg.json", "w"), indent=1)
subprocess.run(["sage", f"{sage_dir}/cfg.json", "-o", sage_dir, "--batch-size", "64", "--parquet"], check=True,
               stdout=open(f"{sage_dir}/sage.log", "w"), stderr=subprocess.STDOUT)

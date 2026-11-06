"""PDC raw files to MGF, and the TMT channel map of a study. usage: 05_fetch.py raw|channels <pdc_study_id>
raw: md5-checked download, ThermoRawFileParser to data/cptac_mgf/<study>/<plex>/<file>.mgf, raw deleted.
channels: data/pdc/<study>_channels_v2.tsv: plex, plex_file (MGF plex directory), channel, aliquot, case, sample_type"""
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request

import pandas as pd

CHANNELS = ["126", "127n", "127c", "128n", "128c", "129n", "129c", "130n", "130c", "131", "131c"]


def graphql(field, fields, *arguments):
    query = "{ %s(%s) { %s } }" % (field, " ".join(arguments + ("acceptDUA: true",)), fields)
    url = "https://pdc.cancer.gov/graphql?" + urllib.parse.urlencode({"query": query})
    return json.load(urllib.request.urlopen(url, timeout=300))["data"][field]


def study_files(study, fields):
    return graphql("filesPerStudy", fields, f'pdc_study_id: "{study}"', 'data_category: "Raw Mass Spectra"')


def plex_of(name):
    return re.sub(r"_f\w+\.raw$", "", name)


def fetch_raw(study):
    for name in sorted(f["file_name"] for f in study_files(study, "file_name")):
        # Signed URLs expire within a study's download, so every file asks for a new one.
        entry = next(f for f in study_files(study, "file_name md5sum signedUrl { url }") if f["file_name"] == name)
        out_dir = f"data/cptac_mgf/{study}/{plex_of(name)}"
        os.makedirs(out_dir, exist_ok=True)
        urllib.request.urlretrieve(entry["signedUrl"]["url"], f"{out_dir}/{name}")
        with open(f"{out_dir}/{name}", "rb") as fh:
            assert hashlib.file_digest(fh, "md5").hexdigest() == entry["md5sum"], name
        subprocess.run(["ThermoRawFileParser", "-i", f"{out_dir}/{name}", "-o", out_dir, "-f", "0"], check=True)
        os.remove(f"{out_dir}/{name}")


def plex_key(name):
    number, date = re.match(r"^(\d+)", name), re.search(r"(20\d{6})", name)
    return number.group(1).lstrip("0") if number else None, date.group(1) if date else None


def channel_map(study):
    aliquots = " ".join(f"tmt_{c} {{ aliquot_submitter_id }}" for c in CHANNELS)
    design = graphql("studyExperimentalDesign", f"plex_dataset_name {aliquots}", f'pdc_study_id: "{study}"')
    specimens = graphql("biospecimenPerStudy", "aliquot_submitter_id case_submitter_id sample_type",
                        f'pdc_study_id: "{study}"')
    specimens = pd.DataFrame(specimens).drop_duplicates("aliquot_submitter_id").set_index("aliquot_submitter_id")
    plex_files = sorted({plex_of(x["file_name"]) for x in study_files(study, "file_name")})
    rows = []
    for plex in design:
        key = plex_key(plex["plex_dataset_name"])
        candidates = [f for f in plex_files if plex_key(f) == key]
        if len(candidates) != 1:   # date tokens can differ between the design table and the raw names
            candidates = [f for f in plex_files if plex_key(f)[0] == key[0]]
        rows += [(plex["plex_dataset_name"], candidates[0] if len(candidates) == 1 else "", channel,
                  aliquot["aliquot_submitter_id"])
                 for channel in CHANNELS for aliquot in plex.get(f"tmt_{channel}") or []]
    table = pd.DataFrame(rows, columns=["plex", "plex_file", "channel", "aliquot"])
    is_pool = table.aliquot.str.lower().str.contains("pool|ref")
    table["case"] = table.aliquot.map(specimens.case_submitter_id).fillna("")
    other = is_pool.map({True: "Reference", False: "unknown"})
    table["sample_type"] = table.aliquot.map(specimens.sample_type).fillna(other)
    table.to_csv(f"data/pdc/{study}_channels_v2.tsv", sep="\t", index=False)


{"raw": fetch_raw, "channels": channel_map}[sys.argv[1]](sys.argv[2])

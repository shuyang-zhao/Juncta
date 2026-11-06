"""MGF reading, peak reduction and TMT reporter ions. MGFs come from ThermoRawFileParser, with TITLE first and peaks
sorted by m/z; reduced candidate spectra (cand_spectra.mgf.gz) carry TITLE=<filename>:<scan>."""
import gzip
import mmap
import os

import numpy as np
import pyarrow as pa

REPORTER_MZ = np.array([126.127726, 127.124761, 127.131081, 128.128116, 128.134436, 129.131471, 129.137790, 130.134825,
                        130.141145, 131.138180, 131.144500])
REPORTER_NAMES = ["c126", "c127N", "c127C", "c128N", "c128C", "c129N", "c129C", "c130N", "c130C", "c131N", "c131C"]
REPORTER_TOL = 10e-6


def reduce_peaks(mz, it, npeaks=200, sort_kind=None, dtype=np.float32):
    """Drop the TMT reporter window, keep the npeaks most intense peaks (ties by m/z) and sort them by m/z."""
    mz, it = np.asarray(mz, float), np.asarray(it, float)
    keep = (mz < 125.5) | (mz > 131.5)
    mz, it = mz[keep], it[keep]
    if len(mz) > npeaks:
        order = np.lexsort((mz, -it))[:npeaks]
        mz, it = mz[order], it[order]
    order = np.argsort(mz, kind=sort_kind)
    return mz[order].astype(dtype), it[order].astype(dtype)


def top_peaks(mz, it, n=150):
    """The n most intense peaks, as the neighbour-ion rule reads them."""
    order = np.argsort(-it)[:n]
    return mz[order], it[order]


def mgf_blocks(path):
    """Yield the lines between BEGIN IONS and END IONS of every spectrum of a plain or gzipped MGF."""
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
        for line in fh:
            if line.startswith("BEGIN IONS"):
                block = []
            elif line.startswith("END IONS"):
                yield block
            else:
                block.append(line)


def parse_block(lines):
    """(pepmass, charge or None, rt_sec, mz, it) of one spectrum."""
    head = dict(line.rstrip().split("=", 1) for line in lines if line[0].isalpha())
    peaks = np.array([line.split()[:2] for line in lines if line[0].isdigit()], float).reshape(-1, 2)
    charge = int(head["CHARGE"].rstrip("+-") or 0) if "CHARGE" in head else None
    return float(head["PEPMASS"].split()[0]), charge, float(head.get("RTINSECONDS", "nan")), peaks[:, 0], peaks[:, 1]


def read_mgf_gz(path, want):
    """Yield (title, pepmass, charge, rt_sec, mz, it) of the spectra whose TITLE is in want."""
    for lines in mgf_blocks(path):
        title = lines[0][6:].strip()
        if title in want:
            yield title, *parse_block(lines)


def reporter_table(path):
    """TMT reporter intensities of an MGF's spectra: the highest peak within 10 ppm of each channel, 0 if none."""
    rows, intensities = [], []
    for lines in mgf_blocks(path):
        pepmass, charge, rt, mz, it = parse_block(lines)
        near = np.abs(mz[:, None] - REPORTER_MZ) <= REPORTER_MZ * REPORTER_TOL
        intensities.append(np.where(near, it[:, None], 0).max(0, initial=0))
        rows.append((int(lines[0].rsplit("scan=", 1)[1]), rt, pepmass, charge or 0))
    scans, rts, pepmasses, charges = zip(*rows)
    matrix = np.array(intensities, np.float32)
    return pa.table({"filename": pa.array([os.path.basename(path)] * len(rows)).dictionary_encode(),
                     "scan": pa.array(scans, pa.int32()), "rt": pa.array(rts, pa.float32()),
                     "pepmass": pa.array(pepmasses, pa.float64()), "charge": pa.array(charges, pa.int8()),
                     **{name: pa.array(matrix[:, j]) for j, name in enumerate(REPORTER_NAMES)}})

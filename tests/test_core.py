"""Fast checks of the shared library on synthetic data; no real data needed."""
import numpy as np

from juncta import peptide, spectra, stats


def test_reference_search_is_il_blind_and_finds_the_longest_ends(tmp_path):
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">p desc\nACDEFG\nHIK\n")
    assert list(peptide.read_fasta(str(fasta))) == [("p desc", "ACDEFGHIK")]
    assert peptide.found_in(["DEFGHLK", "MMMM"], [str(fasta)]) == {"DEFGHLK"}
    assert peptide.ends("ACDEWWHIK", peptide.reference_pieces(["ACDEWWHIK"], str(fasta))) == (4, 3)


def test_rule_counts_only_sites_beyond_the_reference_ends():
    b, y, mass = peptide.ions("AAAWWWKK", 0)
    hit_b, hit_y = peptide.matched_ions("AAAWWWKK", mass / 2 + peptide.PROTON, 2, np.array([b[1], b[3], b[4], y[1]]))
    assert peptide.specific_sites("AAAWWWKK", hit_b, hit_y, 3, 3) == {4, 5}


def test_mgf_reader_and_reporters(tmp_path):
    path = tmp_path / "r.mgf"
    path.write_text("BEGIN IONS\nTITLE=x scan=7\nRTINSECONDS=1.5\nPEPMASS=500.25 3\nCHARGE=2+\n126.127726 5\n"
                    "126.127730 8\n127.2 3\nEND IONS\n")
    (_, pepmass, charge, rt, mz, _), = spectra.read_mgf_gz(str(path), {"x scan=7"})
    assert (pepmass, charge, rt, len(mz)) == (500.25, 2, 1.5, 3)
    row = spectra.reporter_table(str(path)).to_pandas().iloc[0]
    assert row.scan == 7 and row.c126 == 8 and row.c127N == 0


def test_reduce_peaks_drops_reporters_and_keeps_the_strongest():
    mz, it = spectra.reduce_peaks([100.0, 126.1, 200.0, 300.0], [5.0, 99.0, 1.0, 3.0], npeaks=2)
    assert mz.tolist() == [100.0, 300.0] and it.tolist() == [5.0, 3.0]


def test_bh_decisions_and_survival():
    p = np.array([0.001, 0.2, 0.012, 0.04, 0.9])
    assert stats.bh(p, 0.05).tolist() == (stats.bh_adjust(p) <= 0.05).tolist()
    model = stats.fit(np.random.default_rng(1).exponential(size=5000), 200)
    assert (np.diff(stats.surv([0.5, 2.0, 9.0, 20.0], model)) < 0).all()

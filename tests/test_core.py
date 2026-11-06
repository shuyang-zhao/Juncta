"""Fast checks of the shared library on synthetic data; no real data needed."""
import numpy as np
import pytest

from juncta import peptide, spectra, stats


def test_reference_search_is_il_blind_and_finds_the_longest_ends(tmp_path):
    pytest.importorskip("ahocorasick")
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">p desc\nACDEFG\nHIK\n")
    assert list(peptide.read_fasta(str(fasta))) == [("p desc", "ACDEFGHIK")]
    assert peptide.found_in(["DEFGHLK", "MMMM"], [str(fasta)]) == {"DEFGHLK"}
    assert peptide.ends("ACDEWWHIK", peptide.reference_pieces(["ACDEWWHIK"], str(fasta))) == (4, 3)


def test_il_equivalent_queries_are_all_found_in_any_order(tmp_path):
    pytest.importorskip("ahocorasick")
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">p\nACDEFGHLK\n")
    both = {"ACDEFGHIK", "ACDEFGHLK"}
    assert peptide.found_in(["ACDEFGHIK", "ACDEFGHLK"], [str(fasta)]) == both
    assert peptide.found_in(["ACDEFGHLK", "ACDEFGHIK"], [str(fasta)]) == both
    assert peptide.found_in([], [str(fasta)]) == set()


def test_rule_counts_only_sites_beyond_the_reference_ends():
    b, y, _ = peptide.ions("AAAWWWKK")
    hit_b, hit_y = peptide.matched_ions("AAAWWWKK", 2, np.array([b[1], b[3], b[4], y[1]]))
    assert peptide.specific_sites("AAAWWWKK", hit_b, hit_y, 3, 3) == {4, 5}


def test_oxidation_stays_on_the_modified_methionine():
    modified = "[+229.16293]-AMAAAM[+15.994915]AK[+229.16293]"
    assert peptide.parse_modified(modified) == ("AMAAAMAK", [5])
    A, M = peptide.AA["A"], peptide.AA["M"]
    b4, b5 = peptide.LAB + peptide.PROTON + M + np.array([3, 4]) * A   # free of the oxidation, which sits on residue 6
    assert peptide.ions("AMAAAMAK", [5])[0][3:5] == pytest.approx([b4, b5])
    assert peptide.ions("AMAAAMAK", [1])[0][3:5] == pytest.approx([b4 + peptide.OX, b5 + peptide.OX])
    assert peptide.ions("AMAAAMAK", [5])[2] == pytest.approx(peptide.ions("AMAAAMAK", [1])[2])   # same precursor
    hit_b, hit_y = peptide.matched_ions(modified, 2, np.array([b4, b5]))
    assert peptide.specific_sites("AMAAAMAK", hit_b, hit_y, 3, 3) == {4, 5}


def test_entrapments_digest_like_their_peptides():
    free_prolines = ["ACPDKEFGKHLPMKNQTR", "PAGKPVTKHPAKLPSR", "GPEPKDPKSPTKAPWK"]
    peptides = [*free_prolines, "ACDKPEFRPGHKPLMNRPQSTKPVWYR", "GGPSTVWK", "AAAKPAAKAAAKAAAR", "EREEEEEK"]
    pairs = peptide.shuffle_entrapments(peptides)
    assert {p for _, p in pairs} == set(peptides[:-2])   # the last two leave no other sequence with these cleavages
    for e, p in pairs:
        assert e != p and sorted(e) == sorted(p) and e[-1] == p[-1]
        assert peptide.cleavage_sites(e) == peptide.cleavage_sites(p)


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


def test_a_stratum_without_decoy_hits_has_no_zero_rate():
    rate = stats.hit_rate([False, False, False, True, True, False], [0, 0, 0, 1, 1, 1])
    assert rate[0] == pytest.approx(1 / 4) and rate[1] == pytest.approx(3 / 4)


@pytest.mark.parametrize("n", [1, 50, 51, 99])
def test_sparse_strata_stay_empirical(n):
    models, F, validation = stats.models_with_validation(np.arange(float(n)), np.zeros(n, int), 1000)
    assert F == 1 and validation.empty and models[0][2] == 0
    assert stats.surv([n + 100.0], models[0])[0] == pytest.approx(1 / (n + 1))   # the floor of n decoys, not a tail
    assert stats.surv([-1.0], models[0])[0] == 1


def test_a_stratum_too_small_to_split_is_fitted_but_not_validated():
    models, _, validation = stats.models_with_validation(np.arange(120.0), np.zeros(120, int), 1000)
    assert models[0][2] == 50 and validation.empty

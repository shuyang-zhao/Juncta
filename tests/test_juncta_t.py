"""Fast tests of Juncta-T on synthetic data (no real data needed)."""
import numpy as np
import pandas as pd
import pytest

from juncta.peptide import ions
from juncta.stats import models_with_validation
from juncta_t import common


def test_common_helpers():
    assert common.sage_decoy("ACDEFK") == "AFEDCK" and common.sage_decoy(common.sage_decoy("ACDEFK")) == "ACDEFK"
    assert common.plex_half("plexA_f01.mgf") == common.plex_half("plexA_f24.mgf")
    mapping = common.stratum_map(np.r_[np.full(300, 2), np.full(300, 5), np.full(10, 3), np.full(10, 4)], 200)
    assert mapping[3] == 2 and mapping[4] == 5 and mapping[0] == 2 and mapping[7] == 5


def test_final_survival_rule_and_tkf():
    evaluate = pytest.importorskip("juncta_t.evaluate")
    models, F, _ = models_with_validation(np.arange(1000.0), np.zeros(1000, int), 100)
    assert evaluate.final_survival(np.array([900.0]), np.zeros(1, int), models, F)[0] == pytest.approx(101 / 1001)
    b_ions = ions("AGSTVK")[0]
    assert evaluate.rule_sites("AGSTVK", b_ions, 2, 0, 0) == 5 and evaluate.rule_sites("AGSTVK", b_ions, 2, 3, 0) == 2
    table = pd.DataFrame({"kind": ["PSEUDO", "PSEUDO", "REJ", "PSEUDO"], "true": [True, True, False, True],
                          "p": [1e-5, 1e-4, 1e-3, 1e-2], "A": [9.0, 8.0, 7.0, 6.0], "rule_ok": 1.0})
    assert evaluate.tkf_value(table, 0.3) == 3 and evaluate.tkf_value(table, 0.1) == 2


def test_models_keep_the_stored_layout_and_invariances():
    torch = pytest.importorskip("torch")
    from juncta_t import model_entry, model_spectrum
    assert sum(p.numel() for p in model_spectrum.SpectrumCandidateModel().parameters()) == 12_670_977
    assert len(model_entry.TOKEN_FEATS) == 34 and len(model_entry.SET_FEATS) == 12
    model = model_entry.EntryTransformer().eval()
    tok, gfeat = torch.randn(1, 37, 34), torch.randn(1, 12)
    with torch.no_grad():
        logit = model(tok, torch.ones(1, 37, dtype=torch.bool), gfeat)
        permuted = model(tok[:, torch.randperm(37)], torch.ones(1, 37, dtype=torch.bool), gfeat)
    assert float((permuted - logit).abs()) < 1e-5


def test_b2_fits_a_shard_smaller_than_its_training_sample():
    pytest.importorskip("mokapot")
    rescore_b2 = pytest.importorskip("juncta_t.rescore_b2")
    rng = np.random.default_rng(0)
    n = 4000
    decoy = rng.random((n, 1)) < 0.5
    cfeat = rng.normal(size=(n, 1, 20))
    cfeat[:, :, 0] += np.where(decoy, 0.0, 3.0)
    cfeat[:, :, 19] = 1   # every candidate is rank 1
    rows, slots = np.arange(n), np.zeros(n, int)
    peptides = np.array([f"PEP{i}" for i in range(n)], dtype=object)
    extra = rng.normal(size=(n, rescore_b2.N_MS2PIP + rescore_b2.N_DEEPLC)).astype(np.float32)
    scores = rescore_b2.fit_and_score(extra, cfeat, decoy, rows, slots, peptides, n, 1)
    assert np.isfinite(scores).all() and scores[~decoy].mean() > scores[decoy].mean()

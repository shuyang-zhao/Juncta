"""Fast tests of Juncta-T on synthetic data (no real data needed)."""
import numpy as np
import pandas as pd
import pytest

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
    b_ions = evaluate.ions("AGSTVK", 0)[0]
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

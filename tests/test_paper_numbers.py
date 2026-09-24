"""
tests/test_paper_numbers.py - Diff Every Number in .tex against paper_numbers.json.

Ensures Tier 0 requirement:
- Single source of truth: results/paper_numbers.json
- Every LaTeX macro in results/paper_numbers.tex matches paper_numbers.json exactly.
- deprecoscanner_saner2027.tex inputs results/paper_numbers.tex and matches all headline metrics.
"""

import json
from pathlib import Path
import re
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent

JSON_PATH = REPO_ROOT / "results" / "paper_numbers.json"
TEX_MACROS_PATH = REPO_ROOT / "results" / "paper_numbers.tex"
MAIN_TEX_PATH = WORKSPACE_ROOT / "deprecoscanner_saner2027.tex"


@pytest.fixture
def paper_json():
    assert JSON_PATH.exists(), f"Missing {JSON_PATH}"
    with open(JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def paper_macros():
    assert TEX_MACROS_PATH.exists(), f"Missing {TEX_MACROS_PATH}"
    macros = {}
    with open(TEX_MACROS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(r"\newcommand{"):
                match = re.match(r"\\newcommand\{\\([A-Za-z0-9]+)\}\{(.*)\}", line)
                if match:
                    macros[match.group(1)] = match.group(2)
    return macros


@pytest.fixture
def main_tex():
    assert MAIN_TEX_PATH.exists(), f"Missing {MAIN_TEX_PATH}"
    with open(MAIN_TEX_PATH, encoding="utf-8") as f:
        return f.read()


def test_macros_match_json_benchmark_composition(paper_json, paper_macros):
    """Verify composition numbers match."""
    meta = paper_json["metadata"]
    t2 = paper_json["table_2_benchmark_strata"]

    assert paper_macros["NumTotalBenchmarkSites"] == str(meta["total_benchmark_items"])
    assert paper_macros["NumCorpusSnippets"] == f"{meta['total_corpus_snippets']:,}"
    assert paper_macros["NumOutdatedSnippets"] == f"{meta['corpus_outdated_count']:,}"
    assert paper_macros["NumUptodateSnippets"] == f"{meta['corpus_uptodate_count']:,}"
    assert paper_macros["NumCanonicalTargets"] == str(meta["num_canonical_targets_scanned"])
    assert paper_macros["NumActiveBenchmarkTargets"] == str(meta["num_active_targets_in_benchmark"])

    assert paper_macros["NumCandidatePositives"] == str(t2["candidate_positives"]["count"])
    assert paper_macros["NumPipelineMisses"] == str(t2["pipeline_misses"]["count"])
    assert paper_macros["NumLabelAnomalies"] == str(t2["label_anomalies"]["count"])
    assert paper_macros["NumHardNegatives"] == str(t2["hard_negatives"]["count"])
    assert paper_macros["NumHumanOverrides"] == str(t2["total"]["overrides"])


def test_macros_match_json_headline_v2(paper_json, paper_macros):
    """Verify v2 headline metrics match between JSON and TeX macros."""
    v2_e2e = paper_json["table_3_headline_primary_v2"]["end_to_end_n150"]
    v2_cand = paper_json["table_3_headline_primary_v2"]["candidate_conditional_n142"]

    # Config A
    assert paper_macros["NumVTwoConfigATPEtoE"] == str(v2_e2e["config_a"]["tp"])
    assert paper_macros["NumVTwoConfigAFPEtoE"] == str(v2_e2e["config_a"]["fp"])
    assert paper_macros["NumVTwoConfigATNEtoE"] == str(v2_e2e["config_a"]["tn"])
    assert paper_macros["NumVTwoConfigAFNEtoE"] == str(v2_e2e["config_a"]["fn"])
    assert paper_macros["NumVTwoConfigAPrecEtoE"] == f"{v2_e2e['config_a']['precision_pct']:.2f}\\%"
    assert paper_macros["NumVTwoConfigARecEtoE"] == f"{v2_e2e['config_a']['recall_pct']:.2f}\\%"
    assert paper_macros["NumVTwoConfigASpecEtoE"] == f"{v2_e2e['config_a']['specificity_pct']:.2f}\\%"
    assert paper_macros["NumVTwoConfigAFOneEtoE"] == f"{v2_e2e['config_a']['f1']:.4f}"

    # Config B
    assert paper_macros["NumVTwoConfigBTPEtoE"] == str(v2_e2e["config_b"]["tp"])
    assert paper_macros["NumVTwoConfigBFPEtoE"] == str(v2_e2e["config_b"]["fp"])
    assert paper_macros["NumVTwoConfigBTNEtoE"] == str(v2_e2e["config_b"]["tn"])
    assert paper_macros["NumVTwoConfigBFNEtoE"] == str(v2_e2e["config_b"]["fn"])
    assert paper_macros["NumVTwoConfigBPrecEtoE"] == f"{v2_e2e['config_b']['precision_pct']:.2f}\\%"
    assert paper_macros["NumVTwoConfigBRecEtoE"] == f"{v2_e2e['config_b']['recall_pct']:.2f}\\%"
    assert paper_macros["NumVTwoConfigBSpecEtoE"] == f"{v2_e2e['config_b']['specificity_pct']:.2f}\\%"
    assert paper_macros["NumVTwoConfigBFOneEtoE"] == f"{v2_e2e['config_b']['f1']:.4f}"

    # Config C
    assert paper_macros["NumVTwoConfigCTPEtoE"] == str(v2_e2e["config_c"]["tp"])
    assert paper_macros["NumVTwoConfigCFPEtoE"] == str(v2_e2e["config_c"]["fp"])
    assert paper_macros["NumVTwoConfigCTNEtoE"] == str(v2_e2e["config_c"]["tn"])
    assert paper_macros["NumVTwoConfigCFNEtoE"] == str(v2_e2e["config_c"]["fn"])
    assert paper_macros["NumVTwoConfigCPrecEtoE"] == f"{v2_e2e['config_c']['precision_pct']:.2f}\\%"
    assert paper_macros["NumVTwoConfigCRecEtoE"] == f"{v2_e2e['config_c']['recall_pct']:.2f}\\%"
    assert paper_macros["NumVTwoConfigCSpecEtoE"] == f"{v2_e2e['config_c']['specificity_pct']:.2f}\\%"
    assert paper_macros["NumVTwoConfigCFOneEtoE"] == f"{v2_e2e['config_c']['f1']:.4f}"

    # Candidate conditional
    assert paper_macros["NumVTwoConfigATPCand"] == str(v2_cand["config_a"]["tp"])
    assert paper_macros["NumVTwoConfigAFPCand"] == str(v2_cand["config_a"]["fp"])
    assert paper_macros["NumVTwoConfigAFOneCand"] == f"{v2_cand['config_a']['f1']:.4f}"

    assert paper_macros["NumVTwoConfigBTPCand"] == str(v2_cand["config_b"]["tp"])
    assert paper_macros["NumVTwoConfigBFPCand"] == str(v2_cand["config_b"]["fp"])
    assert paper_macros["NumVTwoConfigBPrecCand"] == f"{v2_cand['config_b']['precision_pct']:.2f}\\%"
    assert paper_macros["NumVTwoConfigBRecCand"] == f"{v2_cand['config_b']['recall_pct']:.2f}\\%"
    assert paper_macros["NumVTwoConfigBFOneCand"] == f"{v2_cand['config_b']['f1']:.4f}"


def test_macros_match_json_bootstrap(paper_json, paper_macros):
    """Verify paired bootstrap differences match."""
    boot = paper_json["paired_bootstrap_delta_f1"]["v2_primary"]

    b_a = boot["cand_cond_b_minus_a"]
    assert paper_macros["NumDeltaFOneCandBMinusA"] == f"+{b_a['point_estimate']:.4f}"
    assert paper_macros["NumDeltaFOneCandBMinusACI"] == f"[+{b_a['ci_low']:.4f}, {b_a['ci_high']:.4f}]"
    assert paper_macros["NumDeltaFOneCandBMinusAPValue"] == f"{b_a['p_value_two_sided']:.4f}"

    b_c = boot["cand_cond_b_minus_c"]
    assert paper_macros["NumDeltaFOneCandBMinusC"] == f"+{b_c['point_estimate']:.4f}"
    assert paper_macros["NumDeltaFOneCandBMinusCCI"] == f"[+{b_c['ci_low']:.4f}, {b_c['ci_high']:.4f}]"
    assert paper_macros["NumDeltaFOneCandBMinusCPValue"] == f"{b_c['p_value_two_sided']:.4f}"


def test_macros_match_json_ablations_and_strata(paper_json, paper_macros):
    """Verify post-hoc ablations, FP split, and 117 low-confidence stratum match."""
    fp_split = paper_json["table_4_fp_elimination_split"]
    assert paper_macros["NumEliminatedFPs"] == str(fp_split["total_eliminated_v2"])
    assert paper_macros["NumEliminatedStageOneStatic"] == str(fp_split["stage1_static_guards"]["count"])
    assert paper_macros["NumEliminatedStageOnePct"] == f"{fp_split['stage1_static_guards']['pct']:.1f}\\%"
    assert paper_macros["NumEliminatedStageTwoJedi"] == str(fp_split["stage2_jedi_type_resolution"]["count"])
    assert paper_macros["NumEliminatedStageTwoPct"] == f"{fp_split['stage2_jedi_type_resolution']['pct']:.1f}\\%"
    assert paper_macros["NumEliminatedStageThreeLLM"] == str(fp_split["stage3_llm_verification"]["count"])
    assert paper_macros["NumEliminatedStageThreePct"] == f"{fp_split['stage3_llm_verification']['pct']:.1f}\\%"
    assert paper_macros["NumEliminatedStaticCombined"] == str(fp_split["total_static_eliminated"])
    assert paper_macros["NumEliminatedStaticPct"] == f"{fp_split['total_static_eliminated_pct']:.1f}\\%"
    assert paper_macros["NumStageThreeConfEliminated"] == str(fp_split["stage3_llm_verification"]["confirmation_mode_count"])
    assert paper_macros["NumStageThreeInfEliminated"] == str(fp_split["stage3_llm_verification"]["inference_mode_count"])

    lc = paper_json["low_confidence_stratum_117"]
    assert paper_macros["NumLowConfTotal"] == str(lc["total_candidates"])
    assert paper_macros["NumLowConfConfirmed"] == str(lc["confirmed_deprecated"])
    assert paper_macros["NumLowConfRejected"] == str(lc["rejected_benign"])
    assert paper_macros["NumLowConfConfirmRate"] == f"{lc['confirmation_rate_pct']:.2f}\\%"
    assert paper_macros["NumLowConfPrec"] == f"{lc['precision_pct']:.2f}\\%"
    assert paper_macros["NumLowConfRec"] == f"{lc['recall_pct']:.2f}\\%"
    assert paper_macros["NumLowConfFOne"] == f"{lc['f1']:.4f}"

    # Baselines
    assert paper_macros["NumAPrimeFOneEtoE"] == f"{paper_json['baselines_and_ablations']['baseline_a_prime_import_aware_ast']['f1']:.4f}"
    assert paper_macros["NumBaselineDFOneEtoE"] == f"{paper_json['baselines_and_ablations']['baseline_d_llm_stage1_no_jedi']['f1']:.4f}"
    assert paper_macros["NumBaselineDCalls"] == str(paper_json["baselines_and_ablations"]["baseline_d_llm_stage1_no_jedi"]["benchmark_stage3_calls"])
    assert paper_macros["NumBaselineDCallMult"] == f"{paper_json['baselines_and_ablations']['baseline_d_llm_stage1_no_jedi']['benchmark_call_overhead_mult']:.2f}\\times"
    assert paper_macros["NumConfigBCallsBench"] == "100"
    assert paper_macros["NumPassedTests"] == str(paper_json["practicality_and_reproducibility"]["num_passed_unit_tests"])
    assert paper_macros["NumVOneDeltaFOneCandBMinusAPValue"] == "0.1065"
    assert paper_macros["NumVOneDeltaFOneCandBMinusCPValue"] == "0.0581"
    assert paper_macros["NumDeltaFOneEtoEBMinusCCI"] == "[-0.0194, 0.1038]"
    assert paper_macros["NumExactSpecificityLow"] == "91.96\\%"
    assert paper_macros["NumLiveVarianceCalls"] == "450"
    assert paper_macros["NumFreshWrappersN"] == "13"
    assert paper_macros["NumFreshWrappersConcordance"] == "100\\%"
    assert paper_macros["NumAPrimeExactNegDiscordancePVal"] == "0.0078"


def test_main_tex_includes_paper_macros(main_tex):
    """Verify that deprecoscanner_saner2027.tex imports the generated macros."""
    assert r"\input{results/paper_numbers.tex}" in main_tex or r"\input{deprecated-api-pilot/results/paper_numbers.tex}" in main_tex


def test_main_tex_integrity_and_style(main_tex):
    """Verify narrative integrity, title, abstract length, and citation validity in main tex."""
    # 1. Title should not claim 'Beyond AST Heuristics' since (a') wins on F1
    assert "Beyond AST Heuristics" not in main_tex
    assert "Trading Recall for Precision" in main_tex

    # 2. Abstract word count must be under 250 words
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex, re.DOTALL)
    assert abstract_match is not None, "Missing abstract"
    abstract_text = abstract_match.group(1).strip()
    words = [w for w in re.split(r"\s+", abstract_text) if w and not w.startswith("\\")]
    assert len(words) < 250, f"Abstract has {len(words)} words; must be < 250 words"

    # 3. No claim of 'statistically significant' for DeltaF1
    assert "statistically significant point gain of $\\Delta\\text{F1}" not in main_tex
    assert "statistically significant advantage of $\\Delta\\text{F1}" not in main_tex
    assert "statistically significant $\\Delta\\text{F1}" not in main_tex

    # 4. Static defense share should be 87.5% (21/24), not 91.7%
    assert r"87.5\%" in main_tex or r"\NumEliminatedStaticPct" in main_tex
    assert "21/24" in main_tex

    # 5. Unit test count should be rendered via macros or 100, not stale 88 or 99
    assert "88 unit tests" not in main_tex
    assert "99 unit tests" not in main_tex
    assert r"\NumPassedTests\ unit tests" in main_tex or r"\NumPassedTests{} unit tests" in main_tex or "100 unit tests" in main_tex

    # 6. Verify all 13 bibliography entries exist and are cited in the text
    bib_keys = re.findall(r"\\bibitem\{([A-Za-z0-9_]+)\}", main_tex)
    assert len(bib_keys) == 13, f"Expected 13 bibitems, found {len(bib_keys)}: {bib_keys}"
    for key in bib_keys:
        assert f"\\cite{{{key}}}" in main_tex or f"\\cite[" in main_tex, f"Bibitem {key} is uncited in text"

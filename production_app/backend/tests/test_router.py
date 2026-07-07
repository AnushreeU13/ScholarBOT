from app import router


def test_keyword_fast_path_tb():
    decision = router.route("What is the treatment for tuberculosis?")
    assert decision["abstain"] is False
    assert decision["domain"] == "TB"
    assert "guidelines_kb" in decision["target_kbs"]


def test_keyword_fast_path_drug():
    decision = router.route("Please describe the maximum daily dosage of isoniazid.")
    assert decision["abstain"] is False
    assert "druglabels_kb" in decision["target_kbs"]
    assert decision["intent"] == "drug_info"


def test_force_user_kb_short_circuits():
    decision = router.route("summarize it", force_user_kb=True)
    assert decision["target_kbs"] == ["user_kb"]
    assert decision["abstain"] is False


def test_out_of_domain_abstains_when_llm_also_abstains(monkeypatch):
    monkeypatch.setattr(router.llm, "complete", lambda *a, **k: "")
    decision = router.route("What's the weather like today?")
    assert decision["abstain"] is True
    assert decision["domain"] == "out_of_domain"


def test_word_boundary_avoids_false_positive_on_cap_substring():
    decision = router._keyword_fallback("What is the capital of France?", has_user_doc=False)
    assert decision["abstain"] is True


def test_llm_rescues_vague_query(monkeypatch):
    import json

    monkeypatch.setattr(
        router.llm, "complete",
        lambda *a, **k: json.dumps({
            "domain": "TB", "intent": "general",
            "target_kbs": ["guidelines_kb"], "abstain": False, "reason": "vignette",
        }),
    )
    decision = router.route("A 45 year old with a chronic cough and night sweats for 3 weeks")
    assert decision["abstain"] is False
    assert decision["domain"] == "TB"


def test_summarize_with_user_doc_targets_user_kb():
    decision = router._keyword_fallback("please give me a summary about tuberculosis", has_user_doc=True)
    assert decision["target_kbs"] == ["user_kb"]
    assert decision["intent"] == "summarize"


def test_summarize_uploaded_doc_without_clinical_keywords_does_not_abstain():
    """
    Regression test for a real production bug: "Summarize the document I just
    uploaded" has no TB/pneumonia/drug keyword, so it fell through the domain
    gate and incorrectly abstained with "No domain match." even though the
    user has an uploaded document and clearly wants it summarized.
    """
    decision = router.route("Summarize the document I just uploaded.", has_user_doc=True)
    assert decision["abstain"] is False
    assert decision["intent"] == "summarize"
    assert decision["target_kbs"] == ["user_kb"]


def test_summarize_without_uploaded_doc_still_requires_a_domain_keyword():
    # No has_user_doc, no clinical keyword — should still correctly abstain.
    decision = router.route("Summarize the document I just uploaded.", has_user_doc=False)
    assert decision["abstain"] is True

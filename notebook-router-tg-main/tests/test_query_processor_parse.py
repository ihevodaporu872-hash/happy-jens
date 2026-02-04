from query_processor import QueryProcessor


def test_parse_response_includes_action_and_target_store():
    qp = QueryProcessor(api_key="test-key")
    response_text = '''{
        "query_type": "single",
        "user_intent": "получить сроки",
        "optimized_prompt": "Какие сроки?",
        "include_sources": false,
        "target_store": "Дубровка",
        "compare_stores": null,
        "compare_topic": null,
        "action": "none",
        "action_args": {},
        "confidence": 0.8,
        "complexity": "simple"
    }'''

    result = qp._parse_response(response_text, "orig")
    assert result.target_store == "Дубровка"
    assert result.action == "none"
    assert result.confidence == 0.8

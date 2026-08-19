from model.preprocess import normalize, to_bow


def test_normalize_replaces_urls_mentions_and_emoji() -> None:
    assert normalize("Hi @alice https://example.com 😍") == (
        "Hi [user] [url] [smiling_face_with_heart-eyes]"
    )


def test_url_is_replaced_before_mentions() -> None:
    assert normalize("https://example.com/@alice") == "[url]"


def test_normalize_nfkc_and_whitespace() -> None:
    assert normalize("  ＷＯＷ\n\t!!!  ") == "WOW !!!"


def test_normalize_preserves_case_and_punctuation() -> None:
    assert normalize("WHAT!!!") == "WHAT!!!"


def test_to_bow_lowercases_and_keeps_letters_only() -> None:
    assert to_bow("WHAT!!! [user] [url] 123") == "what user url"


def test_empty_and_whitespace_inputs() -> None:
    assert normalize("") == ""
    assert normalize(" \t\n") == ""
    assert to_bow(" \t\n") == ""


def test_cleaners_are_idempotent() -> None:
    text = "  @alice WOW!!! 😍 https://example.com  "
    assert normalize(normalize(text)) == normalize(text)
    assert to_bow(to_bow(text)) == to_bow(text)


def test_null_bytes_and_non_english_text_do_not_crash() -> None:
    assert normalize("café\x00 привет") == "café\x00 привет"
    assert to_bow("café\x00 привет") == "caf"


def test_normalize_expands_escaped_unicode() -> None:
    assert normalize(r"today\u002c not perfect") == "today, not perfect"
    assert normalize(r"don\u2019t stop") == "don’t stop"


def test_normalize_expands_html_entities() -> None:
    assert normalize("Fish &amp; chips") == "Fish & chips"
    assert normalize("a &lt; b &gt; c") == "a < b > c"


def test_to_bow_drops_artifacts_instead_of_creating_junk_tokens() -> None:
    assert to_bow(r"today\u002c not perfect") == "today not perfect"
    assert to_bow("Fish &amp; chips") == "fish chips"


def test_normalize_is_idempotent_on_double_encoded_entities() -> None:
    text = "M&amp;amp;S"
    assert normalize(normalize(text)) == normalize(text)

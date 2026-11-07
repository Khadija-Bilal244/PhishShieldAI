import numpy as np

from phishshield.features import char_tokenizer
from phishshield.features.lexical import FEATURE_NAMES, extract, to_matrix


def test_feature_vector_has_fixed_order_and_length():
    f = extract("https://www.google.com/search?q=python")
    assert list(f) == FEATURE_NAMES
    assert all(isinstance(v, float) for v in f.values())


def test_ip_and_at_detection():
    assert extract("http://192.168.1.5/paypal/login.php")["has_ip"] == 1.0
    assert extract("http://www.paypal.com@evil.tk/login")["has_at"] == 1.0
    assert extract("https://github.com/microsoft/vscode")["has_ip"] == 0.0


def test_suspicious_tld_shortener_punycode():
    assert extract("http://paypal-login.xyz/verify")["tld_suspicious"] == 1.0
    assert extract("https://bit.ly/3abcd")["is_shortener"] == 1.0
    assert extract("https://xn--pypal-4ve.com/login")["has_punycode"] == 1.0


def test_brand_mismatch():
    assert extract("http://paypal-secure-login.xyz")["brand_mismatch"] == 1.0
    assert extract("https://paypal.com/signin")["brand_mismatch"] == 0.0


def test_handles_garbage_without_crashing():
    for u in ["", "not a url", "http://[::1", "ftp://x", "a" * 3000]:
        assert len(extract(u)) == len(FEATURE_NAMES)


def test_matrix_shape():
    m = to_matrix(["http://a.com", "http://b.com/x"])
    assert m.shape == (2, len(FEATURE_NAMES)) and m.dtype == np.float32


def test_char_tokenizer_pads_and_truncates():
    ids = char_tokenizer.encode("ABC", 5)
    assert ids.tolist()[:3] != [0, 0, 0] and ids.tolist()[3:] == [0, 0]
    assert char_tokenizer.encode("x" * 50, 10).shape == (10,)
    assert char_tokenizer.encode("é", 3)[0] == char_tokenizer.UNK

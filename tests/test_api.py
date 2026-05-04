import socket

import pytest

from yugiquery import api


class _MockResp:
    def __init__(self, data=None, status=200):
        self._data = data or {}
        self.status_code = status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("HTTP")


def test_fetch_ygoprodeck(monkeypatch):
    monkeypatch.setattr(api.requests, "get", lambda *args, **kwargs: _MockResp({"data": [{"id": 1}]}))
    data = api.fetch_ygoprodeck()
    assert isinstance(data, list) and data[0]["id"] == 1


def test_check_status_true_and_false(monkeypatch):
    # healthy response
    monkeypatch.setattr(
        api.requests,
        "get",
        lambda *args, **kwargs: _MockResp({"query": {"general": {"generator": "ok"}}}, status=200),
    )
    assert api.check_status() is True

    # simulate request failure and unreachable socket
    def _raise(*a, **k):
        raise api.requests.exceptions.RequestException("fail")

    monkeypatch.setattr(api.requests, "get", _raise)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    assert api.check_status() is False


def test_fetch_redirects(monkeypatch):
    def _get(*args, **kwargs):
        return _MockResp({"query": {"redirects": [{"from": "A", "to": "B"}]}})

    monkeypatch.setattr(api.requests, "get", _get)
    res = api.fetch_redirects("A")
    assert res.get("A") == "B"

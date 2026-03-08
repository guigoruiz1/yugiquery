import pytest

from yugiquery.bot.base import Bot
from yugiquery.utils.dirs import dirs


def test_abort_branches(monkeypatch):
    # avoid filesystem scanning in init
    monkeypatch.setattr(Bot, "init_reports_enum", lambda self: None)

    b = Bot()

    # no process -> abort failed
    b.process = None
    b.repo = None
    assert b.abort() == "No query is currently running."

    # with a process object exposing terminate()
    class P:
        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True

    p = P()
    b.process = p
    b.repo = None
    assert b.abort() == "Query aborted."
    assert p.terminated is True


def test_battle_no_cards(monkeypatch, tmp_path):
    # avoid filesystem scanning in init
    monkeypatch.setattr(Bot, "init_reports_enum", lambda self: None)
    # point WORK so DATA resolves to an empty folder
    d = dirs
    monkeypatch.setattr(d, "WORK", tmp_path)

    b = Bot()

    async def cb(_):
        return None

    import asyncio

    res = asyncio.run(b.battle(cb))
    assert "error" in res

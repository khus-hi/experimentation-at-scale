from pathlib import Path

import duckdb

from ingest import download, load

API = download.API


def _file(fid, name, path, sha="x"):
    return {
        "id": fid,
        "attributes": {
            "kind": "file",
            "name": name,
            "materialized_path": path,
            "size": 10,
            "date_modified": "2020-03-12T00:00:00",
            "extra": {"hashes": {"sha256": sha, "md5": "m"}},
        },
        "links": {"download": f"https://osf.io/download/{fid}/"},
    }


def _folder(name, href):
    return {
        "id": name,
        "attributes": {"kind": "folder", "name": name},
        "relationships": {"files": {"links": {"related": {"href": href}}}},
    }


PAGES = {
    f"{API}/nodes/jd64p/files/osfstorage/": {
        "data": [_folder("data", "F1"), _file("r", "README.md", "/README.md")],
        "links": {"next": "P2"},
    },
    "P2": {"data": [_file("n", "notes.txt", "/notes.txt")], "links": {"next": None}},
    "F1": {
        "data": [
            _file("e", "upworthy-archive-exploratory-packages.csv", "/data/e.csv"),
            _file("c", "upworthy-archive-confirmatory-packages.csv", "/data/c.csv"),
        ],
        "links": {},
    },
    f"{API}/nodes/jd64p/children/": {"data": [], "links": {}},
}


def test_walk_follows_folders_and_pages():
    files = list(download.walk_node(PAGES.__getitem__, "jd64p"))
    cats = {f.name: f.category for f in files}
    assert cats == {
        "README.md": "doc",
        "notes.txt": None,
        "upworthy-archive-exploratory-packages.csv": "exploratory",
        "upworthy-archive-confirmatory-packages.csv": "confirmatory",
    }


def test_unexpected_layout_is_reported():
    try:
        list(download.walk_node(lambda url: {"errors": ["nope"]}, "jd64p"))
    except download.UnexpectedLayout as exc:
        assert exc.payload == {"errors": ["nope"]}
    else:
        raise AssertionError("expected UnexpectedLayout")


def test_load_tags_parts(tmp_path: Path, capsys):
    header = "clickability_test_id,created_at,headline,clicks,problem\n"
    (tmp_path / "x-exploratory-packages.csv").write_text(
        header + "a,2013-01-01 00:00:00,h1,1,false\na,2013-01-01 00:00:00,h2,2,false\n"
    )
    (tmp_path / "x-confirmatory-packages.csv").write_text(
        header + "b,2014-06-01 00:00:00,h3,3,true\n"
    )
    db = tmp_path / "t.duckdb"
    assert load.main(["--db", str(db), "--raw", str(tmp_path)]) == 0
    with duckdb.connect(str(db)) as con:
        rows = con.execute(
            "SELECT source_part, COUNT(*) FROM raw.packages GROUP BY 1 ORDER BY 1"
        ).fetchall()
    assert rows == [("confirmatory", 1), ("exploratory", 2)]
    out = capsys.readouterr().out
    assert "flagged total: 1 tests" in out


def test_readmes_in_different_folders_do_not_collide():
    a = download.OsfFile(
        "a", "README.md", "/google-analytics-data/README.md", "jd64p", 1, None, "u", None, None
    )
    b = download.OsfFile(
        "b",
        "README.md",
        "/upworthy-timestamped-screenshots/README.md",
        "jd64p",
        1,
        None,
        "u",
        None,
        None,
    )
    c = download.OsfFile(
        "c",
        "x-holdout-packages.csv",
        "/upworthy-archive-datasets/x-holdout-packages.csv",
        "jd64p",
        1,
        None,
        "u",
        None,
        None,
    )
    assert download.local_name(a) == "google-analytics-data__README.md"
    assert download.local_name(b) == "upworthy-timestamped-screenshots__README.md"
    assert download.local_name(c) == "x-holdout-packages.csv"

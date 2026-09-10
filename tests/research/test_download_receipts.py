"""HTTP test doubles; these tests never contact Broad's servers."""

import io
import zipfile

import pytest

from counterfactual_microscopy.research.downloads import Download, fetch, plan


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}


def test_counts_url_is_csv_not_its_misleading_link_label():
    item = plan("bbbc006", [16])[-1]
    assert item.name == "BBBC006_v1_counts.csv"
    assert item.url.endswith("/BBBC006/BBBC006_v1_counts.csv")


def test_csv_receipt_roundtrip_and_budget(tmp_path, monkeypatch):
    data = b"Image_Count_Nuclei,Image_Metadata_Site,Image_Metadata_Well\n6,1,a01\n"
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(data))
    item = plan("bbbc006", [16])[-1]
    result = fetch(item, tmp_path, 1000)
    assert result.read_bytes() == data
    assert fetch(item, tmp_path, 1000) == result
    with pytest.raises(ValueError, match="budget"):
        fetch(item, tmp_path, 1)
    result.write_bytes(b"changed")
    with pytest.raises(ValueError, match="receipt"):
        fetch(item, tmp_path, 1000)


def test_counts_html_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(b"<html>error</html>"))
    with pytest.raises(ValueError, match="columns"):
        fetch(plan("bbbc006", [16])[-1], tmp_path, 1000)


def test_zip_download_resume_validated(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("tiny.txt", "fixture")
    data = buffer.getvalue()
    item = Download("fixture.zip", "https://data.broadinstitute.org/bbbc/fixture.zip", len(data))
    (tmp_path / "fixture.zip.part").write_bytes(data[:12])

    def response(request, **kwargs):
        assert request.headers["Range"] == "bytes=12-"
        return Response(
            data[12:],
            206,
            {
                "Content-Range": f"bytes 12-{len(data) - 1}/{len(data)}",
                "Content-Length": str(len(data) - 12),
            },
        )

    monkeypatch.setattr("urllib.request.urlopen", response)
    assert fetch(item, tmp_path, 1000).read_bytes() == data


def test_bad_resume_rejected(tmp_path, monkeypatch):
    item = Download("fixture.zip", "https://data.broadinstitute.org/bbbc/fixture.zip", 100)
    (tmp_path / "fixture.zip.part").write_bytes(b"partial")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: Response(b"x", 206, {"Content-Range": "bytes 0-0/1"}),
    )
    with pytest.raises(ValueError, match="offset"):
        fetch(item, tmp_path, 1000)


@pytest.mark.parametrize("name", ["../bad.zip", "/bad.zip", "bad\\name.zip"])
def test_download_filename_cannot_escape_root(tmp_path, name):
    with pytest.raises(ValueError, match="filename"):
        fetch(Download(name, "https://data.broadinstitute.org/bbbc/file.zip", 10), tmp_path, 100)

from ecg_linkage.cli import main


def test_generate_synthetic_cli(tmp_path, capsys):
    rc = main(["generate-synthetic", "--out", str(tmp_path), "--n-entities", "30", "--seed", "5"])
    assert rc == 0
    assert (tmp_path / "roster.csv").exists() and (tmp_path / "nppes_sample.csv").exists()
    assert "generated 30 entities" in capsys.readouterr().out

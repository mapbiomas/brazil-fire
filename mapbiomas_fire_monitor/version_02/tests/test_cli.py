"""Testes da CLI (M5)."""

from mapbiomas_cog.cli import build_parser, main


def test_parse_export():
    args = build_parser().parse_args(
        [
            "export",
            "--country", "ecuador",
            "--theme", "lulc",
            "--collection", "collection_03",
            "--product", "coverage",
            "--unit", "1985",
        ]
    )
    assert args.country == "ecuador"
    assert args.unit == ["1985"]
    assert args.runner == "ee_batch_tiled"
    assert args.force is False


def test_status_catalog():
    assert main(["status"]) == 0


def test_status_product_without_manifest():
    rc = main(
        [
            "status",
            "--country", "ecuador",
            "--theme", "lulc",
            "--collection", "collection_03",
            "--product", "coverage",
        ]
    )
    assert rc == 0


def test_no_command_prints_help():
    assert main([]) == 0

"""These are browser checks of a BUILT Spain map (Playwright, a local server, `data/_spain`), run by hand after a build
(`python tests/test_prefix.py data/_spain`), not unit tests: pytest does not collect them."""
collect_ignore_glob = ["test_*.py", "check_*.py", "rangeserver.py"]

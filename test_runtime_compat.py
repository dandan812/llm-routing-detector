def test_detector_and_server_import_on_supported_runtime():
    # Importing these modules catches runtime-evaluated typing constructs and
    # other compatibility regressions before the desktop launcher is used.
    import gpt56_vnext.detector  # noqa: F401
    import gpt56_vnext.server  # noqa: F401

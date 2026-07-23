from electricpy import version


def test_version_fields():
    """Validate version fields behavior."""
    assert version.NAME
    assert version.VERSION
    assert "." in version.VERSION

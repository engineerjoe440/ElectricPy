from electricpy import version


def test_version_fields():
    assert version.NAME
    assert version.VERSION
    assert "." in version.VERSION

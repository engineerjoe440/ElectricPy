from electricpy import latex


def test_clatex_rectangular_format():
    latex_str = latex.clatex(1 + 2j, polar=False)

    assert latex_str.startswith("$")
    assert latex_str.endswith("$")
    assert r"\mathrm{j}" in latex_str


def test_tflatex_basic():
    latex_str = latex.tflatex(([1, 1], [1, 2, 1]), predollar=False, postdollar=False)
    assert r"\frac" in latex_str

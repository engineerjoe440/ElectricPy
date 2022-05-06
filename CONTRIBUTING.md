# Contribution Guidelines for ElectricPy
*ElectricPy - The Electrical Engineer's Python Toolkit*

We'd *gladly* accept contributions that add functions, enhance documentation,
improve testing practices, or better round out this project in general; but to
help move things along more quickly, here are a few things to keep in mind.

### Adding New Functions

When adding additional functions to the package, we'd love to maintain
consistency wherever possible, a few of these things to keep in mind include:

**Documentation:** *(a must!)*
* Format function docstrings according to
[NumPyDoc](https://numpydoc.readthedocs.io/en/latest/format.html) standards
* Use the very first line of docstrings to give a brief (one-line) description
of what the function is used for
* Whenever possible, use LaTeX `.. math::` blocks to help document the formula(s)
used in the function; as a general rule, any time there's more to a function than
a simple addition/multiplication, it's best to show the formula.
* If diagrams or images would make the documentation more clear, their inclusion
would be *greatly* appreciated

**Code:**
* When possible, use other core functions already in ElectricPy to build upon
* When multiple voltages, currents, or other similar quantities need to be used
in the same function, their variables should be uniquely named so to help
clarify their purpose. So, instead of using `V1` and `V2`, use descriptive names
such as `Vgenerator` and `Vline`.
* Whenever possible, comments should be added in the code to help clarify what
operations are being performed


### Adding new Tests/Test Routines

When adding additional test functions and routines, they should be added using
the `pytest` framework.

* Tests should be added under the `test` directory in the repository

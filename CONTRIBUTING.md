# Contributing

Any form of contribution is welcomed, therefore, feel free to help improve this 
project.

This guide is adapted from [ctgan](https://sdv-dev.github.io/CTGAN/contributing.html)
and from [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/CONTRIBUTING.md).

## Types of Contributions

You can contribute in many ways

### Report bugs

Report bugs at the [GitHub Issues page](https://github.com/pbmartins/ctgan-tf/issues).

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix bugs

Look through the GitHub issues for bugs. Anything tagged with `bug` and `help
wanted` is open to whoever wants to implement it.

### Implement or propose features

Look through the GitHub issues for features. Anything tagged with `enhancement`
and `help wanted` is open to whoever wants to implement it.

Nonetheless, if you have any new feature that you think will improve the quality
of this project, please open a new issue labelled as an `enhancement`, and try
to explain in detail how it would work. If you think it requires significant 
changes, maybe break it down into narrower components.

### Write documentation

Our documentation could and should always look into ways of improving.
Contributions to make our docstrings and docs more clear are greatly appreciated.

## How to contribute

The preferred way to contribute is to set up for local development by forking 
the [main repository](https://github.com/pbmartins/ctgan-tf) from GitHub.

1. Fork the `ctgan-tf` repo on GitHub: click on the 'Fork' button near the top 
   of the page. This creates a copy of the code under your account on the 
   GitHub server.
   
2. Clone your fork locally:

    ```
    $ git clone git@github.com:your_username/ctgan-tf.git
    $ cd ctgan-tf
    ```
   
3. Create a branch for local development:
   
   ```
   $ git checkout -b CTGAN-TF-N_feature_description
   ```
   
   Please follow this branch naming convention, where `N` corresponds to the
   GitHub issue number, and `feature_description` to a brief feature or bug fix
   description, separated by underscores. Remember never to work directly on
   the `master` branch! Example: `CTGAN-TF-15_Write_docstrings`.


4. We strongly advise you to use a virtualenv wrapper of any king.
   For demonstration purposes, we will be using `pipenv`, by running the 
   provided Makefile script:
   
   ```
   $ make env
   ```
   
   If you want to use a different wrapper, simply do:
   
   ```
   $ mkvirtualenv ctgan
   $ source ctgan/bin/activate
   $ make install-develop
   ```

5. While hacking your changes, make sure to cover all your developments with 
   the required unit tests, and that none of the old tests fail as a 
   consequence of your changes. For this, make sure to run the tests suite and 
   check the code coverage:

   ```
   $ make lint       # Checks code styling
   $ make test       # Runs the unit tests
   $ make coverage   # Generates the coverage report
   ```

6. Make also sure to include the necessary documentation in the code as 
   docstrings following the [numpy docstrings style](https://numpydoc.readthedocs.io/en/latest/format.html).
   If you want to view how your documentation will look like when it is 
   published, you can generate the docs by running:

   ```
   $ make docs 
   ```
   
   The resulting HTML files will be placed in `docs/_build/html/` 
   and are viewable in a web browser.

7. Commit your changes and push your branch to GitHub:

   ```
   $ git add -A
   $ git commit -m "Your detailed description of your changes."
   $ git push origin CTGAN-TF-N_name_of_your_bugfix_or_feature
   ```

8. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. It resolves an open GitHub Issue and contains its reference in the title or
   the comment. If there is no associated issue, feel free to create one.
2. Whenever possible, it resolves only **one** issue. If your PR resolves more 
   than one issue, try to split it in more than one pull request.
3. The pull request should include unit tests that cover all the changed code.
4. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the documentation in an appropriate place.
5. The pull request should work for all the supported Python versions. 
   Check the [Travis Build Status page](https://travis-ci.com/github/pbmartins/ctgan-tf/pull_requests)
   and make sure that all the checks pass.

## Unit Testing Guidelines

All the Unit Tests should comply with the following requirements:

1. Unit Tests should be based only in `unittest` and `pytest` modules.

2. The tests that cover a module called `ctgan/path/to/a/module/_code.py`
   should be implemented in a sub-module called
   `/ctgan/path/to/a/module/tests/test_code.py`.
   Note that the module name has the ``test_`` prefix and is located in a path 
   similar to the one of the tested module, just inside the ``tests`` folder.

3. Each method of the tested module should have at least one associated test 
   method, and each test method should cover only **one** use case or scenario.

4. Test case methods should start with the ``test_`` prefix and have 
   descriptive names that indicate which scenario they cover.

5. Each test should validate only what the code of the method being tested does, 
   and not cover the behavior of any third party package or tool being used, 
   which is assumed to work properly as far as it is being passed the right 
   values.

## Release Workflow

The process of releasing a new version involves several steps combining both 
`git` and `bump2version` which, briefly:

1. Merge what is in `master` branch into `stable` branch.
2. Update the version in `setup.cfg`, `ctgan/__init__.py` and
   `HISTORY.md` files.
3. Create a new git tag pointing at the corresponding commit in `stable` branch.
4. Merge the new commit from `stable` into `master`.
5. Update the version in `setup.cfg` and `ctgan/__init__.py`
   to open the next development iteration.

Before starting the process, make sure that ``HISTORY.md`` has been updated 
with a new entry that explains the changes that will be included in the
new version. Normally this is just a list of the Pull Requests that have 
been merged to `master` since the last release.

Once this is done, run of the following commands:

1. If you are releasing a patch version:

   ```
   $ make release
   ```

2. If you are releasing a minor version::

   ```
   $ make release-minor 
   ```

3. If you are releasing a major version::
   
   ```
   $ make release-major 
   ```


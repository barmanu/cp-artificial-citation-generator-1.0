# cp-artificial-citation-generator-1.0

The cp-artificial-citation-generator-1.0 is a python based library.

### Pre-requisites
- Python 3.7
- Poetry

### Install dependencies
`$ poetry install`

### Install pre-commit hooks
After cloning the repository, run the following command to install pre-commit hook.
This automatically runs [black](https://black.readthedocs.io/en/stable/the_black_code_style.html) , 
[flake8](https://flake8.pycqa.org/en/stable/) and [mypy](https://mypy.readthedocs.io/en/stable/index.html)
tools, which perform code formatting. Make sure to add the formatted files to git again.

`$ poetry run pre-commit install`

### Run tests
`$ poetry run pytest`

### Versioning
The cp-artificial-citation-generator-1.0 is using the [semantic versioning scheme](https://semver.org/). The initial version is `0.1.0`. 
Every release bump the patch version manually. 

Please follow below steps to bump version: 
- Open `pyproject.toml` and update `version` value under `[tool.poetry]`, i.e major version update `version = "1.0.0"`.
- Commit `pyproject.toml` as part of your changes.

### Configure private repository (artifactory)
Configure private repository (artifactory) if you have not done already on your machine. This needs to be done only 
once, not for every project.

1. Configure private repository:

    `$ poetry config repositories.dp-caps-repository https://rt.artifactory.tio.systems/artifactory/api/pypi/pypi-dp-caps-local/simple`

2. Configure credentials to access private repository:

    `$ poetry config http-basic.dp-caps-repository <your_regn_username> <your_jfrog_api_key>`

    [how to get artifactory API key](https://www.jfrog.com/confluence/display/JFROG/User+Profile#UserProfile-APIKey)
    
##### [More poetry commands](https://python-poetry.org/docs/cli/)

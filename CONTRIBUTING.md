# Contributing

We are pleased that you would like to contribute to SteerX. We welcome both reporting issues and submitting pull requests.

## Reporting issues
Please make sure to include any potentially useful information in the issue, so we can pinpoint the issue faster without going back and forth.

- What SHA of SteerX are you running? If this is not the latest SHA on the main branch, please try if the problem persists with the latest version.
- Python versions

## Contributing a change

Contributors must _sign off_ that they adhere to requirements by adding a `Signed-off-by` line to all commit messages with an email address that matches the commit author:

```
feat: this is my commit message

Signed-off-by: Random J Developer <random@developer.example.org>
```


Coding Style Guidelines
We are using tools to enforce code style:
- iSort, to sort imports
- Black, to format code

We run a series of checks on the codebase on every commit using pre-commit. To install the hooks, run:
`pre-commit install`

To run the checks on-demand, run:
`pre-commit run --all-files`

## Contributing to documentation
`uv pip install  -e ".[docs]"`

We use [MkDocs](https://www.mkdocs.org/) to write documentation.

To run the documentation server, run:

```bash
uv run mkdocs serve
```

The server will be available at [http://localhost:8000](http://localhost:8000).

## Submitting a pull request

1. Fork and clone the repository
2. Create a new branch: `git checkout -b my-branch-name`
3. Make your change, push to your fork and submit a pull request
4. Wait for your pull request to be reviewed and merged.


## Developer info

We are pleased that you are developing new functionality for the toolkit. Before committing any code, please make sure
you run the secret scanning below.

### Secret scanning

To update the secrets database manually, run:

```commandline
detect-secrets scan --update .secrets.baseline
```

To audit detected secrets, use:

```commandline
detect-secrets audit .secrets.baseline
```

If the pre-commit hook raises an error but the audit command succeeds with just
`Nothing to audit!` then run `detect-secrets scan --update .secrets.baseline`
to perform a full scan and then repeat the `audit` command.

# CLI Reference

## `repo2readme run`

Generates a `README.md` by analyzing a repository (git URL or local path).

```bash
repo2readme run [OPTIONS]
```

### Options

| Flag | Short | Description |
|---|---|---|
| `--url <URL>` | `-u` | Git repository URL to clone and process. Any form `git clone` accepts. |
| `--local <PATH>` | `-l` | Path to a local repository. |
| `--output <FILE_PATH>` | `-o` | File path to save the generated README. Defaults to `README.md`. |
| `--force` | `-f` | Overwrite the output file and skip the token estimation confirmation prompt. |
| `--strict` | | Exit with a non-zero status if any file fails to summarize. |
| `--respect-gitignore` | | Honor `.gitignore` and `.git/info/exclude` patterns during repository traversal. This is opt-in; default behavior is unchanged. |
| `--dry-run` | | Preview the analysis (repo tree, token estimate, files to be processed) without making any API calls or requiring API keys. |
| `--include <PATTERN>` | | Glob pattern for files to include, even if normally filtered out. Can be passed multiple times. |
| `--exclude <PATTERN>` | | Glob pattern for files to exclude. Can be passed multiple times. |
| `--max-file-size-kb <N>` | | Skip files larger than N KB. |
| `--rollup-threshold <N>` | | Roll file summaries up into directory summaries above this file count. Default 15. |
| `--provider <NAME>` | | LLM provider to use. See `repo2readme providers`. |
| `--model <NAME>` | | Model name. Defaults to the selected provider's default model. |
| `--base-url <URL>` | | Base URL override for OpenAI-compatible providers. |

You must provide exactly one of `--url` or `--local`.

### `--url`

Any URL form `git clone` accepts works, not just `https://github.com/...`:

```bash
repo2readme run --url https://github.com/user/repo
repo2readme run --url https://github.com/user/repo.git
repo2readme run --url git@github.com:user/repo.git
repo2readme run --url ssh://git@github.com/user/repo.git
repo2readme run --url git://github.com/user/repo.git
repo2readme run --url https://gitlab.com/group/subgroup/repo
repo2readme run --url https://git.company.internal/team/service.git
```

Schemes are matched case-insensitively, and `git+ssh://` / `git+https://`
prefixes are accepted. A `file://` URL is treated as a local path. Anything
else (`s3://`, for example) is rejected up front with the list of supported
schemes, rather than being read as a directory name.

The repository is cloned shallowly (`--depth 1`) into a private temporary
directory that is removed when the run finishes, so two runs against the same
repository cannot interfere with each other.

### `--branch`

Branch to clone when using `--url`. Defaults to `main`.

```bash
repo2readme run --url https://github.com/user/repo --branch develop
```

### `--max-workers`

Number of worker threads used to read and process files. Defaults to 4, capped
at the number of files. This applies to both `--local` and `--url` runs.

```bash
repo2readme run --url https://github.com/user/repo --max-workers 8
```

It bounds three stages: reading files during traversal, summarizing them, and
summarizing directories. The directory roll-up used to ignore it and run one
call at a time.

### `--rollup-threshold`

Above this many files, individual file summaries are rolled up into directory
summaries before the README prompt is built, so a large repository does not
send hundreds of file summaries to the model at once. Defaults to 15.

```bash
repo2readme run --local . --rollup-threshold 40    # roll up later
repo2readme run --local . --rollup-threshold 99999 # never roll up
```

Directories at the same depth are summarized concurrently, and the results are
cached: an incremental run where nothing changed under a directory reuses its
summary instead of paying for it again. The cached entry is keyed on the
summaries the directory was built from, so changing one file invalidates that
file's directory and nothing else.

A directory whose roll-up fails is reported after the run and the file
summaries underneath it are used instead. The failure placeholder is never
handed to the README prompt.

### `--respect-gitignore`

Honor `.gitignore` and `.git/info/exclude` patterns during repository traversal. This is opt-in, so the default behavior remains unchanged. When enabled, files and directories matching gitignore rules are skipped before language detection, parsing, summarization, and token estimation.

```bash
repo2readme run --local ./repo --respect-gitignore
repo2readme run --url https://github.com/user/repo --respect-gitignore
```

### `--strict`

Summarization is best-effort: a file that fails (rate limit, timeout, bad
response) is skipped and the README is generated from the rest. Every run
prints a report when that happens:

```
Summarization report

Succeeded          : 37/40
Failed             : 3/40

3 file(s): Error code: 429 - rate limit reached for model ...
    - src/api/client.py
    - src/api/routes.py
    - src/api/schema.py

The README was generated from the files that succeeded.
```

Failed files are never passed to the README prompt. If *every* file fails the
run stops with exit code 1 instead of asking the model to write a README from
nothing.

Use `--strict` in CI to turn any failure into a non-zero exit code. The README
is still written first, so you keep the partial result:

```bash
repo2readme run --local . --output README.md --force --strict
```

### `--dry-run`

Runs local analysis only — repo tree generation, file filtering, and token estimation — with no LLM calls and no API keys required. Useful for verifying your include/exclude filters before spending tokens.

```bash
repo2readme run --local ./path/to/your/repo --dry-run
```

Example output:

```
Repository Tree

project/
├── src/
├── tests/
└── README.md

Files to be processed

✓ src/main.py
✓ src/api.py
✓ tests/test_api.py
...

Repository Analysis

Files selected     : 45
Estimated tokens   : ~120,000
Request size       : ~420.5 KB

Dry run complete.
No API requests were made.
```

## `repo2readme providers`

Prints the supported LLM providers with their aliases, default models and API
key environment variables.

```bash
repo2readme providers
```

```text
        Supported providers
Provider    Aliases  Default model            API key env var
groq        -        openai/gpt-oss-120b      GROQ_API_KEY
google      gemini   gemini-2.5-flash         GOOGLE_API_KEY
...
ollama      -        llama3                   not required
```

## `repo2readme reset`

Deletes the locally stored API key configuration file (`~/.repo2readme_env.json`).

```bash
repo2readme reset
```

You'll be prompted to re-enter your `GROQ_API_KEY` and `GOOGLE_API_KEY` on the next `run`.

## See also

- [Configuration](./configuration.md) — API keys and env vars
- [Examples](./examples.md) — real command examples

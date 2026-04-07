# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this repository.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

### 1. Code Style and Readability

- Clear, descriptive naming (variables, functions, classes, modules)
- Appropriate function/method length (single responsibility)
- Logical code organization and flow
- Avoidance of deeply nested structures
- Linting via **ruff** (`uv run --group lint ruff check .`) — lint the **entire repo**, not just the package
- Type-hint checking via **ty** (`uv run --group typecheck ty check somax`)

> **Rule of thumb**: Sacrifice *cleverness* for *clarity*. Sacrifice *brevity* for *explicitness*.
> Don't worry about formatting — our CI pipeline (ruff format, pre-commit) handles that automatically.

### 2. Modern Python Idioms (Python >= 3.12)

- `from __future__ import annotations` at the top of every module
- Type hints on **all** public functions, methods, and module-level variables
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- Context managers (`with` statements) for resource handling
- Modern union syntax (`X | Y` instead of `Union[X, Y]`)
- Modern optional syntax (`X | None` instead of `Optional[X]`)
- Built-in generics (`list[int]`, `dict[str, Any]` instead of `List[int]`, `Dict[str, Any]`)

### 3. Packaging and Project Structure

- Proper `pyproject.toml` configuration (PEP 621)
- Appropriate use of `__init__.py` exports
- Clear module boundaries and dependencies
- Correct use of relative vs absolute imports

### 4. Documentation

- Module-level docstrings explaining purpose
- Function/method docstrings for **all** public APIs (Google style)
- Inline comments explaining *why*, not *what*
- All scientific algorithms should include Unicode equations in docstrings where appropriate

### 5. Error Handling

- Specific exception types (never bare `except:`)
- Custom exceptions for domain-specific errors
- Helpful error messages with context
- Proper exception chaining (`raise ... from ...`)

### 6. Testing Considerations

- Functions should be easily testable (pure functions where possible)
- Dependencies should be injectable
- Side effects should be isolated and explicit
- Consider edge cases and boundary conditions

### 7. Performance (when relevant)

- Appropriate data structures for the use case
- Avoid premature optimization
- Note O(n) implications for critical paths

### 8. Security

- No hardcoded secrets or credentials
- Input validation and sanitization
- Safe handling of file paths

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${Summary of suggestion with necessary context}

* **Priority**: ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions...)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Level | Use when |
|-------|----------|
| **Critical** | Bugs, security issues, or code that will fail |
| **High** | Significant issues affecting maintainability or correctness |
| **Medium** | Improvements for code quality or consistency |
| **Low** | Minor polish or optional enhancements |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step

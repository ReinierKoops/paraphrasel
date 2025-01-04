# Naverdict
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/naverdict)
![PyPI](https://img.shields.io/pypi/v/naverdict)
![PyPI - License](https://img.shields.io/pypi/l/naverdict)

## Introduction

Find similar word pairs based on semantics. Makes use of sentence transformers.

## Installation

Install via pip:

```cmd
pip install paraphrasel
```

## How to use with Python

Using this as a Python package you can use the following commands:

### Single:
```python
>>> from paraphrasel.match import compare

>>> compare("study", "해요 to do", "all", 2)
```
```
0.21
```

### Multiple:
```python
>>> from paraphrasel.match import compare_multiple

>>> compare("study", "해요 to do", "all", 2)
```
```
{
  "\ud574\uc694 to do": 0.21,
  "\uc9d1 house": 0.23,
  "\ub298\ub2e4 to play": 0.28
}
```

### Above Cutoff:
```python
>>> from paraphrasel.match import get_above_cutoff

>>> compare("study", "해요 to do", "all", 2, 0.22)
```
```
{
  "\uc9d1 house": 0.23,
  "\ub298\ub2e4 to play": 0.28
}
```

### Best Match:
```python
>>> from paraphrasel.match import get_best_match

>>> compare("study", "해요 to do", "all", 2, 0.22)
```
```
{
  "\ub298\ub2e4 to play": 0.28
}
```

## How to use with CMD/CLI

Using this as a CMD/CLI you can use the following commands:

### Single:
```bash
$ paraphrasel single study "해요 to do" --language all --decimals 2
```
```
0.21
```

### Multiple:
```bash
$ paraphrasel multiple study "해요 to do" "집 house" "늘다 to play" --language all --decimals 2
```
```
{
  "\ud574\uc694 to do": 0.21,
  "\uc9d1 house": 0.23,
  "\ub298\ub2e4 to play": 0.28
}
```

### Above Cutoff:
```bash
$ paraphrasel above-cutoff study "해요 to do" "집 house" "늘다 to play" --language all --decimals 2 --cutoff 0.22
```
```
{
  "\uc9d1 house": 0.23,
  "\ub298\ub2e4 to play": 0.28
}
```

### Best Match:
```bash
$ paraphrasel best-match study "해요 to do" "집 house" "늘다 to play" --language all --decimals 2 --cutoff 0.2
```
```
{
  "\ub298\ub2e4 to play": 0.28
}
```
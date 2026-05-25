# Hackathon Setup Guide

## Repository

This is a fork of [NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo) maintained at [Nitesh3000/NeMo](https://github.com/Nitesh3000/NeMo) for hackathon collaboration.

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/Nitesh3000/NeMo.git
cd NeMo
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows:** `.\venv\Scripts\Activate.ps1`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Set your git identity (local only)

```bash
git config user.name "YourName"
git config user.email "your@email.com"
```

## Collaboration Workflow

- **Push changes** → goes to `Nitesh3000/NeMo` (our fork)
- **Pull latest** → `git pull origin main`
- **Sync with NVIDIA upstream** → `git pull upstream main`

### Remotes

| Remote   | URL                                          | Purpose              |
|----------|----------------------------------------------|----------------------|
| origin   | https://github.com/Nitesh3000/NeMo.git       | Our hackathon fork   |
| upstream | https://github.com/NVIDIA-NeMo/NeMo.git      | NVIDIA original repo |

## Branch & PR Flow

1. Create a branch for your feature/fix: `git checkout -b your-branch-name`
2. Make changes, commit, and push: `git push origin your-branch-name`
3. Open a Pull Request on GitHub against `Nitesh3000/NeMo`
4. Get it reviewed and merged

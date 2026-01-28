# FFN-Checker Installation Guide

Complete step-by-step installation instructions for FFN-Checker.

## Prerequisites

Before starting, ensure you have:
- Python 3.10 or newer
- 16GB RAM minimum (32GB recommended)
- 50GB free disk space
- Internet connection (for downloading models)

### Check Python Version

```bash
python3 --version
# Should show Python 3.10.x or higher
```

If you don't have Python 3.10+, install it:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

**macOS (with Homebrew):**
```bash
brew install python@3.10
```

**Windows:**
Download from https://www.python.org/downloads/

## Step 1: Clone Repository

```bash
# Clone your repository
git clone https://github.com/yourusername/ffn-checker.git
cd ffn-checker

# Or if you're starting fresh, create directory
mkdir ffn-checker
cd ffn-checker
```

## Step 2: Create Virtual Environment

### Linux/macOS

```bash
# Create virtual environment
python3 -m venv ffn_env

# Activate it
source ffn_env/bin/activate

# Your prompt should now show (ffn_env)
```

### Windows

```bash
# Create virtual environment
python -m venv ffn_env

# Activate it
ffn_env\Scripts\activate

# Your prompt should now show (ffn_env)
```

## Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

## Step 4: Install PyTorch (CPU Version)

**This is the most important step!** We need CPU-only PyTorch.

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
```

This will download about 200-300MB.

### Verify PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
# Should print: PyTorch 2.3.0+cpu installed
```

## Step 5: Install FFN-Checker Dependencies

```bash
# Install all requirements
pip install -r requirements.txt
```

This will take 5-10 minutes and download about 1-2GB.

### If you get errors:

**Error: "No module named 'pip._vendor.urllib3'"**
```bash
pip install --upgrade pip setuptools wheel
```

**Error: Building wheel for package X failed**
```bash
# Try installing system dependencies (Ubuntu)
sudo apt install python3-dev build-essential
```

## Step 6: Install FFN-Checker

```bash
# Install in editable mode (recommended for development)
pip install -e .

# Or regular installation
# pip install .
```

## Step 7: Verify Installation

Run the quick test to ensure everything works:

```bash
python scripts/quick_test.py
```

**Expected output:**
```
==================================================
FFN-Checker Quick Test
==================================================

Test 1: Fault Injection
----------------------------------------
Original tensor: mean=0.0012, has_inf=False
After INF injection at (0, 5, 42): has_inf=True
After NaN injection at (1, 3, 102): has_nan=True
✓ Fault injection working

Test 2: EEC-ABFT
----------------------------------------
Clean data: shape=torch.Size([2, 4, 768])
Checksums: shape=torch.Size([2, 4, 2])
Errors in clean data: 0
Injected INF at [0, 0, 10]
Errors after injection: 1
Corrected 1 errors
Has INF after correction: False
✓ EEC-ABFT working

Test 3: FFN-Checker
----------------------------------------
Input: shape=torch.Size([2, 10, 768]), mean=0.0023
Standard FFN output: shape=torch.Size([2, 10, 768]), mean=-0.0015
Protected FFN output: shape=torch.Size([2, 10, 768]), mean=-0.0015
Protection stats: {'errors_detected_stage1': 0, 'errors_detected_stage2': 0, 'errors_corrected': 0}

Corrupted input: has_inf=True
Standard FFN with corruption: has_inf=True, has_nan=False
Protected FFN with corruption: has_inf=False, has_nan=False
Protection stats: {'errors_detected_stage1': 1, 'errors_detected_stage2': 0, 'errors_corrected': 1}
✓ FFN-Checker working

Test 4: Performance Overhead
----------------------------------------
Standard FFN: 45.23 ms/iter
Protected FFN: 49.87 ms/iter
Overhead: 10.3%
✓ Performance measurement working

==================================================
✓ All tests passed!
==================================================

Your FFN-Checker installation is working correctly.
```

If you see this, **installation is complete!**

## Troubleshooting

### Issue 1: Out of Memory During Testing

**Solution:** Your system might not have enough RAM. Try:
```bash
# Edit scripts/quick_test.py line with batch size
# Change: x = torch.randn(8, 128, 768)
# To: x = torch.randn(4, 64, 768)
```

### Issue 2: Slow Performance

**Solution:** This is normal on CPU. Expect:
- Quick test: 5-10 minutes
- Full experiments: 8-16 hours

To speed up, reduce iterations:
```bash
python scripts/run_vulnerability_analysis.py --num_faults 100  # Instead of 1000
```

### Issue 3: ModuleNotFoundError

**Solution:** Ensure you're in the virtual environment:
```bash
# Check if (ffn_env) shows in prompt
# If not, activate it again:
source ffn_env/bin/activate  # Linux/macOS
# or
ffn_env\Scripts\activate  # Windows
```

### Issue 4: Can't Download Models

**Solution:** HuggingFace models need internet. If behind proxy:
```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

Or download manually and place in `~/.cache/huggingface/`

### Issue 5: Permission Denied

**Solution:** Don't use sudo with pip in venv:
```bash
# Wrong: sudo pip install ...
# Right: pip install ...
```

## Next Steps

After successful installation:

1. **Quick validation (5-10 min):**
   ```bash
   python scripts/quick_test.py
   ```

2. **Run small experiment (1-2 hours):**
   ```bash
   python scripts/run_vulnerability_analysis.py --model bert-base-uncased --num_faults 100
   ```

3. **Run full experiments (8-16 hours):**
   ```bash
   bash scripts/run_all_experiments.sh
   ```

4. **Analyze results:**
   ```bash
   jupyter notebook analysis/analyze_results.ipynb
   ```

## Uninstallation

To remove FFN-Checker:

```bash
# Deactivate virtual environment
deactivate

# Remove directory
cd ..
rm -rf ffn-checker

# Or keep code but remove venv
cd ffn-checker
rm -rf ffn_env
```

## Getting Help

- **Issues:** https://github.com/yourusername/ffn-checker/issues
- **Discussions:** https://github.com/yourusername/ffn-checker/discussions
- **Email:** your.email@university.edu

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16GB | 32GB |
| Storage | 50GB | 100GB |
| Python | 3.10 | 3.11 |
| OS | Linux, macOS, Windows | Linux |
| GPU | Not required | Not used |

## Installation Time Estimate

- Steps 1-3: 5 minutes
- Step 4 (PyTorch): 5-10 minutes
- Step 5 (Dependencies): 5-10 minutes
- Step 6-7 (Install & Test): 10-15 minutes

**Total: 25-40 minutes**

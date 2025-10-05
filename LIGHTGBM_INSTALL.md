# LightGBM Install Tips (Optional)

If you want to enable the LightGBM model locally:

## Windows
1. Install Microsoft C++ Build Tools (via Visual Studio Installer → C++ build tools).
2. Then in your venv:
   ```bash
   pip install lightgbm
   ```

## macOS
1. Install Homebrew (if not already): https://brew.sh
2. Install OpenMP:
   ```bash
   brew install libomp
   ```
3. Then:
   ```bash
   pip install lightgbm
   ```

## Linux (Debian/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install -y libomp-dev build-essential cmake
pip install lightgbm
```

## Conda (Any OS)
```bash
conda install -c conda-forge lightgbm
```

If it still fails, you can run without LightGBM using:
```bash
pip install -r requirements_no_lgbm.txt
python train.py --model lgbm   # will auto-fallback to RandomForest
```

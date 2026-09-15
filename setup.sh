#!/bin/bash
set -e

echo "🚀 Setting up Boltz Design Environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Create and activate environment
echo "📦 Creating conda environment..."
conda create -n boltz_design python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate boltz_design

# Install boltz
if [ -d "boltz2" ]; then
    echo "📂 Installing Boltz..."
    cd boltz2
    pip install -e .
    cd ..
else
    echo "❌ boltz directory not found. Please run this script from the project root."
    exit 1
fi
# Install conda dependencies
echo "🔧 Installing conda dependencies..."
conda install -c anaconda ipykernel -y

# Install Python dependencies
echo "🔧 Installing Python dependencies..."
pip install matplotlib seaborn prody tqdm PyYAML requests pypdb py3Dmol logmd==0.1.45

# Install PyRosetta
echo "⏳ Installing PyRosetta (this may take a while)..."
pip install pyrosettacolabsetup pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'

# Download Boltz weights and dependencies
echo "⬇️  Downloading Boltz weights and dependencies..."
python -c "
from boltz.main import download_boltz2, download_boltz1
from pathlib import Path
cache = Path('~/.boltz').expanduser()
cache.mkdir(parents=True, exist_ok=True)
download_boltz2(cache)
download_boltz1(cache)
print('✅ Boltz weights downloaded successfully!')
"

# Setup LigandMPNN if directory exists
if [ -d "LigandMPNN" ]; then
    echo "🧬 Setting up LigandMPNN..."
    cd LigandMPNN
    bash get_model_params.sh "./model_params"
    cd ..
fi

# ---------------------------------------------------------------------------
# AlphaFold 3 validation environment (optional but recommended)
#
# AlphaFold 3 is NOT bundled: its source is licensed CC BY-NC-SA 4.0 and its model
# parameters must be requested from Google DeepMind. Obtain both yourself:
#   https://github.com/google-deepmind/alphafold3
#
# AF3 runs on JAX while BoltzDesign runs on PyTorch, so AF3 lives in its own conda
# env and the pipeline calls it as a subprocess (boltzdesign/af3_driver.py).
# ---------------------------------------------------------------------------
AF3_ROOT="${AF3_ROOT:-$HOME/alphafold3}"
AF3_ENV="${AF3_ENV:-$HOME/.conda/envs/af3}"

if [ ! -d "$AF3_ROOT" ]; then
    echo "⬇️  Cloning AlphaFold 3 source to $AF3_ROOT ..."
    git clone https://github.com/google-deepmind/alphafold3.git "$AF3_ROOT" || \
        echo "⚠️  Clone failed - clone it manually to $AF3_ROOT."
fi

if [ -d "$AF3_ROOT" ]; then
    echo "🧪 Setting up AlphaFold 3 validation env at $AF3_ENV ..."
    if [ ! -x "$AF3_ENV/bin/python" ]; then
        conda create -p "$AF3_ENV" python=3.11 -y
    fi
    "$AF3_ENV/bin/pip" install --upgrade pip
    # AlphaFold 3's own install sequence (see its docker/Dockerfile)
    ( cd "$AF3_ROOT" \
      && "$AF3_ENV/bin/pip" install -r dev-requirements.txt \
      && "$AF3_ENV/bin/pip" install --no-deps . \
      && "$AF3_ENV/bin/build_data" ) || \
        echo "⚠️  AlphaFold 3 install failed - follow $AF3_ROOT/docs/installation.md manually."
    # extra packages the validator uses
    "$AF3_ENV/bin/pip" install pandas gemmi biopython
    echo "✅ AF3 env ready: $AF3_ENV/bin/python"
else
    echo "ℹ️  Skipping AlphaFold 3 setup (no checkout at $AF3_ROOT)."
fi

echo ""
echo "🔑 AlphaFold 3 MODEL PARAMETERS are not downloadable here."
echo "   They must be requested from Google DeepMind (terms of use apply):"
echo "   https://github.com/google-deepmind/alphafold3#obtaining-model-parameters"
echo "   Place them in $AF3_ROOT/models"

# Make DAlphaBall.gcc executable
chmod +x "boltzdesign/DAlphaBall.gcc" || { echo -e "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }

# Setup Jupyter kernel for the environment
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name=boltz_design --display-name="Boltz Design 2"

echo ""
echo "🎉 Installation complete!"
echo "   BoltzDesign : conda activate boltz_design   (Boltz-1 + Boltz-2 weights in ~/.boltz)"
echo "   LigandMPNN  : LigandMPNN/model_params"
echo "   AlphaFold 3 : $AF3_ENV/bin/python  (weights from DeepMind in $AF3_ROOT/models)"

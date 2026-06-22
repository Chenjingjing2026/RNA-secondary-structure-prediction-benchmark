# GCNfold Installation and Usage Guide

---

## 🔧 Installation Steps

### 1️⃣ Install RNAplfold

`RNAplfold` is a tool from the ViennaRNA package that generates the pairing probability of each base in the RNA sequence. It is a core dependency of GCNfold.

**Steps:**

```bash
# 1. Create and enter the installation directory
mkdir rnafold && cd rnafold

# 2. Download the ViennaRNA source code
wget https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_5_x/ViennaRNA-2.5.0.tar.gz

# 3. Extract and set permissions (replace the path with your actual directory)
tar xvzf ViennaRNA-2.5.0.tar.gz
chmod 755 -R /root/rnafold

# 4. Compile and install
cd ViennaRNA-2.5.0
./configure
make
sudo make install
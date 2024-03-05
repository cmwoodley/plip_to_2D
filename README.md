## PLIP To 2D

A package to run [PLIP](https://plip-tool.biotec.tu-dresden.de/plip-web/plip/index) analysis on a ligand bound pdb and generate a 2D representation of binding interactions.

### Installation
1. Clone or download this repo
```
git clone https://github.com/cmwoodley/plip_to_2D.git
```
2. Create environment and install dependencies
```
conda create -n PLIP_2D -c conda-forge openbabel plip pandas numpy pymol-open-source
conda activate PLIP_2D
pip install pycairo
```


### Running analysis

1. By command line

plip_2D.py provides interactive commandline functionality. Running analysis is as simple as:
```
python .\plip_2D.py -f .\example_pdb\5ml3.pdb
```
**utils.py must be in the same folder as plip_2D.py**

Output of ```python .\plip_2D.py -h```:
```
usage: plip_2D.py [-h] -f FILE [--pymol PYMOL] [--canvas_height CANVAS_HEIGHT] [--canvas_width CANVAS_WIDTH]
                  [-o OUT_FILE] [-y ANALYSE_ALL]

Generate 2D and 3D representations from X-ray or simulated protein-ligand binding poses

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  pdb for analysis
  --pymol PYMOL         Save pymol session
  --canvas_height CANVAS_HEIGHT
                        Output Canvas Height
  --canvas_width CANVAS_WIDTH
                        Output Canvas Width
  -o OUT_FILE, --out_file OUT_FILE
                        Output name - must be a .png or .svg file
  -y ANALYSE_ALL, --analyse_all ANALYSE_ALL
                        Analyse all small molecules
```
2. As an importable function
See example jupyter notebook.

## Limitations and Issues

Label placement in 2D interaction diagram isn't perfect - the ability to output an SVG file editable in vector graphics software somewhat gets around this.

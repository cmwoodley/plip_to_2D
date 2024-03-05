from utils import plip_2d_interactions
from plip.structure.preparation import PDBComplex
import argparse

def parse_args():
    #Load in files for input
    my_parser = argparse.ArgumentParser(description='Generate 2D and 3D representations from X-ray or simulated protein-ligand binding poses')
    my_parser.add_argument('-f','--file', action='store', type=str, required=True, help="pdb for analysis")
    my_parser.add_argument('--pymol', action='store', type=bool, required=False, default="True", help="Save pymol session")
    my_parser.add_argument('--canvas_height', action='store', type=int, required=False, default="400", help="Output Canvas Height")
    my_parser.add_argument('--canvas_width', action='store', type=int, required=False, default="800", help="Output Canvas Width")
    my_parser.add_argument('-o','--out_file', action='store', type=str, required=False, default="PLIP_interactions.png", help="Output name - must be a .png or .svg file")
    my_parser.add_argument('-y','--analyse_all', action='store', type=bool, required=False, default=False, help="Analyse all small molecules")

    args = my_parser.parse_args()
    return args

def main(args):
    my_mol = PDBComplex()
    my_mol.load_pdb(args.file)

    for my_id in [x for x in str(my_mol).split("\n")[1:] if x.split(":")[0] not in ["ARN", "ASH", "GLH", "LYN", "HIE", "HIP"]]:
        user_input = ''

        if not args.analyse_all:
            while True:
                user_input = input('Do you want analyse object {}? y/n: '.format(my_id))

                if user_input.lower() in ['n','y']:
                    break
                else:
                    print('Do you want analyse object {}? y/n: '.format(my_id))    
        else:
            user_input == "y"

        if user_input == "n": 
            continue
        else:
            plip_2d_interactions(args.file, my_id, save_files=True, save_pymol=args.pymol, canvas_height=args.canvas_height, canvas_width=args.canvas_width, out_name=args.out_file)

if __name__ == "__main__":
    args = parse_args()
    main(args)
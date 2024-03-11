from plip.basic import config
from plip.structure.preparation import PDBComplex
from plip.exchange.report import BindingSiteReport
from plip.visualization.visualize import PyMOLVisualizer
from plip.basic.remote import VisualizerData
from plip.basic.supplemental import start_pymol
import pandas as pd
import numpy as np
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
import cairo
from pymol import cmd
import os
import tempfile
from rdkit.Chem import rdDetermineBonds

config.NOHYDRO = True
ob = pybel.ob

def _save_pymol(my_mol, my_id, outdir):
    '''
    Save pymol session from MD snapshot. Needs a plip mol object, bsid and outdir.
    Adapted from PLIP.
    '''
    complex = VisualizerData(my_mol, my_id)
    vis = PyMOLVisualizer(complex)

    lig_members = complex.lig_members
    chain = complex.chain

    ligname = vis.ligname
    hetid = complex.hetid

    metal_ids = complex.metal_ids
    metal_ids_str = '+'.join([str(i) for i in metal_ids])

    start_pymol(run=True, options='-pcq', quiet=True)
    vis.set_initial_representations()

    cmd.load(complex.sourcefile)

    current_name = cmd.get_object_list(selection='(all)')[0]

    current_name = cmd.get_object_list(selection='(all)')[0]
    cmd.set_name(current_name, complex.pdbid)

    cmd.hide('everything', 'all')
    cmd.select(ligname, 'resn %s and chain %s and resi %s*' % (hetid, chain, complex.position))


    # Visualize and color metal ions if there are any
    if not len(metal_ids) == 0:
        vis.select_by_ids(ligname, metal_ids, selection_exists=True)
        cmd.show('spheres', 'id %s and %s' % (metal_ids_str))

    # Additionally, select all members of composite ligands
    if len(lig_members) > 1:
        for member in lig_members:
            resid, chain, resnr = member[0], member[1], str(member[2])
            cmd.select(ligname, '%s or (resn %s and chain %s and resi %s)' % (ligname, resid, chain, resnr))

    cmd.show('sticks', ligname)
    cmd.color('myblue')
    cmd.color('myorange', ligname)
    cmd.util.cnc('all')
    if not len(metal_ids) == 0:
        cmd.color('hotpink', 'id %s' % metal_ids_str)
        cmd.hide('sticks', 'id %s' % metal_ids_str)
        cmd.set('sphere_scale', 0.3, ligname)
    cmd.deselect()

    vis.make_initial_selections()
    vis.show_hydrophobic()  # Hydrophobic Contacts
    vis.show_hbonds()  # Hydrogen Bonds
    vis.show_halogen()  # Halogen Bonds
    vis.show_stacking()  # pi-Stacking Interactions
    vis.show_cationpi()  # pi-Cation Interactions
    vis.show_sbridges()  # Salt Bridges
    vis.show_wbridges()  # Water Bridges
    vis.show_metal()  # Metal Coordination
    vis.refinements()
    vis.zoom_to_ligand()
    vis.selections_cleanup()
    vis.selections_group()
    vis.additional_cleanup()
    vis.save_session(outdir)

def _get_interactions(input_pdb, hydrophobic_df, hbond_df, pi_stacking_df, pi_cation_df, saltbridge_df, coord_dict):
    interactions = []
    centroids = []
    centroid_counter = 0

    for i, row in hbond_df.iterrows():
        if row.protisdon:
            atom = input_pdb.atoms[row.a_orig_idx-1].OBAtom
            int_atom  = atom.GetResidue().GetAtomID(atom).strip()
        else:
            atom = input_pdb.atoms[row.d_orig_idx-1].OBAtom
            int_atom  = atom.GetResidue().GetAtomID(atom).strip()
        
        interactions.append((int_atom, row["restype"]+str(row["resnr"])+"_"+row["reschain"],"HB"))

    for i,row in hydrophobic_df.drop_duplicates(subset=["LIGCARBONIDX","RESTYPE","RESNR","RESCHAIN"]).iterrows():
        atom = input_pdb.atoms[row.LIGCARBONIDX-1].OBAtom
        int_atom  = atom.GetResidue().GetAtomID(atom).strip()
        interactions.append((int_atom, row["RESTYPE"]+str(row["RESNR"])+"_"+row["RESCHAIN"], "HPI"))

    for df, int_type in zip([pi_stacking_df,pi_cation_df,saltbridge_df],
                                    ["PS","PC","SB"]):
        for i,row in df.drop_duplicates(subset=["LIG_IDX_LIST","RESTYPE","RESNR","RESCHAIN"]).iterrows():

            atoms = [input_pdb.atoms[x-1].OBAtom for x in np.array(row["LIG_IDX_LIST"].split(","), dtype=int)]
            com = np.stack([coord_dict[atom.GetResidue().GetAtomID(atom).strip()] for atom in atoms]).mean(axis=0)
            if (com[0],com[1],"centroid","centroid_{}".format(i)) not in centroids:
                centroids.append((com[0],com[1],"centroid","centroid_{}".format(centroid_counter)))
                coord_dict["centroid_{}".format(centroid_counter)] = (com[0],com[1])
                centroid_counter += 1
                interactions.append(("centroid_{}".format(centroid_counter-1), row["RESTYPE"]+str(row["RESNR"])+"_"+row["RESCHAIN"],int_type))
            else:
                centroid_index = centroids.index((com[0],com[1],"centroid","centroid_{}".format(i)))
                interactions.append(("centroid_{}".format(centroid_index), row["RESTYPE"]+str(row["RESNR"])+"_"+row["RESCHAIN"],int_type))

    used_res = np.unique(np.array([x[1] for x in interactions]))
    return interactions, centroids, used_res

def _get_res_info(used_res, coord_dict, interactions):
    '''
    Greedy label placement algorithm.
    Calculates number of overlaps in a grid of candidate positions.
    Returns a solution with fewest overlaps.
    '''
    res_info = []
    _coord_temp = np.array([np.array(coord_dict[key]) for key in coord_dict.keys()])
    for i,res in enumerate(used_res):
        res_ints = [x[0] for x in interactions if x[1] == res]
        initial_lab_coords = np.array([coord_dict[x] for x in res_ints]).mean(axis=0)
        candidates = (np.mgrid[-3.5:3.6:0.1, -3.5:3.6:0.1].reshape(2,-1).T) + initial_lab_coords

        overlaps = []
        for cand in candidates:
            overlap_count = 0
            for coord in _coord_temp:
                if np.abs(cand-coord)[0] < 2 and np.abs(cand-coord)[1] < 2:
                    overlap_count += 1
            for coord in [(x[0],x[1]) for x in res_info]:
                if np.abs(cand-coord)[0] < 2.5 and np.abs(cand-coord)[1] < 2:
                    overlap_count += 1

            overlaps.append(overlap_count)

        lab_coords = candidates[np.argmin(overlaps)]

        coord_dict[res] = (lab_coords[0],lab_coords[1])
        res_info.append((lab_coords[0],lab_coords[1],"residue",res))

    return res_info

def _draw_mol(atom_info, connections, padding, canvas_height, canvas_width, out_name):
    # Define padding
    padding = padding

    if canvas_height <= 800:
        sgl_witdh = 2
        dbl_width = 6
        font_weight = 12

    else:
        sgl_witdh = 3
        dbl_width = 8
        font_weight = 15

    color_dict = {"O":(1, 0, 0),
                    "N":(0,0,1.0),
                    "S":(0.9,0.775,0.25),
                    "P":(1.0,0.5,0),
                    "B":(1.0,0.71,0.71)}

    # Set canvas size including padding
    # canvas_width, canvas_height = canvas_height + 2 * padding, 800 + 2 * padding

    if out_name is None:
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, canvas_width, canvas_height)
    else:
        if out_name.lower().endswith(".svg"):
            surface = cairo.SVGSurface(out_name, canvas_width, canvas_height)
        else:
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, canvas_width, canvas_height)

    ctx = cairo.Context(surface)

    # Set background color
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()

    # Set line width
    ctx.set_line_width(4)

    # Draw a simple graph with labels and bond types
    data_points = atom_info

    # Calculate scaling factors to fit within the canvas
    min_x, min_y = min(point[0] for point in data_points), min(point[1] for point in data_points)
    max_x, max_y = max(point[0] for point in data_points), max(point[1] for point in data_points)

    x_scale = y_scale = min((canvas_width - 2.5 * padding) / (max_x - min_x), (canvas_height - 2.5 * padding) / (max_y - min_y))

    # Calculate modifying factors to center molecule on canvas

    canvc_x = canvas_width/2
    canvc_y = canvas_height/2

    cent_x, cent_y = np.array([x[:2] for x in data_points]).mean(axis=0)
    cent_x = (cent_x - min_x) * x_scale + padding
    cent_y = (cent_y - min_y) * y_scale + padding

    mod_x = canvc_x - cent_x
    mod_y = canvc_y - cent_y

    # Draw connections with different line styles based on bond type
    for start, end, bond_type in connections:
        start_x, start_y, _, _ = data_points[start]
        end_x, end_y, _, _ = data_points[end]
        jitter = np.random.uniform(-10,10)

        # Apply scaling and padding to coordinates
        start_x = (start_x - min_x) * x_scale + padding + mod_x
        start_y = (start_y - min_y) * y_scale + padding + mod_y     
        end_x = (end_x - min_x) * x_scale + padding + mod_x
        end_y = (end_y - min_y) * y_scale + padding + mod_y

        # Set line style based on bond type
        if bond_type == "SINGLE":
            ctx.set_source_rgb(0, 0, 0)  
            ctx.set_line_width(sgl_witdh)  
            ctx.move_to(start_x, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()
            ctx.set_line_width(0)

        elif bond_type == "DOUBLE":
            ctx.set_source_rgb(0,0,0)  
            ctx.set_line_width(dbl_width)  
            ctx.move_to(start_x, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()
            ctx.set_source_rgb(1,1,1) 
            ctx.set_line_width(2)  
            ctx.move_to(start_x, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()
            ctx.set_line_width(0)

        elif bond_type == "AROMATIC":
            ctx.set_source_rgb(0,0,0)  
            ctx.set_line_width(dbl_width)  
            ctx.set_dash([10, 5], 0)  
            ctx.move_to(start_x, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()
            ctx.set_source_rgb(1,1,1)  
            ctx.set_dash([], 0)  
            ctx.set_line_width(2)  
            ctx.move_to(start_x, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()
            ctx.set_line_width(0)

        elif bond_type == "TRIPLE":
            ctx.set_source_rgb(0,0,0)  
            ctx.set_line_width(10)  
            ctx.move_to(start_x, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()
            ctx.set_source_rgb(1,1,1)  
            ctx.set_line_width(6)  
            ctx.move_to(start_x, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()
            ctx.set_source_rgb(0,0,0)  
            ctx.set_line_width(2)  
            ctx.move_to(start_x, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()
            ctx.set_line_width(0)
        
        elif bond_type == "HPI":
            ctx.set_source_rgba(0.5,0.5,0.5, 0.7)  
            ctx.set_line_width(3) 
            ctx.set_dash([10, 5], 0)  
            ctx.move_to(start_x, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()
            ctx.set_line_width(0)
        elif bond_type == "HB":
            ctx.set_source_rgba(0, 0, 1, 0.7)  
            ctx.set_line_width(3) 
            ctx.set_dash([10, 5], 0) 
            ctx.move_to(start_x+7, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()
            ctx.set_line_width(0)
        elif bond_type == "PS":
            ctx.set_source_rgba(0, 0.6, 0, 0.7)  
            ctx.set_line_width(3) 
            ctx.set_dash([10, 5], 0)  
            ctx.move_to(start_x+15, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()
            ctx.set_line_width(0)
        elif bond_type == "PC":
            ctx.set_source_rgba(1, 0.7, 0, 0.7)  
            ctx.set_line_width(3) 
            ctx.set_dash([10, 5], 0)  
            ctx.move_to(start_x-15, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()
            ctx.set_line_width(0)
        elif bond_type == "SB":
            ctx.set_source_rgba(1, 0, 1, 0.7)  
            ctx.set_line_width(3) 
            ctx.set_dash([10, 5], 0)  
            ctx.move_to(start_x-7, start_y)
            ctx.line_to(end_x, end_y)
            ctx.stroke()
            ctx.set_line_width(0)

    # Draw filled circles and labels with padding
    for x, y, label, res in data_points:
        if label in ["C", "centroid"]:
            continue

        # Draw a filled white circle at each data point
        ctx.set_source_rgb(1, 1, 1)  
        if res == "Charge":
            ctx.arc((x - min_x) * x_scale + padding + mod_x, (y - min_y) * y_scale + padding + mod_y, 5, 0, 2 * 3.14)
        elif label == "residue":
            ctx.arc((x - min_x) * x_scale + padding + mod_x, (y - min_y) * y_scale + padding + mod_y, 0, 0, 2 * 3.14)
        else:
            ctx.arc((x - min_x) * x_scale + padding + mod_x, (y - min_y) * y_scale + padding + mod_y, 10, 0, 2 * 3.14)
        ctx.fill_preserve()  
        ctx.stroke()
            
        # Set font settings
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(font_weight)

        # Calculate text width and height
        text_extents = ctx.text_extents(res if label == "residue" else label)
        text_width = text_extents[2]
        text_height = text_extents[3]

        # Position text at the center of the point with padding
        text_x = (x - min_x) * x_scale + padding + mod_x - text_width / 2
        text_y = (y - min_y) * y_scale + padding + mod_y + text_height / 2
        ctx.move_to(text_x, text_y)

        if label == "residue":
            # Draw a white rectangle behind the text for label "residue"
            ctx.rectangle(text_x - 2, text_y - text_height, text_width + 4, text_height + 2)
            ctx.set_source_rgba(1,1,1,1)  # White color
            ctx.fill()
        
        if label in ["O","N","S","P","B"]:
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            color = color_dict[label]
            ctx.set_source_rgba(color[0],color[1],color[2],1)
        else:
            ctx.set_source_rgba(0, 0, 0, 1)
        ctx.move_to(text_x, text_y)
        ctx.text_path(res if label == "residue" else label)
        ctx.fill()

    return ctx, surface

###### Function from  matteoferla on Github https://gist.github.com/matteoferla/94eb8e4f8441ddfb458bfc45722469b8 ######

def set_to_neutral_pH(mol: Chem):
    """
    Not great, but does the job.
    
    * Protonates amines, but not aromatic bound amines.
    * Deprotonates carboxylic acid, phosphoric acid and sulfuric acid, without ruining esters.
    """
    protons_added = 0
    protons_removed = 0
    for indices in mol.GetSubstructMatches(Chem.MolFromSmarts('[N;D1]')):
        atom = mol.GetAtomWithIdx(indices[0])
        if atom.GetNeighbors()[0].GetIsAromatic():
            continue # aniline
        atom.SetFormalCharge(1)
        protons_added += 1
    for indices in mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)[O;D1]')):
        atom = mol.GetAtomWithIdx(indices[2])
        # benzoic acid pKa is low.
        atom.SetFormalCharge(-1)
        protons_removed += 1
    for indices in mol.GetSubstructMatches(Chem.MolFromSmarts('P(=O)[Oh1]')):
        atom = mol.GetAtomWithIdx(indices[2])
        # benzoic acid pKa is low.
        atom.SetFormalCharge(-1)
        protons_removed += 1
    for indices in mol.GetSubstructMatches(Chem.MolFromSmarts('S(=O)(=O)[Oh1]')):
        atom = mol.GetAtomWithIdx(indices[3])
        # benzoic acid pKa is low.
        atom.SetFormalCharge(-1)
        protons_removed += 1
    return (protons_added, protons_removed)

# Define periodic table dictionary
periodic_table = {
    1: 'H',  2: 'He', 3: 'Li', 4: 'Be', 5: 'B',  6: 'C',  7: 'N',  8: 'O',  9: 'F',  10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
    21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
    41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
    61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
    91: 'Pa', 92: 'U',  93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
    101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt',
    110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og',
}

def convert_and_write_pdb(input_pdb, output_pdb, bsid):
    # Load PDB from string
    pdb = [x for x in pybel.readfile("pdb", input_pdb)][0]

    for res in ob.OBResidueIter(pdb.OBMol):
        if not res.GetResidueProperty(0):
            if not res.GetName().isalnum():
                res.SetName("LIG")
            if res.GetChain().isspace():
                res.SetChain(bsid.split(":")[1])    
        for i,atom in enumerate(ob.OBResidueAtomIter(res)):
            if not res.GetAtomID(atom).isalnum():
                atom_name = periodic_table[atom.GetAtomicNum()]+str(i+1)
                res.SetAtomID(atom, atom_name)

    pdb.addh()
    pdb.write("pdb", output_pdb, overwrite=True)

    return pdb

keys = (
    "hydrophobic",
    "hbond",
    "waterbridge",
    "saltbridge",
    "pistacking",
    "pication",
    "halogen",
    "metal",
)

hbkeys = [
    "resnr",
    "restype",
    "reschain",
    "resnr_l",
    "restype_l",
    "reschain_l",
    "sidechain",
    "distance_ah", 
    "distance_ad", 
    "angle", 
    "type",
    "protisdon",
    "d_orig_idx",
    "a_orig_idx",
    "h"
]

def plip_2d_interactions(file, bsid, padding=40, canvas_height=500, canvas_width=800, save_files=True, save_pymol=True, out_name="PLIP_interactions.png"):

    if not save_files:
        out_name = None
        save_pymol = False

    outdir = "{}/{}_output".format(os.path.split(file)[0], os.path.split(file)[1].split(".")[0])
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    outdir = "{}/{}".format(outdir,"_".join(bsid.split(":")))
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    file_prot = outdir+"/{}_prot.pdb".format(os.path.split(file)[1].split(".")[0])
    input_pdb = convert_and_write_pdb(file, file_prot, bsid)

    my_mol = PDBComplex()
    my_mol.load_pdb(file_prot)

    my_mol.analyze()
    my_interactions = my_mol.interaction_sets[bsid]

    if save_pymol:
        _save_pymol(my_mol, bsid, outdir)
        
    bsr = BindingSiteReport(my_interactions)

    interactions = {
        k: [getattr(bsr, k + "_features")] + getattr(bsr, k + "_info")
        for k in keys
    }

    hydrophobic_df = pd.DataFrame(interactions["hydrophobic"][1:], columns=interactions["hydrophobic"][0])
    hbond_df = []
    for hb in my_interactions.all_hbonds_pdon + my_interactions.all_hbonds_ldon:
        hb_interactions = []
        for k in hbkeys:
            hb_interactions.append(getattr(hb, k))

        hbond_df.append(np.array(hb_interactions))
    if len(hbond_df) != 0:
        hbond_df = pd.DataFrame(np.stack(hbond_df), columns=hbkeys)
        hbond_df["h"] = [x.idx for x in hbond_df["h"]]
    else:
        hbond_df = pd.DataFrame()
    pi_stacking_df = pd.DataFrame(interactions["pistacking"][1:], columns=interactions["pistacking"][0])
    pi_cation_df = pd.DataFrame(interactions["pication"][1:], columns=interactions["pication"][0])
    saltbridge_df = pd.DataFrame(interactions["saltbridge"][1:], columns=interactions["saltbridge"][0])

    if save_files:
        if len(hydrophobic_df) > 0:
            hydrophobic_df.to_csv(outdir+"/{}_HPI.csv".format(os.path.split(file)[1].split(".")[0]), index=False)
        if len(hbond_df) > 0:
            hbond_df.to_csv(outdir+"/{}_HB.csv".format(os.path.split(file)[1].split(".")[0]), index=False)
        if len(pi_stacking_df) > 0:
            pi_stacking_df.to_csv(outdir+"/{}_PS.csv".format(os.path.split(file)[1].split(".")[0]), index=False)
        if len(pi_cation_df) > 0:
            pi_cation_df.to_csv(outdir+"/{}_PC.csv".format(os.path.split(file)[1].split(".")[0]), index=False)
        if len(saltbridge_df) > 0:
            saltbridge_df.to_csv(outdir+"/{}_SB.csv".format(os.path.split(file)[1].split(".")[0]), index=False)

    with open(file_prot,"r") as f:
        pdb = f.readlines()

    pdb = [line for line in pdb if line.startswith(("ATOM","HETATM"))]

    with tempfile.TemporaryDirectory() as temp_dir:
        lig_path = os.path.join(temp_dir, "lig.pdb")

        lig = "".join([line for line in pdb if ((line[17:20] == bsid.split(":")[0])&(line[21] == bsid.split(":")[1])&(line[22:26].strip() == bsid.split(":")[2]))])

        ## Pybel seems to protonate differently when running from a jupyter notebook versus command line
        ## No idea why this is, but this is a workaround

        try:
            mol = Chem.MolFromPDBBlock(lig, removeHs=False)
            rdDetermineBonds.DetermineBonds(mol, charge=0)
        except:
            mol = pybel.readstring("pdb",lig)
            mol.write("pdb",lig_path, overwrite=True)
            mol = Chem.MolFromPDBFile(lig_path, removeHs=False)

    AllChem.EmbedMolecule(mol)
    set_to_neutral_pH(mol) #### Helper function to protonate/deprotonate groups. ####

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            bound_atom = atom.GetBonds()[0].GetEndAtom()
            if bound_atom.GetIdx == atom.GetIdx():
                bound_atom = atom.GetBonds()[0].GetBeginAtom()

            if bound_atom.GetSymbol() in ["O","N","S"]: #### Workaround so RDKit keeps explicit polar H's ####
                atom.SetAtomicNum(100) 

    mol=Chem.RemoveHs(mol)

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 100:
            atom.SetAtomicNum(1)

    Chem.rdDepictor.Compute2DCoords(mol)

    atom_info = []
    charge_info = []
    bonds = []
    for i,atom in enumerate(mol.GetAtoms()):
        coords = mol.GetConformer().GetAtomPosition(i)
        atom_info.append((coords.x, coords.y, atom.GetSymbol(), atom.GetPDBResidueInfo().GetName().strip()))

        if atom.GetFormalCharge() < 0:
            charge_info.append((coords.x+0.3, coords.y-0.2, "–", "Charge"))
        if atom.GetFormalCharge() > 0:
            charge_info.append((coords.x+0.3, coords.y-0.2, "+", "Charge"))
            
        startatoms = [bond.GetBeginAtomIdx() for bond in atom.GetBonds()]
        endatoms = [bond.GetEndAtomIdx() for bond in atom.GetBonds()]
        bond_type = [str(bond.GetBondType()).split(".")[-1] for bond in atom.GetBonds()]

        for a,b,c in zip(startatoms,endatoms,bond_type):
            if (a,b,c) not in bonds and (b,a,c) not in bonds:
                bonds.append((a,b,c))

    coord_dict = {}
    for entry in atom_info:
        coord_dict[entry[3]] = (entry[0],entry[1])

    interactions, centroids, used_res = _get_interactions(input_pdb, hydrophobic_df, hbond_df, pi_stacking_df, pi_cation_df, saltbridge_df, coord_dict)

    res_info = _get_res_info(used_res, coord_dict, interactions)

    atom_info = atom_info + centroids
    lines = []
    for i in range(len(res_info)):
        res = res_info[i][3]
        for j in range(len(interactions)):
            if interactions[j][1] == res:
                atom = interactions[j][0]
                atom_index = [x[3] for x in atom_info].index(atom)
                lines.append((i+len(atom_info), atom_index, interactions[j][2]))

    atom_info = atom_info + res_info + charge_info

    connections = bonds + lines

    out_name = outdir + "/{}".format(out_name)

    ctx, surface = _draw_mol(atom_info, connections, 40, canvas_height, canvas_width, out_name)


    if canvas_height <= 800:
        legend_scale = 7
    else:
        legend_scale = 9

    # Define legend items
    legend_items = [
        ("Hydrophobic", (0.5,0.5,0.5)),
        ("H-bond", (0, 0, 1)),
        ("Pi-Stacking", (0,0.6,0)),
        ("Pi-cation", (1,0.7,0)),
        ("Salt-bridge", (1,0,1)),
    ]

    # Calculate legend size
    legend_width = np.sum([40+len(x[0])*legend_scale for x in legend_items])
    legend_height = 20
    legend_x, legend_y = (canvas_width - legend_width)/2 , canvas_height - padding/3

    # Draw legend rectangle
    ctx.set_source_rgb(0,0,0)  # Black color
    ctx.set_line_width(2)
    ctx.set_dash([], 0)  # Set dash pattern for aromatic bond

    ctx.rectangle(legend_x-5, legend_y - legend_height + 10, legend_width, legend_height)
    ctx.stroke_preserve()
    ctx.set_source_rgb(1,1,1)
    ctx.fill()

    # Draw legend items
    for label, color in legend_items:
        ctx.set_line_width(2)
        ctx.set_source_rgba(color[0], color[1], color[2],0.8)  
        ctx.set_dash([10, 5], 0)  


        # Draw legend line
        ctx.move_to(legend_x, legend_y)
        ctx.line_to(legend_x + 30, legend_y)
        ctx.stroke()

        # Draw legend label
        ctx.set_source_rgb(0, 0, 0)
        ctx.move_to(legend_x + 40, legend_y + 5)
        ctx.show_text(label)

        # Move along for the next legend item
        legend_x += 40+len(label)*legend_scale

    # Save the image to a file
    if save_files:
        if out_name.lower().endswith(".png"):
            surface.write_to_png(out_name)
        elif out_name.lower().endswith(".svg"):
            surface.finish()
        else:
            raise ValueError("Unsupported file format. Please provide either PNG or SVG extension.")
    else:
        try:
            from IPython.display import display, Image
            import io
            image_stream = io.BytesIO()
            surface.write_to_png(image_stream)
            display(Image(data=image_stream.getvalue(), format="png"))
        except:
            None
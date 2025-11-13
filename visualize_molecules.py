#!/usr/bin/env python3
"""
Simple script to visualize generated ligands with their protein pocket using py3Dmol
"""
import argparse
from pathlib import Path
import py3Dmol

def visualize_ligands(pdb_file, sdf_file, width=800, height=600, molecule_only=False):
    """
    Visualize protein structure with generated ligands

    Args:
        pdb_file: Path to PDB file
        sdf_file: Path to SDF file with generated molecules
        width: Viewer width in pixels
        height: Viewer height in pixels
        molecule_only: If True, only show the molecule without protein
    """
    # Create viewer
    view = py3Dmol.view(width=width, height=height)

    # Add protein structure (unless molecule_only)
    if not molecule_only and pdb_file:
        with open(pdb_file, 'r') as f:
            pdb_data = f.read()
        view.addModel(pdb_data, 'pdb')
        view.setStyle({'model': -1}, {'cartoon': {'color': 'lightblue'}})

    # Add generated ligands
    with open(sdf_file, 'r') as f:
        sdf_data = f.read()
    view.addModelsAsFrames(sdf_data)
    view.setStyle({'model': -1}, {'stick': {'colorscheme': 'greenCarbon'}})

    # Set view
    view.zoomTo({'model': -1})
    view.zoom(0.8)

    # Animate through different generated molecules
    view.animate({'loop': 'forward', 'interval': 2000})

    # Save to HTML for browser viewing
    html_file = str(Path(sdf_file).with_suffix('.html'))
    with open(html_file, 'w') as f:
        f.write(view._make_html())

    print(f"\nVisualization saved to: {html_file}")
    print("Open this file in your browser to view the molecule(s)")

    return view


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize protein-ligand complexes')
    parser.add_argument('--pdbfile', type=str, help='Path to PDB file (optional if --molecule-only)')
    parser.add_argument('--sdffile', type=str, required=True, help='Path to SDF file with generated molecules')
    parser.add_argument('--width', type=int, default=800, help='Viewer width')
    parser.add_argument('--height', type=int, default=600, help='Viewer height')
    parser.add_argument('--molecule-only', action='store_true', help='Show only molecule without protein')
    args = parser.parse_args()

    if args.pdbfile and not args.molecule_only:
        print(f"Loading protein from: {args.pdbfile}")
    print(f"Loading ligands from: {args.sdffile}")
    print("Creating visualization...")

    view = visualize_ligands(args.pdbfile, args.sdffile, args.width, args.height, args.molecule_only)

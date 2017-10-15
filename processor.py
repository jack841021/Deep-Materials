import json
import math
import pymatgen
from scipy.interpolate import spline

prtable = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']

def cubic_searcher():
    with pymatgen.MPRester('3GmMFkADz6xU7aJ1') as REST:
        infos = REST.query(criteria={'has_bandstructure': True, 'is_hubbard': False, 'crystal_system': 'cubic'}, properties=['material_id'])
    ids = []
    for info in infos:
        ids += [info['material_id']]
    with open('saved_ids', 'w') as saved_ids:
        json.dump(ids, saved_ids)
    print(len(ids))

def band_crawler():
    with open('saved_ids', 'r') as saved_ids:
        ids = json.load(saved_ids)
    with pymatgen.MPRester('3GmMFkADz6xU7aJ1') as REST:
        bands = []
        for id in ids:
            band = REST.get_bandstructure_by_material_id(id).as_dict()
            band['id'] = id
            for i in range(len(band['vbm']['kpoint_index'])):
                band['vbm']['kpoint_index'][i] = str(band['vbm']['kpoint_index'][i])
            for i in range(len(band['cbm']['kpoint_index'])):
                band['cbm']['kpoint_index'][i] = str(band['cbm']['kpoint_index'][i])
            bands += [band]
            print(len(bands))
    with open('saved_bands', 'w') as saved_bands:
        json.dump(bands, saved_bands)

def band_trimmer(kpoints):
    with open('saved_bands', 'r') as saved_bands:
        bands = json.load(saved_bands)
    trimmed_bands = []
    for band in bands:
        if len(band['bands']) == 1:
            branches = band['branches']
            for branch in branches:
                if branch['name'] == '\\Gamma-X':
                    start, end, = branch['start_index'], branch['end_index']
                    incre = (end - start) / (kpoints - 1)
                    index = [start + incre * i for i in range(kpoints)]
                    trimmed_band = [band['id']]
                    for ban in band['bands']['1']:
                        x = [start + i for i in range(end - start + 1)]
                        y = ban[start: end + 1]
                        trimmed_band += [list(spline(x, y, index))]
                    trimmed_bands += [trimmed_band]
                    print(len(trimmed_bands))
                    break
    with open('saved_trimmed_bands', 'w') as saved_trimmed_bands:
        json.dump(trimmed_bands, saved_trimmed_bands)

def structure_crawler(gener):
    with open(str(gener) + '/saved_trimmed_bands', 'r') as saved_bands:
        bands = json.load(saved_bands)
    ids = [band[0] for band in bands]
    tem = ids
    batch = 1000
    structures = []
    for i in range(math.ceil(len(ids) / batch)):
        if len(tem) >= batch:
            par = tem[:batch]
            tem = tem[batch:]
        else:
            par = tem
        with pymatgen.MPRester('3GmMFkADz6xU7aJ1') as REST:
            structure = REST.query(criteria={'material_id': {'$in': par}}, properties=['material_id', 'final_structure'], mp_decode=False)
        structures += structure
    with open(str(gener) + '/saved_structures', 'w') as saved_structures:
        json.dump(structures, saved_structures)

#==========================================================================================

def standardizer(gener, kpoint, nband):
    with open(str(gener) + '/saved_trimmed_bands', 'r') as saved_trimmed_bands:
        bands = json.load(saved_trimmed_bands)
    with open(str(gener) + '/saved_structures', 'r') as saved_structures:
        structures = json.load(saved_structures)
    stands = []
    for band in bands:
        eigen = []
        for ban in band[1:]:
            eigen += [ban[kpoint]]
        if (len(eigen) < nband) and (max(eigen) < 50):
            stands += [band[0]]
    smalls = []
    for stand in stands:
        for structure in structures:
            if structure['material_id'] == stand:
                smalls += [structure]
                break
    with open(str(gener) + '/saved_stands_' + str(kpoint), 'w') as saved_stands:
        json.dump(stands, saved_stands)
    with open(str(gener) + '/saved_structures_' + str(kpoint), 'w') as saved_structures:
        json.dump(smalls, saved_structures)
    print('KPOINT ' + str(kpoint) + ': ' + str(len(stands)) + ' ids have been standardized.')

def structure_trimmer(gener, kpoint):
    with open(str(gener) + '/saved_structures_' + str(kpoint), 'r') as saved_structures:
        structures = json.load(saved_structures)
    ids = []
    trimmed = []
    for structure in structures:
        id = structure['material_id']
        lattice = structure['final_structure']['lattice']
        angle = [lattice['alpha'], lattice['beta'], lattice['gamma']]
        const = [lattice['a'],     lattice['b'],    lattice['c']    ]
        sites = [                                                   ]
        for site in structure['final_structure']['sites']:
            sites += [site['label']]
        if round(lattice['a'], 1) == round(lattice['b'], 1) == round(lattice['c'], 1):
            if round(lattice['alpha'], 0) == round(lattice['beta'], 0) == round(lattice['gamma'], 0):
                ids += [id]
                trimmed += [[id, angle, const, sites]]
    with open(str(gener) + '/saved_stands_' + str(kpoint), 'r') as saved_stands:
        stands = json.load(saved_stands)
    new_stands = []
    for stand in stands:
        if stand in ids:
            new_stands += [stand]
    ordered = []
    for stand in new_stands:
        for s in trimmed:
            if s[0] == stand:
                ordered += [s]
                trimmed.remove(s)
                break
    with open(str(gener) + '/saved_stands_' + str(kpoint), 'w') as saved_stands:
        json.dump(new_stands, saved_stands)
    with open(str(gener) + '/saved_trimmed_structures_' + str(kpoint), 'w') as saved_trimmed_structures:
        json.dump(ordered, saved_trimmed_structures)
    print('KPOINT ' + str(kpoint) + ': ' + str(len(ordered)) + ' structures have been trimmed.')

def feature_creator(gener, kpoint, winke, konst, atome):
    with open(str(gener) + '/saved_trimmed_structures_' + str(kpoint), 'r') as saved_trimmed_structures:
        structures = json.load(saved_trimmed_structures)
    features = []
    for structure in structures:
        feature = []
        angle = round(sum(structure[1]) / 3 * 0.0174533, 2)
        const = round(sum(structure[2]) / 3            , 2)
        atoms = [0 for i in range(94)]
        for site in structure[3]:
            atoms[prtable.index(site)] += 1
        if winke == True:
            feature += [angle]
        if konst == True:
            feature += [const]
        if atome == True:
            feature += atoms
        features += [feature]
    with open(str(gener) + '/saved_features_' + str(kpoint), 'w') as saved_features:
        json.dump(features, saved_features)
    print('KPOINT ' + str(kpoint) + ': ' + str(len(features)) + ' features have been created.')

def eigen_extractor(gener, kpoint, nband):
    with open(str(gener) + '/saved_trimmed_bands', 'r') as saved_trimmed_bands:
        bands = json.load(saved_trimmed_bands)
    with open(str(gener) + '/saved_stands_' + str(kpoint), 'r') as saved_stands:
        stands = json.load(saved_stands)
    eigens = []
    for band in bands:
        if band[0] in stands:
            eigen = []
            for ban in band[1:]:
                eigen += [ban[kpoint]]
            eigen += [50 for i in range(nband - len(eigen))]
            eigens += [sorted(eigen)]
    with open(str(gener) + '/saved_eigens_' + str(kpoint), 'w') as saved_eigens:
        json.dump(eigens, saved_eigens)
    print('KPOINT ' + str(kpoint) + ': ' + str(len(eigens)) + ' eigenvalues have been extracted.')

def main(gener, kpoints, nband, angle, const, atoms):
    structure_crawler(gener)
    for kpoint in range(kpoints):
        standardizer     (gener, kpoint, nband)
        structure_trimmer(gener, kpoint)
        feature_creator  (gener, kpoint, angle, const, atoms)
        eigen_extractor  (gener, kpoint, nband)

main(gener = 0, kpoints = 50, nband = 100, angle = True, const = True, atoms = True)

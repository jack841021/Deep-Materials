import json, math
from pymatgen import MPRester
from scipy.interpolate import CubicSpline as spline
from numpy.random import choice


titles = ['saved_bands_0' , 'saved_bands_1' , 'saved_bands_2' , 'saved_bands_3' , 'saved_bands_4' ,
          'saved_bands_5' , 'saved_bands_6' , 'saved_bands_7' , 'saved_bands_8' , 'saved_bands_9' ,
          'saved_bands_10', 'saved_bands_11', 'saved_bands_12', 'saved_bands_13', 'saved_bands_14',
          'saved_bands_15', 'saved_bands_16', 'saved_bands_17', 'saved_bands_18', 'saved_bands_19',
          'saved_bands_20', 'saved_bands_21', 'saved_bands_22', 'saved_bands_23', 'saved_bands_24',
          'saved_bands_25', 'saved_bands_26', 'saved_bands_27', 'saved_bands_28', 'saved_bands_29',
          'saved_bands_30', 'saved_bands_31', 'saved_bands_32', 'saved_bands_33', 'saved_bands_34',
          'saved_bands_35', 'saved_bands_36', 'saved_bands_37', 'saved_bands_38', 'saved_bands_39',
          'saved_bands_40', 'saved_bands_41', 'saved_bands_42', 'saved_bands_43', 'saved_bands_44',
          'saved_bands_45', 'saved_bands_46', 'saved_bands_47', 'saved_bands_48', 'saved_bands_49',
          'saved_bands_50', 'saved_bands_51', 'saved_bands_52', 'saved_bands_53', 'saved_bands_54',
          'saved_bands_55', 'saved_bands_56', 'saved_bands_57', 'saved_bands_58', 'saved_bands_59',
          'saved_bands_60', 'saved_bands_61', 'saved_bands_62', 'saved_bands_63', 'saved_bands_64',
          'saved_bands_65', 'saved_bands_66', 'saved_bands_67']

table = {'H' : 1, 'He': 2, 'Li': 3, 'Be': 4, 'B' : 5, 'C' : 6, 'N' : 7, 'O' : 8, 'F' : 9, 'Ne':10,
         'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P' :15, 'S' :16, 'Cl':17, 'Ar':18, 'K' :19, 'Ca':20,
         'Sc':21, 'Ti':22, 'V' :23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,
         'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36, 'Rb':37, 'Sr':38, 'Y' :39, 'Zr':40,
         'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48, 'In':49, 'Sn':50,
         'Sb':51, 'Te':52, 'I' :53, 'Xe':54, 'Cs':55, 'Ba':56, 'La':57, 'Ce':58, 'Pr':59, 'Nd':60,
         'Pm':61, 'Sm':62, 'Eu':63, 'Gd':64, 'Tb':65, 'Dy':66, 'Ho':67, 'Er':68, 'Tm':69, 'Yb':70,
         'Lu':71, 'Hf':72, 'Ta':73, 'W' :74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80,
         'Tl':81, 'Pb':82, 'Bi':83, 'Ac':89, 'Th':90, 'Pa':91, 'U' :92, 'Np':93, 'Pu':94}

kpoint = {'\\Gamma'  : 1, '\\Sigma'  : 2, '\\Sigma_1': 3, 'A'        : 4, 'A_1'      : 5,
          'B'        : 6, 'B_1'      : 7, 'C'        : 8, 'C_1'      : 9, 'D'        :10,
          'D_1'      :11, 'E'        :12, 'F'        :13, 'F_1'      :14, 'H'        :15,
          'H_1'      :16, 'I'        :17, 'I_1'      :18, 'K'        :19, 'L'        :20,
          'L_1'      :21, 'M'        :22, 'M_1'      :23, 'N'        :24, 'P'        :25,
          'P_1'      :26, 'Q'        :27, 'Q_1'      :28, 'R'        :29, 'S'        :30,
          'T'        :31, 'U'        :32, 'W'        :33, 'X'        :34, 'X_1'      :35,
          'Y'        :36, 'Y_1'      :37, 'Z'        :38, 'Z_1'      :39}


def ksort(l):
    sorted = []
    l_ = l.copy()
    if type(l[0]) == str:
        while len(l_) > 0:
            s = l_[0]
            v = kpoint[l_[0]]
            for j in l_[1:]:
                if kpoint[j] < v:
                    s = j
                    v = kpoint[j]
            sorted += [s]
            l_.remove(s)
    elif type(l[0]) == list:
        while len(l_) > 0:
            s = l_[0]
            sl = len(l_[0])
            for j in l_[1:]:
                if len(j) < sl:
                    s = j
                    sl = len(j)
            sorted += [s]
            l_.remove(s)
    return sorted


def sort(l):
    sorted = []
    l_ = l.copy()
    while len(l_) > 0:
        s = l_[0]
        for j in l_[1:]:
            if j < s:
                s = j
        sorted += [s]
        l_.remove(s)
    return sorted


def hubbard_crawler():
    with MPRester('pvs2K6TlecH3tyGF') as REST:
        hubbards = REST.query(criteria={'is_hubbard' : True}, properties=['material_id', 'hubbards'])
    collated_hubbards = {}
    for hubbard in hubbards:
        collated_hubbards[hubbard['material_id']] = hubbard['hubbards']
    with open('saved_hubbards', 'w') as saved_hubbards:
        json.dump(collated_hubbards, saved_hubbards)


def hubbard_crawler():
    with MPRester('pvs2K6TlecH3tyGF') as REST:
        hubbards = REST.query(criteria={'is_hubbard' : True}, properties=['material_id', 'hubbards'])
    collated_hubbards = {}
    for hubbard in hubbards:
        collated_hubbards[hubbard['material_id']] = hubbard['hubbards']
    with open('saved_hubbards', 'w') as saved_hubbards:
        json.dump(collated_hubbards, saved_hubbards)


def band_crawler(start, end, batch):
    with open('saved_IDs', 'r') as saved_IDs:
        IDs = json.load(saved_IDs)
    with MPRester('pvs2K6TlecH3tyGF') as REST:
        for i in range(start, end + 1):
            bands = []
            for j in range(i * 1000, i * 1000 + batch):
                ID = IDs[j]
                try:
                    band = REST.get_bandstructure_by_material_id(ID).as_dict()
                    band['id'] = ID
                    for k in range(len(band['vbm']['kpoint_index'])):
                        band['vbm']['kpoint_index'][k] = str(band['vbm']['kpoint_index'][k])
                    for k in range(len(band['cbm']['kpoint_index'])):
                        band['cbm']['kpoint_index'][k] = str(band['cbm']['kpoint_index'][k])
                    bands += [band]
                    print(j, 'Y', ID)
                except:
                    print(j, 'N', ID)
            title = titles[i]
            with open(title, 'w') as saved_bands:
                json.dump(bands, saved_bands)


def band_checker(title):
    with open(title, 'r') as collated_bands:
        bands = json.load(collated_bands)
    with open('true', 'r') as saved_true:
        true = json.load(saved_true)
    with open('false', 'r') as saved_false:
        false = json.load(saved_false)
    removed = []
    for band in bands:
        if band[0] not in true:
            if band[0] not in false:
                print(band[0])
                removed += [band[0]]
                bands.remove(band)
    with open('saved_remained', 'w') as saved_remained:
        json.dump(bands, saved_remained)
    with open('saved_removed', 'w') as saved_removed:
        json.dump(removed, saved_removed)


def band_creater(points, spacing, nbands):
    for title in titles:
        with open(title, 'r') as saved_bands:
            bands = json.load(saved_bands)
        collated_bands = []
        for band in bands:
            if band['is_spin_polarized'] == False:
                lock = 0
                fermi = band['efermi']
                kpoints = ksort(list(band['labels_dict'].keys()))
                path = []
                for branch in band['branches']:
                    path += [branch['name']]
                if len(band['bands']['1']) > nbands:
                    if band['is_metal'] == False:
                        upper = band['cbm']['band_index']['1'][0]
                        if upper >= nbands / 2 and len(band['bands']['1']) - (upper + 1) >= nbands / 2:
                            selected_band = [band['bands']['1'][i] for i in range(upper - int(nbands / 2), upper + int(nbands / 2))]
                            band['bands'] = selected_band
                        else:
                            print('discarded')
                            lock = 1
                    else:
                        means = []
                        for ban in band['bands']['1']:
                            total = 0
                            for ba in ban:
                                total += ba
                            means += [abs(total / len(ban) - fermi)]
                        sorted_means = sort(means)[:nbands]
                        selected_band = []
                        for mean in sorted_means:
                            selected_band += [band['bands']['1'][means.index(mean)]]
                        band['bands'] = selected_band
                else:
                    lock = 1
                if lock == 0:
                    n = len(band['bands'][0])
                    x_axis = [i * spacing for i in range(n)]
                    d = (n - 1) * spacing / (points - 1)
                    new_x_axis = [d * i for i in range(points)]
                    totals = []
                    inter_band = []
                    for ban in band['bands']:
                        total = 0
                        inter_ban = []
                        for new_x in new_x_axis:
                            ba = spline(x_axis, ban)(new_x) - fermi
                            inter_ban += [ba]
                            total += ba
                        totals += [total]
                        inter_band += [inter_ban]
                    sorted_totals = sort(totals)
                    sorted_inter_band = []
                    for sorted_total in sorted_totals:
                        sorted_inter_band += [inter_band[totals.index(sorted_total)]]
                    ID = band['id']
                    collated_bands += [[ID, kpoints, path, sorted_inter_band]]
            else:
                print('spin')
            print(bands.index(band) + 1, '/', len(bands))
        with open(title + '_c', 'w') as saved_bands_c:
            json.dump(collated_bands, saved_bands_c)
    collated_bands = []
    for title in titles:
        with open(title + '_c', 'r') as saved_bands_c:
            bands_c = json.load(saved_bands_c)
        collated_bands += bands_c
    with open('saved_collated_bands', 'w') as saved_collated_bands:
        json.dump(collated_bands, saved_collated_bands)


def kpoint_identifier(type):
    with open('saved_collated_bands', 'r') as saved_collated_bands:
        bands = json.load(saved_collated_bands)
    kpoints = []
    if type == 'kpoint':
        choice = 1
    elif type == 'path':
        choice = 2
    for band in bands:
        if band[choice] not in kpoints:
            kpoints += [band[1]]
    sorted_kpoints = ksort(kpoints)
    return sorted_kpoints


def structure_crawler(start, end, index):
    with MPRester('tRKiauSUW6DpO19n') as REST:
        with open('collated_IDs', 'r') as collated_IDs:
            IDs = json.load(collated_IDs)[start:end]
        structures = []
        error = []
        for ID in IDs:
            try:
                structure = REST.get_structure_by_material_id(ID).as_dict()
                del structure['@class'], structure['@module']
                structure['id'] = ID
                structures += [structure]
            except:
                error += [ID]
                print('error:', ID)
            print(len(structures) + len(error), '/', end - start)
        with open('saved_structures_' + index, 'w') as saved_structures:
            json.dump(structures, saved_structures)
        print(error)


def ksort(l):
    sorted = []
    l_ = l.copy()
    if type(l[0]) == str:
        while len(l_) > 0:
            s = l_[0]
            v = kpoint[l_[0]]
            for j in l_[1:]:
                if kpoint[j] < v:
                    s = j
                    v = kpoint[j]
            sorted += [s]
            l_.remove(s)
    elif type(l[0]) == list:
        while len(l_) > 0:
            s = l_[0]
            sl = len(l_[0])
            for j in l_[1:]:
                if len(j) < sl:
                    s = j
                    sl = len(j)
            sorted += [s]
            l_.remove(s)
    return sorted


def sort(l):
    sorted = []
    l_ = l.copy()
    while len(l_) > 0:
        s = l_[0]
        for j in l_[1:]:
            if j < s:
                s = j
        sorted += [s]
        l_.remove(s)
    return sorted


def final_creater(sites, kinds):
    with open('saved_structures', 'r') as saved_structures:
        structures = json.load(saved_structures)
    with open('saved_collated_bands', 'r') as saved_collated_bands:
        bands = json.load(saved_collated_bands)
    with open('saved_hubbards', 'r') as saved_hubbards:
        hubbards = json.load(saved_hubbards)
    collated_structures = []
    discarded = []
    for structure in structures:
        atoms = []
        for site in structure['sites']:
            if site['label'] not in atoms:
                atoms += [site['label']]
        structure['atoms'] = atoms
        if len(structure['sites']) <= sites and len(atoms) <= kinds:
            collated_structures += [structure]
        else:
            discarded += [structure['id']]
    structures = collated_structures
    trimmed_bands = []
    for band in bands:
        if band[0] not in discarded:
            trimmed_bands += [band]
    bands = trimmed_bands
    path = []
    for band in bands:
        if band[2] not in path:
            path += [band[2]]
    path = ksort(path)
    features = []
    r = math.pi / 180
    trimmed_bands = []
    for i in range(len(structures)):
        feature = []
        structure = structures[i]
        band = bands[i]
        feature += [structure['lattice']['a'    ],     structure['lattice']['b'   ],     structure['lattice']['c'    ],
                    structure['lattice']['alpha'] * r, structure['lattice']['beta'] * r, structure['lattice']['gamma'] * r]
        for site in structure['sites']:
            feature += site['abc']
            feature += [table[site['label']] / 100]
        feature += [0] * 4 * (sites - len(structure['sites']))
        keys = list(hubbards.keys())
        if structure['id'] in keys:
            for atom in structure['atoms']:
                feature += [hubbards[structure['id']][atom]]
            feature += [0] * (kinds - len(structure['atoms']))
        else:
            feature += [0] * kinds
        feature += [path.index(band[2]) + 1]
        features += [feature]
        trimmed_band = []
        for ban in band[3]:
            trimmed_band += ban
        trimmed_bands += [trimmed_band]
    with open('saved_discarded', 'w') as saved_discarded:
        json.dump(discarded, saved_discarded)
    with open('saved_features', 'w') as saved_features:
        json.dump(features, saved_features)
    with open('saved_bands', 'w') as saved_bands:
        json.dump(trimmed_bands, saved_bands)
    print(len(discarded), len(features))


def splitter(train, valid, evalu):
    with open('saved_features', 'r') as saved_features:
        features = json.load(saved_features)
    with open('saved_bands', 'r') as saved_bands:
        bands = json.load(saved_bands)
    index = [i for i in range(len(features))]
    train_choice = choice(index, size = train, replace = False)
    new_index = []
    for i in index:
        if i not in train_choice:
            new_index += [i]
    index = new_index
    valid_choice = choice(index, size = valid, replace = False)
    new_index = []
    for i in index:
        if i not in valid_choice:
            new_index += [i]
    index = new_index
    evalu_choice = choice(index, size = evalu, replace = False)
    train_features = []
    train_bands = []
    for i in train_choice:
        train_features += [features[i]]
        train_bands += [bands[i]]
    valid_features = []
    valid_bands = []
    for i in valid_choice:
        valid_features += [features[i]]
        valid_bands += [bands[i]]
    evalu_features = []
    evalu_bands = []
    for i in evalu_choice:
        evalu_features += [features[i]]
        evalu_bands += [bands[i]]
    with open('saved_train_features', 'w') as saved_train_features:
        json.dump(train_features, saved_train_features)
    with open('saved_train_bands', 'w') as saved_train_bands:
        json.dump(train_bands, saved_train_bands)
    with open('saved_valid_features', 'w') as saved_valid_features:
        json.dump(valid_features, saved_valid_features)
    with open('saved_valid_bands', 'w') as saved_valid_bands:
        json.dump(valid_bands, saved_valid_bands)
    with open('saved_evalu_features', 'w') as saved_evalu_features:
        json.dump(evalu_features, saved_evalu_features)
    with open('saved_evalu_bands', 'w') as saved_evalu_bands:
        json.dump(evalu_bands, saved_evalu_bands)
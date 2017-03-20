# 1. no spin, bands >= 20
# 2. no path
# 3. linear: 100 * 20
# 4. 6 + 4 * 36 = 150

import json, math
from pymatgen import MPRester
from numpy.random import choice
from scipy.interpolate import interp1d as interpolate

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

def sort(l):
    sorted = []
    l_ = l.copy()
    while len(l_) > 0:
        s = l_[0]
        for i in l_[1:]:
            if i < s:
                s = i
        sorted += [s]
        l_.remove(s)
    return sorted

def list_sort(l):
    sorted = []
    l_ = l.copy()
    while len(l_) > 0:
        s = l_[0]
        sl = len(l_[0])
        for i in l_[1:]:
            if len(i) < sl:
                s = i
                sl = len(i)
        sorted += [s]
        l_.remove(s)
    return sorted

def band_creater(nbands, points, method):
    spin_polarized = []
    discarded = []
    for title in titles:
        with open(title, 'r') as saved_bands:
            bands = json.load(saved_bands)
        collated_bands = []
        for band in bands:
            if band['is_spin_polarized'] == False:
                if len(band['bands']['1']) >= nbands and len(band['bands']['1'][0]) >= points:
                    path = []
                    for branch in band['branches']:
                        path += [branch['name']]
                    fermi = band['efermi']
                    if band['is_metal'] == False:
                        upper = band['cbm']['band_index']['1'][0]
                        if upper >= nbands / 2 and len(band['bands']['1']) - (upper + 1) >= nbands / 2:
                            selected_band = [band['bands']['1'][i] for i in range(upper - int(nbands / 2), upper + int(nbands / 2))]
                            lock = 0
                        else:
                            discarded += [band['id']]
                            lock = 1
                    else:
                        positive_means = []
                        negative_means = []
                        for ban in band['bands']['1']:
                            total = 0
                            for ba in ban:
                                total += ba
                            mean = total / len(ban) - fermi
                            if mean >= 0:
                                positive_means += [mean]
                                negative_means += [100]
                            else:
                                negative_means += [abs(mean)]
                                positive_means += [100]
                        p = 0
                        for i in positive_means:
                            if i < 100:
                                p += 1
                        n = 0
                        for i in negative_means:
                            if i < 100:
                                n += 1
                        if p >= nbands / 2 and n >= nbands / 2:
                            sorted_positive_means = sort(positive_means)[:int(nbands / 2)]
                            sorted_negative_means = sort(negative_means)[:int(nbands / 2)]
                            selected_band = []
                            for i in range(int(nbands / 2 - 1), -1, -1):
                                mean = sorted_negative_means[i]
                                selected_band += [band['bands']['1'][negative_means.index(mean)]]
                            for mean in sorted_positive_means:
                                selected_band += [band['bands']['1'][positive_means.index(mean)]]
                            lock = 0
                        else:
                            discarded += [band['id']]
                            lock = 1
                else:
                    discarded += [band['id']]
                    lock = 1
                if lock == 0:
                    n = len(selected_band[0])
                    spacing = 10 / (n - 1)
                    x_axis = [i * spacing for i in range(n)]
                    new_x_axis = [10 / points * i for i in range(points)]
                    totals = []
                    inter_band = []
                    for ban in selected_band:
                        total = 0
                        inter_ban = []
                        inter = interpolate(x_axis, ban, method)
                        for new_x in new_x_axis:
                            ba = inter(new_x) - fermi
                            inter_ban += [ba]
                            total += ba
                        inter_band += [inter_ban]
                        totals += [total]
                    sorted_totals = sort(totals)
                    sorted_inter_band = []
                    for sorted_total in sorted_totals:
                        sorted_inter_band += [inter_band[totals.index(sorted_total)]]
                    ID = band['id']
                    collated_bands += [[ID, path, sorted_inter_band]]
            else:
                spin_polarized += [band['id']]
        with open('saved_collated_bands' + title[11:], 'w') as saved_collated_bands:
            json.dump(collated_bands, saved_collated_bands)
        print(title[12:] + ' is done.')
    collated_bands = []
    for title in titles:
        with open('saved_collated_bands' + title[11:], 'r') as saved_collated_bands:
            collated_band = json.load(saved_collated_bands)
        collated_bands += collated_band
    with open('saved_collated_bands', 'w') as saved_collated_bands:
        json.dump(collated_bands, saved_collated_bands)
    with open('saved_spin_polarized', 'w') as saved_spin_polarized:
        json.dump(spin_polarized, saved_spin_polarized)
    with open('saved_discarded', 'w') as saved_discarded:
        json.dump(discarded, saved_discarded)

def structure_modifier():
    with open('saved_structures', 'r') as saved_structures:
        structures = json.load(saved_structures)
    with open('saved_collated_bands', 'r') as saved_collated_bands:
        bands = json.load(saved_collated_bands)
    SIDs = []
    for structure in structures:
        SIDs += [structure['id']]
    new_structures = []
    for band in bands:
        BID = band[0]
        for structure in structures:
            if BID == structure['id']:
                structure['path'] = band[1]
                new_structures += [structure]
                structures.remove(structure)
                SIDs.remove(BID)
                break
        else:
            with MPRester('tRKiauSUW6DpO19n') as REST:
                structure = REST.get_structure_by_material_id(BID).as_dict()
            del structure['@class'], structure['@module']
            structure['id'] = BID
            structure['path'] = band[1]
            new_structures += [structure]
            print(str(BID) + ' is added.')
        print(len(new_structures), '/', len(bands))
    with open('saved_structures_new', 'w') as saved_structures:
        json.dump(new_structures, saved_structures)

def feature_creater(sites):
    with open('saved_structures', 'r') as saved_structures:
        structures = json.load(saved_structures)
    with open('saved_paths', 'r') as saved_paths:
        paths = json.load(saved_paths)
    r = math.pi / 180
    features = []
    discarded = []
    for structure in structures:
        if (structure['path'] in paths) and (len(structure['sites']) <= sites):
            feature = [structure['lattice']['a'    ],     structure['lattice']['b'   ],     structure['lattice']['c'    ],
                       structure['lattice']['alpha'] * r, structure['lattice']['beta'] * r, structure['lattice']['gamma'] * r]
            feature += [paths.index(structure['path']) + 1]
            for site in structure['sites']:
                feature += site['abc']
                feature += [table[site['label']] / 100]
            feature += [0] * 4 * (sites - len(structure['sites']))
            features += [feature]
        else:
            discarded += [structure['id']]
            print('discarded')
        print(len(features) + len(discarded), '/', len(structures))
    with open('saved_features', 'w') as saved_features:
        json.dump(features, saved_features)


def path_finder(top):
    with open('saved_collated_bands', 'r') as saved_collated_bands:
        bands = json.load(saved_collated_bands)
    paths = []
    for band in bands:
        path = band[1]
        if path not in paths:
            paths += [path]
    counter = [0 for i in range(len(paths))]
    for band in bands:
        index = paths.index(band[1])
        counter[index] += 1
    selected = []
    for i in range(top):
        current = 0
        for count in counter:
            if count > current:
                current = count
        selected += [paths[counter.index(current)]]
        counter[counter.index(current)] = 0
    with open('saved_paths', 'w') as saved_paths:
        json.dump(selected, saved_paths)

def band_trimmer():
    with open('saved_collated_bands', 'r') as saved_collated_bands:
        bands = json.load(saved_collated_bands)
    with open('saved_features', 'r') as saved_features:
        features = json.load(saved_features)
    FIDs = []
    for feature in features:
        FIDs += [feature[0]]
    trimmed = []
    for band in bands:
        if band[0] in FIDs:
            bans = []
            for ban in band[2]:
                bans += ban
            trimmed += [bans]
    with open('saved_bands', 'w') as saved_bands:
        json.dump(trimmed, saved_bands)
    print(len(trimmed[0]))

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

splitter(16000, 1420, 7)

# 17427
# import plotly.plotly as py
# import plotly.graph_objs as go
# with open('saved_collated_bands', 'r') as saved_collated_bands:
#     bands = json.load(saved_collated_bands)
# x = [i * 0.1 for i in range(100)]
# y = bands[10][2]
# trace = []
# for i in y:
#     trace += [go.Scatter(x = x, y = i, line=dict(shape='spline'))]
# figure = dict(data = trace)
# py.iplot(figure, filename='fuck')
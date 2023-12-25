#!/usr/bin/env python 
"""
remove the lr energy, in this script, we need to generate the local frame from topo, predict the tensor with nn 
"""
import numpy as np
import pickle
import torch
from dmff_torch.pairwise import (generate_pairwise_interaction,
                     TT_damping_qq_c6_kernel,
                     TT_damping_qq_kernel,
                     slater_disp_damping_kernel,
                     slater_sr_kernel,
                     distribute_scalar,
                     distribute_multipoles,
                     distribute_v3, 
                     distribute_dispcoeff)
from dmff_torch.nblist import build_covalent_map
from dmff_torch.disp import energy_disp_pme
from dmff_torch.pme import ADMPPmeForce, energy_pme, setup_ewald_parameters
from functools import partial
from dmff_torch.utils import regularize_pairs, pair_buffer_scales
from dmff_torch.spatial import pbc_shift, v_pbc_shift, generate_construct_local_frames
from dmff_torch.multipole import rot_global2local, rot_local2global, C1_c2h, C2_c2h
import ase
from e3_layers.utils import build
from e3_layers import configs
from e3_layers.data import Batch, computeEdgeIndex
from dmff_torch.recip import pme_recip
from e3nn import o3

atomic_num = ase.data.atomic_numbers

def convert_tensor(mom):
    return torch.tensor(np.array(mom),dtype=torch.float32)

def irreps2matrix(qua,device='cuda:0'):
    basis = o3.wigner_3j(1, 1, 2, device=device)
    # multiple 15. or 5. should be check
    result = basis@qua * torch.tensor(15.,dtype=torch.float32)
    result = result.view(9,result.shape[2])[[0,4,8,1,2,5]]
    return result

def get_axis_idx(ii,conn_atom,ele):
    ele = [atomic_num[u] for u in ele]
    z_idx = None; x_idx = None
    nei_0 = conn_atom[ii]
    if len(nei_0) == 1:
        z_idx = nei_0[0]
        nei_1 = conn_atom[z_idx]
        nei_ele = np.array([ele[u] for u in nei_1])
        nei_1 = np.array(nei_1)[np.argsort(-nei_ele)]
        for uu in nei_1:
            if uu != ii and x_idx == None:
                x_idx = uu
    else:
        nei_ele = [ele[u] for u in nei_0]
        z_idx = nei_0[np.argsort(nei_ele)[-1]]
        x_idx = nei_0[np.argsort(nei_ele)[-2]]
    assert(z_idx != None and x_idx !=None)
    return z_idx, x_idx

def check_topo(coord, topo, symbol):
    # check whether topo is reasonable, charge transfer may happen
    state = True; coord = np.array(coord)
    topo_0 = topo[0]; topo_1 = topo[1] + np.max(topo_0) + 1
    for bond in topo_0:
        sym_0 = symbol[bond[0]]; sym_1 = symbol[bond[1]]
        c_0 = coord[bond[0]]; c_1 = coord[bond[1]]
        if sym_0 == 'H' or sym_1 == 'H':
            dis = np.linalg.norm((c_0 - c_1))
            if dis > 2.0:
                state = False            
    for bond in topo_1:
        sym_0 = symbol[bond[0]]; sym_1 = symbol[bond[1]]
        c_0 = coord[bond[0]]; c_1 = coord[bond[1]]
        if sym_0 == 'H' or sym_1 == 'H':
            dis = np.linalg.norm((c_0 - c_1))
            if dis > 2.0:
                state = False
    return state 

def input_infor(topo, mol_num, coord, symbol):
    topo_0 = topo[0]; topo_1 = topo[1]
    topo = np.vstack((np.array(topo_0), np.array(topo_1) + mol_num[0]))
    
    symbol_0 = symbol[0:mol_num[0]]; symbol_1 = symbol[mol_num[0]:]
    axis_types, axis_indices = init_axis(topo, symbol)
    axis_types_0, axis_indices_0 = init_axis(topo_0, symbol_0)
    axis_types_1, axis_indices_1 = init_axis(topo_1, symbol_1)
    coord_0 = coord[0:mol_num[0]]; coord_1 = coord[mol_num[0]:]
    #coord = convert_tensor(coord); coord_0 = convert_tensor(coord_0)
    #coord_1 = convert_tensor(coord_1) 
    axis_types = torch.tensor(axis_types); axis_indices = torch.tensor(axis_indices)
    axis_types_0 = torch.tensor(axis_types_0); axis_indices_0 = torch.tensor(axis_indices_0)
    axis_types_1 = torch.tensor(axis_types_1); axis_indices_1 = torch.tensor(axis_indices_1)
    return coord, coord_0, coord_1, axis_types, axis_types_0, axis_types_1, axis_indices, axis_indices_0, axis_indices_1, \
           topo, topo_0, topo_1

def init_axis(topo,symbol):
    #topo = topo.tolist()
    conn_atom = {}
    for pair in topo:
        conn_atom[pair[0]] = []
    for pair in topo:
        conn_atom[pair[0]].append(pair[1])
    axis_types = []; axis_indices = []; ZThenX = 0; yaxis=-1
    for ii in range(len(symbol)):
        axis_types.append(ZThenX)
        zaxis, xaxis = get_axis_idx(ii, conn_atom, symbol)
        axis_indices.append([zaxis,xaxis,yaxis])
    axis_types = np.array(axis_types); axis_indices = np.array(axis_indices)
    return axis_types,axis_indices

def gen_pair(coord, topo):
    # attention build_covalent_map may not suitable
    data = {'positions':coord, 'bonds':topo}
    cov_map = build_covalent_map(data, 6)
    pairs = []
    for na in range(len(coord)):
        for nb in range(na + 1, len(coord)):
            pairs.append([na, nb, 0])
    pairs = np.array(pairs, dtype=int)
    pairs[:,2] = cov_map[pairs[:,0], pairs[:,1]]
    return torch.tensor(pairs)

def pmepol(box, axis_types, axis_indices, rcut, coord, pairs, q_local, pol, tholes, mscales, pscales, dscales):
    pme_es_pol = ADMPPmeForce(box, axis_types, axis_indices, rcut, 5e-4, 2, lpol=True, lpme=False, steps_pol=5)
    #U_ind = pme_es_pol.U_ind
    e_es_pol = pme_es_pol.get_energy(coord, box, pairs, q_local, pol, tholes, mscales, pscales, dscales, None, False)
    U_ind = pme_es_pol.U_ind
    return e_es_pol, U_ind

def pme(box, coord, pairs, q_local, U_ind, pol, tholes, mscales, pscales, dscales, construct_local_frame):
    e = energy_pme(coord, box, pairs, q_local, U_ind, pol, tholes, mscales, pscales, dscales, None, 0, None, None, None, 2, True, None, None, None, False, lpme=False)
    return e

def load_model(config_name,f_path,device):
    config = config_name.model_config
    model = build(config).to(device)
    state_dict = torch.load(f_path, map_location=device)
    model_state_dict = {}
    for key, value in state_dict.items():
        if key[:7] == 'module.':
            key = key[7:]
        model_state_dict[key] = value
    model.load_state_dict(model_state_dict)
    return model

def load_input(coord, atom_type, n_nodes,r_cutnn,device):
    data = {'pos': coord, 'species': atom_type, '_n_nodes': n_nodes}
    attrs = {'pos': ('node', '1x1o'), 'species': ('node','1x0e')}
    _data, _attrs = computeEdgeIndex(data, attrs, r_max=r_cutnn)
    data.update(_data)
    attrs.update(_attrs)
    input_batch = Batch(attrs, **data).to(device)
    return input_batch

def convert_e3nn(q,dip,qua,pol,c6,c8,c10,device_0):
    dip = torch.matmul(C1_c2h,dip.T).T
    qua = irreps2matrix(qua.T,device=device_0)
    qua = torch.matmul(C2_c2h,qua).T
    pol = pol.squeeze()
    q_global = torch.hstack((q, dip, qua))
    c_list = torch.hstack((c6, c8, c10))
    return q_global, pol, c_list

# shift along the center of mass direction
def find_closest_distance(coord_A, coord_B):
    coord_A = np.array(coord_A); coord_B = np.array(coord_B)
    n_atoms1 = len(coord_A); n_atoms2 = len(coord_B)
    min_i = -1; min_j = -1
    min_dr = 10000
    for i in range(n_atoms1):
        r1 = coord_A[i]
        for j in range(n_atoms2):
            r2 = coord_B[j]
            if np.linalg.norm(r1-r2) < min_dr:
                min_dr = np.linalg.norm(r1-r2)
                min_i = i
                min_j = n_atoms1 + j
    return min_i, min_j, min_dr

# sqrt the c6, c8, and c10
def sqrt_monopole(disp_coeff):
    disp_coeff = torch.sqrt(torch.clamp(disp_coeff, min=0.))
    return disp_coeff

if __name__ == '__main__':
    # read the topo, coord and energy components
    data = np.load('./saptdata/sapt.npz',allow_pickle=True)
    
    coords_A = data['coord_A']; symbols_A = data['symbol_A']
    coords_B = data['coord_B']; symbols_B = data['symbol_B']
    topos = data['topo']; topos_A = data['topo_A']; topos_B = data['topo_B']
    box = torch.tensor([[50., 0., 0.], [0.,50.,0.],[0.,0.,50.]], dtype=torch.float32, requires_grad=False)

    # we need es; pol; disp; ex; dhf; tot 
    E_es_sapt = data['E1pol']; E_ind_sapt = data['E2ind']; E_disp_sapt = data['E2disp'] 
    confs = data['conf']; Dis = []

    # then, predict the tensor 
    # first load the nn model, we may use multi-task model latter
    device = 'cpu'; r_cutnn = 5.; rcut = 10.
    model_q = load_model(configs.config_monopole(), './e3nnmodel/q.pt', device)
    model_dip = load_model(configs.config_dipole(), './e3nnmodel/dipole.pt', device)
    model_qua = load_model(configs.config_quadrupole(), './e3nnmodel/qua.pt', device)
    model_pol = load_model(configs.config_monopole(), './e3nnmodel/pol.pt', device)
    model_c6 = load_model(configs.config_monopole(), './e3nnmodel/c6.pt', device)
    model_c8 = load_model(configs.config_monopole(), './e3nnmodel/c8.pt', device)
    model_c10 = load_model(configs.config_monopole(), './e3nnmodel/c10.pt', device)
    
    ###############################################################################################################
    # then predict the tensor, calc the lr, obtain the sr energy, lr include E_es, E_pol and E_disp 
    keys = ['es', 'pol', 'disp', 'conf_0', 'conf_1', 'conf_2', 'dis', 'es_l', 'pol_l', 'disp_l']; npts = len(E_es_sapt)
    #scan_res = {}; 
    scan_res_lr = {}
    # only es, pol, disp and tot need deduct 
    for key in keys:
        #scan_res[key] = np.zeros(npts)
        scan_res_lr[key] = np.zeros(npts)

    #scan_res['es'] = scan_res['es'] + E_es_sapt    
    #scan_res['pol'] = scan_res['pol'] + E_ind_sapt
    #scan_res['disp'] = scan_res['disp'] + E_disp_sapt
    #scan_res['ex'] = scan_res['ex'] + E_ex_sapt
    #scan_res['dhf'] = scan_res['dhf'] + E_dhf_sapt
    #scan_res['tot'] = scan_res['tot'] + E_tot_sapt
    #################################################################################################################

    #for ipt in range(100):
    #import time
    #time0 = time.time()
    for ipt in range(10):
    #for ipt in range(npts):
        #print(time.time() - time0)
        coord = np.concatenate((coords_A[ipt], coords_B[ipt]))
        symbol = np.concatenate((symbols_A[ipt], symbols_B[ipt]))
        topo = [np.array(topos_A[ipt], dtype=np.int32), np.array(topos_B[ipt],dtype=np.int32)]
        mol_num = [len(coords_A[ipt]), len(coords_B[ipt])]
        coord = torch.tensor(coord, dtype=torch.float32)
        species = torch.tensor([atomic_num[u] for u in symbol],dtype=torch.long)
        
        topo_stat = check_topo(coord, topo, symbol)
        _, _, min_dis = find_closest_distance(coord[0:mol_num[0]], coord[mol_num[0]:])
        if min_dis > 0.:     
            if topo_stat == False:
                print('attention %s' %(ipt))  
            Dis.append(min_dis)
            coord_A = coord[0:mol_num[0]]; coord_B = coord[mol_num[0]:]
            species_A = species[0:mol_num[0]]; species_B = species[mol_num[0]:]
            n_nodes_A = torch.ones((1, 1), dtype=torch.long)* len(coord_A)
            n_nodes_B = torch.ones((1, 1), dtype=torch.long)* len(coord_B)
            
            input_A = load_input(coord_A, species_A, n_nodes_A, r_cutnn, device)
            input_B = load_input(coord_B, species_B, n_nodes_B, r_cutnn, device)
            
            q_0 = model_q(input_A)['monopole']; dip_0 = model_dip(input_A)['dipole']
            qua_0 = model_qua(input_A)['quadrupole_2']; pol_0 = model_pol(input_A)['monopole']
            q_1 = model_q(input_B)['monopole']; dip_1 = model_dip(input_B)['dipole']
            qua_1 = model_qua(input_B)['quadrupole_2']; pol_1 = model_pol(input_B)['monopole']

            c6_0 = model_c6(input_A)['monopole']; c8_0 = model_c8(input_A)['monopole']
            c10_0 = model_c10(input_A)['monopole']
            c6_1 = model_c6(input_B)['monopole']; c8_1 = model_c8(input_B)['monopole']
            c10_1 = model_c10(input_B)['monopole']

            # if we use the nn model trained from c6 rather than sqrt(c6), we need use the sqrt value
            c6_0 = sqrt_monopole(c6_0); c8_0 = sqrt_monopole(c8_0); c10_0 = sqrt_monopole(c10_0)
            c6_1 = sqrt_monopole(c6_1); c8_1 = sqrt_monopole(c8_1); c10_1 = sqrt_monopole(c10_1)

            q_global_0, pol_0, c_list_0 = convert_e3nn(q_0, dip_0, qua_0, pol_0, c6_0, c8_0, c10_0, device)
            q_global_1, pol_1, c_list_1 = convert_e3nn(q_1, dip_1, qua_1, pol_1, c6_1, c8_1, c10_1, device)
            q_global = torch.vstack((q_global_0,q_global_1))
            pol = torch.hstack((pol_0, pol_1))
            c_list = torch.vstack((c_list_0, c_list_1))
            
            coord, coord_0, coord_1, axis_types, axis_types_0, axis_types_1, axis_indices, axis_indices_0, axis_indices_1, topo_t, topo_0, topo_1 = \
                input_infor(topo, mol_num, coord, symbol)
            
            #construct_local_frame_fn = generate_construct_local_frames(axis_types, axis_indices)
            #construct_local_frame_fn_0 = generate_construct_local_frames(axis_types_0, axis_indices_0)
            #construct_local_frame_fn_1 = generate_construct_local_frames(axis_types_1, axis_indices_1) 

            #localframe = construct_local_frame_fn(coord, box)
            #localframe_0 = construct_local_frame_fn_0(coord_0, box)
            #localframe_1 = construct_local_frame_fn_1(coord_1, box)

            # get the local properties
            Q_local = q_global
            Q_local_0 = Q_local[0:len(q_0)] 
            Q_local_1 = Q_local[len(q_0):]

            # get the pairs, tholes, mscales, pscales, dscales
            _tholes = [0.33] * len(coord)
            _tholes_0 = _tholes[0:len(q_0)]; _tholes_1 = _tholes[len(q_0):]
            _tholes = convert_tensor(_tholes); _tholes_0 = convert_tensor(_tholes_0); _tholes_1 = convert_tensor(_tholes_1)
            mScales = torch.tensor([0.,0.,0.,0.,0.,1.])
            pScales = torch.tensor([0.,0.,0.,0.,0.,1.])
            dScales = torch.tensor([0.,0.,0.,0.,0.,1.])

            # gen pairs 
            pairs = gen_pair(coord, topo_t)
            pairs_0 = gen_pair(coord_0, topo_0)
            pairs_1 = gen_pair(coord_1, topo_1)

            #################################
            # electrostatic + pol
            #################################
            e_es_pol_AB, U_ind_AB = pmepol(box, axis_types, None, rcut, coord, pairs, Q_local, pol, _tholes, mScales, pScales, dScales)
            e_es_pol_A, U_ind_A = pmepol(box, axis_types_0, None, rcut, coord_0, pairs_0, Q_local_0, pol_0, _tholes_0, mScales, pScales, dScales)
            e_es_pol_B, U_ind_B = pmepol(box, axis_types_1, None, rcut, coord_1, pairs_1, Q_local_1, pol_1, _tholes_1, mScales, pScales, dScales)
            E_espol = e_es_pol_AB - e_es_pol_A - e_es_pol_B

            #################################
            # polarization (induction) energy
            #################################
            U_ind_AB_mono = torch.vstack((U_ind_A, U_ind_B))
            e_AB_nonpol = pme(box, coord, pairs, Q_local, U_ind_AB_mono, pol, _tholes, mScales, pScales, dScales, None)
            e_A_nonpol = pme(box, coord_0, pairs_0, Q_local_0, U_ind_A, pol_0, _tholes_0, mScales, pScales, dScales, None)
            e_B_nonpol = pme(box, coord_1, pairs_1, Q_local_1, U_ind_B, pol_1, _tholes_1, mScales, pScales, dScales, None)
            E_es = e_AB_nonpol - e_A_nonpol - e_B_nonpol
            E_pol = E_espol - E_es
            #################################
            # dispersion energy 
            #################################
            e_disp_AB = energy_disp_pme(coord, box, pairs, c_list, mScales, None, None, None, None, 10, None, None, None, None, None, None, None, None, False, lpme=False)
            e_disp_A = energy_disp_pme(coord_0, box, pairs_0, c_list_0, mScales, None, None, None, None, 10, None, None, None, None, None, None, None, None, False, lpme=False)
            e_disp_B = energy_disp_pme(coord_1, box, pairs_1, c_list_1, mScales, None, None, None, None, 10, None, None, None, None, None, None, None, None, False, lpme=False)
            E_disp = e_disp_AB - e_disp_A - e_disp_B

            # attention to the unit, all the data in sapt are in kcal/mol, disp in dmff is hartree, other in dmff is kj/mol
            E_es = E_es / 4.184; E_pol = E_pol / 4.184; E_disp = - E_disp * 627.509608 
            conf = confs[ipt].split('_')
            scan_res_lr['es'][ipt] = E_es
            scan_res_lr['pol'][ipt] = E_pol
            scan_res_lr['disp'][ipt] = E_disp
            scan_res_lr['conf_0'][ipt] = conf[0]
            scan_res_lr['conf_1'][ipt] = conf[1]
            scan_res_lr['conf_2'][ipt] = conf[2]
            scan_res_lr['dis'][ipt] = min_dis
            scan_res_lr['es_l'][ipt] = E_es_sapt[ipt]
            scan_res_lr['pol_l'][ipt] = E_ind_sapt[ipt]
            scan_res_lr['disp_l'][ipt] = E_disp_sapt[ipt]

np.savez_compressed('camcasp_nn_sapt.npz', **scan_res_lr)
    # then save the data if we want 
    #with open('data_sr.pickle','wb') as ofile:
    #    pickle.dump(scan_res, ofile)

    #with open('data_lr.pickle','wb') as ofile:
    #    pickle.dump(scan_res_lr, ofile)

    # save the idx with large dis 
    #np.save('largeidx.npy', large_dis)

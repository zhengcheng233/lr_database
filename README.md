# lr_database
Author: Zheng Cheng (chengz@bjaisi.com)

This repository includes e3nn models for asymptotic parameters (in the e3nnmodel directory) and sapt data of dimers with the combination of different protein fragments (in the saptdata directory). You can run calc_e_component_camp.py and calc_e_component_hir.py to calculate long-range interactions and compare them with the sapt components. These calculations include the intermolecular distance for each dimer pair, as recorded in the npz file. We only need to compare the long-range energy calculated by our force fields and SAPT with the dimers whose distance are over 3.5 angstrom. If you want to run these two scripts, e3_layer(https://github.com/zhengcheng233/e3_layer.git) and dmff_torch(https://github.com/zhengcheng233/dmff_torch.git) should be first installed. 




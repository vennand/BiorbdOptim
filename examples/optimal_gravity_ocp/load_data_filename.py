def load_data_filename(subject, trial):
    if subject == 'DoCi':
        model_name = 'DoCi.s2mMod'
        # model_name = 'DoCi_SystemesDaxesGlobal_surBassin_rotAndre.s2mMod'
        if trial == '822':
            c3d_name = 'Do_822_contact_2.c3d'
            q_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(3089, 3360)
        elif trial == '44_1':
            c3d_name = 'Do_44_mvtPrep_1.c3d'
            q_name = 'Do_44_mvtPrep_1_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_44_mvtPrep_1_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_44_mvtPrep_1_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(2449, 2700)
        elif trial == '44_2':
            c3d_name = 'Do_44_mvtPrep_2.c3d'
            q_name = 'Do_44_mvtPrep_2_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_44_mvtPrep_2_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_44_mvtPrep_2_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(2599, 2850)
        elif trial == '44_3':
            c3d_name = 'Do_44_mvtPrep_3.c3d'
            q_name = 'Do_44_mvtPrep_3_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_44_mvtPrep_3_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_44_mvtPrep_3_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(4099, 4350)
    elif subject == 'JeCh':
        model_name = 'JeCh_201.s2mMod'
        # model_name = 'JeCh_SystemeDaxesGlobal_surBassin'
        if trial == '833_1':
            c3d_name = 'Je_833_1.c3d'
            q_name = 'Je_833_1_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_1_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_1_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(1919, 2220)
            frames = range(2299, 2590)
        if trial == '833_2':
            c3d_name = 'Je_833_2.c3d'
            q_name = 'Je_833_2_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_2_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_2_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(1899, 2210)
            frames = range(2289, 2590)
        if trial == '833_3':
            c3d_name = 'Je_833_3.c3d'
            q_name = 'Je_833_3_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_3_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_3_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(2179, 2490)
            frames = range(2569, 2880)
        if trial == '833_4':
            c3d_name = 'Je_833_4.c3d'
            q_name = 'Je_833_4_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_4_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_4_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(2269, 2590)
            frames = range(2669, 2970)
        if trial == '833_5':
            c3d_name = 'Je_833_5.c3d'
            q_name = 'Je_833_5_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_5_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_5_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(2279, 2600)
            frames = range(2669, 2980)
    elif subject == 'BeLa':
        model_name = 'BeLa.s2mMod'
        # model_name = 'BeLa_SystemeDaxesGlobal_surBassin.s2mMod'
        if trial == '44_1':
            c3d_name = 'Ben_44_mvtPrep_1.c3d'
            q_name = 'Ben_44_mvtPrep_1_MOD202.00_GenderM_BeLag_Q.mat'
            qd_name = 'Ben_44_mvtPrep_1_MOD202.00_GenderM_BeLag_V.mat'
            qdd_name = 'Ben_44_mvtPrep_1_MOD202.00_GenderM_BeLag_A.mat'
            frames = range(1799, 2050)
        elif trial == '44_2':
            c3d_name = 'Ben_44_mvtPrep_2.c3d'
            q_name = 'Ben_44_mvtPrep_2_MOD202.00_GenderM_BeLag_Q.mat'
            qd_name = 'Ben_44_mvtPrep_2_MOD202.00_GenderM_BeLag_V.mat'
            qdd_name = 'Ben_44_mvtPrep_2_MOD202.00_GenderM_BeLag_A.mat'
            frames = range(2149, 2350)
        elif trial == '44_3':
            c3d_name = 'Ben_44_mvtPrep_3.c3d'
            q_name = 'Ben_44_mvtPrep_3_MOD202.00_GenderM_BeLag_Q.mat'
            qd_name = 'Ben_44_mvtPrep_3_MOD202.00_GenderM_BeLag_V.mat'
            qdd_name = 'Ben_44_mvtPrep_3_MOD202.00_GenderM_BeLag_A.mat'
            frames = range(2449, 2700)
    elif subject == 'GuSe':
        model_name = 'GuSe.s2mMod'
        # model_name = 'GuSe_SystemeDaxesGlobal_surBassin.s2mMod'
        if trial == '44_2':
            c3d_name = 'Gui_44_mvt_Prep_2.c3d'
            q_name = 'Gui_44_mvt_Prep_2_MOD200.00_GenderM_GuSeg_Q.mat'
            qd_name = 'Gui_44_mvt_Prep_2_MOD200.00_GenderM_GuSeg_V.mat'
            qdd_name = 'Gui_44_mvt_Prep_2_MOD200.00_GenderM_GuSeg_A.mat'
            frames = range(1649, 1850)
        elif trial == '44_3':
            c3d_name = 'Gui_44_mvt_Prep_3.c3d'
            q_name = 'Gui_44_mvt_Prep_3_MOD200.00_GenderM_GuSeg_Q.mat'
            qd_name = 'Gui_44_mvt_Prep_3_MOD200.00_GenderM_GuSeg_V.mat'
            qdd_name = 'Gui_44_mvt_Prep_3_MOD200.00_GenderM_GuSeg_A.mat'
            frames = range(1699, 1950)
        elif trial == '44_4':
            c3d_name = 'Gui_44_mvtPrep_4.c3d'
            q_name = 'Gui_44_mvtPrep_4_MOD200.00_GenderM_GuSeg_Q.mat'
            qd_name = 'Gui_44_mvtPrep_4_MOD200.00_GenderM_GuSeg_V.mat'
            qdd_name = 'Gui_44_mvtPrep_4_MOD200.00_GenderM_GuSeg_A.mat'
            frames = range(1599, 1850)
    elif subject == 'SaMi':
        model_name = 'SaMi.s2mMod'
        if trial == '821_822_2':
            c3d_name = 'Sa_821_822_2.c3d'
            q_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_A.mat'
            # frames = range(2909, 3220)
            frames = range(3279, 3600)
            # frames = range(3659, 3950)
        elif trial == '821_822_3':
            c3d_name = 'Sa_821_822_3.c3d'
            q_name = 'Sa_821_822_3_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_822_3_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_822_3_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3139, 3440)
        elif trial == '821_822_4':
            c3d_name = 'Sa_821_822_4.c3d'
            q_name = 'Sa_821_822_4_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_822_4_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_822_4_MOD200.00_GenderF_SaMig_A.mat'
            # frames = range(3509, 3820)
            frames = range(3909, 4190)
        elif trial == '821_822_5':
            c3d_name = 'Sa_821_822_5.c3d'
            q_name = 'Sa_821_822_5_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_822_5_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_822_5_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3339, 3630)
        elif trial == '821_contact_1':
            c3d_name = 'Sa_821_contact_1.c3d'
            q_name = 'Sa_821_contact_1_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_contact_1_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_contact_1_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3019, 3330)
        elif trial == '821_contact_2':
            c3d_name = 'Sa_821_contact_2.c3d'
            q_name = 'Sa_821_contact_2_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_contact_2_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_contact_2_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3569, 3880)
        elif trial == '821_contact_3':
            c3d_name = 'Sa_821_contact_3.c3d'
            q_name = 'Sa_821_contact_3_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_contact_3_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_contact_3_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3309, 3620)
        elif trial == '822_contact_1':
            c3d_name = 'Sa_822_contact_1.c3d'
            q_name = 'Sa_822_contact_1_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_822_contact_1_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_822_contact_1_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(5009, 5310)
        elif trial == '821_seul_1':
            c3d_name = 'Sa_821_seul_1.c3d'
            q_name = 'Sa_821_seul_1_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_1_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_1_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3349, 3650)
        elif trial == '821_seul_2':
            c3d_name = 'Sa_821_seul_2.c3d'
            q_name = 'Sa_821_seul_2_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_2_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_2_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3429, 3740)
        elif trial == '821_seul_3':
            c3d_name = 'Sa_821_seul_3.c3d'
            q_name = 'Sa_821_seul_3_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_3_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_3_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3209, 3520)
        elif trial == '821_seul_4':
            c3d_name = 'Sa_821_seul_4.c3d'
            q_name = 'Sa_821_seul_4_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_4_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_4_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3309, 3620)
        elif trial == '821_seul_5':
            c3d_name = 'Sa_821_seul_5.c3d'
            q_name = 'Sa_821_seul_5_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_5_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_5_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(2689, 3000)
        elif trial == 'bras_volant_1':
            c3d_name = 'Sa_bras_volant_1.c3d'
            q_name = 'Sa_bras_volant_1_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_bras_volant_1_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_bras_volant_1_MOD200.00_GenderF_SaMig_A.mat'
            # frames = range(0, 4657)
            # frames = range(649, 3950)
            # frames = range(649, 1150)
            # frames = range(1249, 1950)
            # frames = range(2549, 3100)
            frames = range(3349, 3950)
        elif trial == 'bras_volant_2':
            c3d_name = 'Sa_bras_volant_2.c3d'
            q_name = 'Sa_bras_volant_2_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_bras_volant_2_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_bras_volant_2_MOD200.00_GenderF_SaMig_A.mat'
            # frames = range(0, 3907)
            # frames = range(0, 3100)
            # frames = range(49, 849)
            # frames = range(1599, 2200)
            frames = range(2249, 3100)
    else:
        raise Exception(subject + ' is not a valid subject')

    data_filename = {
        'model': model_name,
        'c3d': c3d_name,
        'q': q_name,
        'qd': qd_name,
        'qdd': qdd_name,
        'frames': frames,
    }

    return data_filename
#bismillah-bisa
# BISMILLAH - TERUJI COCOK

from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import skfuzzy as fuzz

app = FastAPI()


class InputData(BaseModel):
    usia: float
    bb: float
    tb: float
    jk: str


def countBB_U(usia, bb, jk):
    # definiskan domain atau rentang nilai variabel
    # VAR INPUT
    x_usia = np.arange(0, 60, 1)
    x_bb = np.arange(2, 28, 1)
    if jk == "L":
        x_tb = np.arange(44, 130, 1)
    else:
        x_tb = np.arange(43, 130, 1)

    # VAR OUTPUT
    if jk == "L":
        x_bb_u = np.arange(3, 30, 1)
        x_tb_u = np.arange(45, 130, 1)
        x_bb_tb = np.arange(3, 40, 1)
    else:
        x_bb_u = np.arange(4, 30, 1)
        x_tb_u = np.arange(44, 130, 1)
        x_bb_tb = np.arange(3, 38, 1)

    # membership / keanggotaan USIA
    usia_tahap1 = fuzz.trapmf(x_usia, [0, 0, 6, 12])
    usia_tahap2 = fuzz.trimf(x_usia, [6, 12, 24])
    usia_tahap3 = fuzz.trimf(x_usia, [12, 24, 36])
    usia_tahap4 = fuzz.trimf(x_usia, [24, 36, 48])
    usia_tahap5 = fuzz.trapmf(x_usia, [36, 48, 60, 60])

    # membership / keanggotaan BERAT BADAN
    bb_sangat_kurang = fuzz.trapmf(x_bb, [2, 2, 4, 6])
    if jk == "L":
        bb_kurang = fuzz.trimf(x_bb, [4, 8, 12])
        bb_normal = fuzz.trimf(x_bb, [8, 13, 18])
        bb_lebih = fuzz.trimf(x_bb, [15, 18, 23])
        bb_sangat_lebih = fuzz.trapmf(x_bb, [21, 23, 28, 28])
    else:
        bb_kurang = fuzz.trimf(x_bb, [4, 8, 11])
        bb_normal = fuzz.trimf(x_bb, [9, 13, 18])
        bb_lebih = fuzz.trimf(x_bb, [15, 20, 24])
        bb_sangat_lebih = fuzz.trapmf(x_bb, [22, 25, 28, 28])

    # membership / keanggotaan BB-USIA
    if jk == "L":
        bb_u_sangat_kurang = fuzz.trapmf(x_bb_u, [3, 3, 6, 8])
        bb_u_kurang = fuzz.trimf(x_bb_u, [6, 8, 10])
        bb_u_normal = fuzz.trimf(x_bb_u, [8, 12, 18])
        bb_u_resiko_bb_lebih = fuzz.trapmf(x_bb_u, [16, 24, 30, 30])
    else:
        bb_u_sangat_kurang = fuzz.trapmf(x_bb_u, [4, 4, 7, 9])
        bb_u_kurang = fuzz.trimf(x_bb_u, [7, 9, 11])
        bb_u_normal = fuzz.trimf(x_bb_u, [9, 13, 18])
        bb_u_resiko_bb_lebih = fuzz.trapmf(x_bb_u, [16, 23, 30, 30])
        
    u_tahap1 = fuzz.interp_membership(x_usia, usia_tahap1, usia)
    u_tahap2 = fuzz.interp_membership(x_usia, usia_tahap2, usia)
    u_tahap3 = fuzz.interp_membership(x_usia, usia_tahap3, usia)
    u_tahap4 = fuzz.interp_membership(x_usia, usia_tahap4, usia)
    u_tahap5 = fuzz.interp_membership(x_usia, usia_tahap5, usia)

    bb_sk = fuzz.interp_membership(x_bb, bb_sangat_kurang, bb)
    bb_k = fuzz.interp_membership(x_bb, bb_kurang, bb)
    bb_n = fuzz.interp_membership(x_bb, bb_normal, bb)
    bb_l = fuzz.interp_membership(x_bb, bb_lebih, bb)
    bb_sl = fuzz.interp_membership(x_bb, bb_sangat_lebih, bb)

    print(u_tahap1, u_tahap2, u_tahap3, u_tahap4, u_tahap5)  # 5 FASE
    print(bb_sk, bb_k, bb_n, bb_l, bb_sl)

    drjt_usia = u_tahap1, u_tahap2, u_tahap3, u_tahap4, u_tahap5  # 5 FASE
    drjt_bb = bb_sk, bb_k, bb_n, bb_l, bb_sl
    
    # Rule      => usia, bb = output aturan     # Rule      => usia, bb = output aturan

    # USIA 5 FASE
    rule1 = np.fmin(drjt_usia[0], drjt_bb[0])  # Rule 1 => tahap1, sangat_kurang        = kurang
    rule2 = np.fmin(drjt_usia[0], drjt_bb[1])  # Rule 2 => tahap1, kurang               = normal
    rule3 = np.fmin(drjt_usia[0], drjt_bb[2])  # Rule 3 => tahap1, normal               = resiko_bb_lebih
    rule4 = np.fmin(drjt_usia[0], drjt_bb[3])  # Rule 4 => tahap1, lebih                = resiko_bb_lebih
    rule5 = np.fmin(drjt_usia[0], drjt_bb[4])  # Rule 5 => tahap1, sangat_lebih         = resiko_bb_lebih

    # USIA 5 FASE
    rule6 = np.fmin(drjt_usia[1], drjt_bb[0])  # Rule 6 => tahap2, sangat_kurang        = kurang
    rule7 = np.fmin(drjt_usia[1], drjt_bb[1])  # Rule 7 => tahap2, kurang               = normal
    rule8 = np.fmin(drjt_usia[1], drjt_bb[2])  # Rule 8 => tahap2, normal               = normal
    rule9 = np.fmin(drjt_usia[1], drjt_bb[3])  # Rule 9 => tahap2, lebih                = resiko_bb_lebih
    rule10 = np.fmin(drjt_usia[1], drjt_bb[4])  # Rule 10 => tahap2, sangat_lebih       = resiko_bb_lebih

    # USIA 5 FASE
    rule11 = np.fmin(drjt_usia[2], drjt_bb[0])  # Rule 11 => tahap3, sangat_kurang      = sangat_kurang
    rule12 = np.fmin(drjt_usia[2], drjt_bb[1])  # Rule 12 => tahap3, kurang             = kurang
    rule13 = np.fmin(drjt_usia[2], drjt_bb[2])  # Rule 13 => tahap3, normal             = normal
    rule14 = np.fmin(drjt_usia[2], drjt_bb[3])  # Rule 14 => tahap3, lebih              = resiko_bb_lebih
    rule15 = np.fmin(drjt_usia[2], drjt_bb[4])  # Rule 15 => tahap3, sangat_lebih       = resiko_bb_lebih

    # USIA 5 FASE
    rule16 = np.fmin(drjt_usia[3], drjt_bb[0])  # Rule 16 => tahap4, sangat_kurang      = sangat_kurang
    rule17 = np.fmin(drjt_usia[3], drjt_bb[1])  # Rule 17 => tahap4, kurang             = kurang
    rule18 = np.fmin(drjt_usia[3], drjt_bb[2])  # Rule 18 => tahap4, normal             = normal
    rule19 = np.fmin(drjt_usia[3], drjt_bb[3])  # Rule 19 => tahap4, lebih              = resiko_bb_lebih
    rule20 = np.fmin(drjt_usia[3], drjt_bb[4])  # Rule 20 => tahap4, sangat_lebih       = resiko_bb_lebih

    # USIA 5 FASE
    rule21 = np.fmin(drjt_usia[4], drjt_bb[0])  # Rule 21 => tahap5, sangat_kurang      = sangat_kurang
    rule22 = np.fmin(drjt_usia[4], drjt_bb[1])  # Rule 22 => tahap5, kurang             = sangat_kurang
    rule23 = np.fmin(drjt_usia[4], drjt_bb[2])  # Rule 23 => tahap5, normal             = kurang
    rule24 = np.fmin(drjt_usia[4], drjt_bb[3])  # Rule 24 => tahap5, lebih              = normal
    rule25 = np.fmin(drjt_usia[4], drjt_bb[4])  # Rule 25 => tahap5, sangat_lebih       = resiko_bb_lebih

    # USIA 5 FASE
    pk_bb_u_sangat_kurang = np.fmax(rule11, np.fmax(rule16, np.fmax(rule21, rule22)))
    pk_bb_u_kurang = np.fmax(rule1, np.fmax(rule6, np.fmax(rule12, np.fmax(rule17, rule23))))
    pk_bb_u_normal = np.fmax(rule2, np.fmax(rule7, np.fmax(rule8, np.fmax(rule13, np.fmax(rule18, rule24)))))
    pk_bb_u_resiko_bb_lebih = np.fmax(rule3,
                                      np.fmax(rule4, np.fmax(rule5, np.fmax(rule9, np.fmax(rule10, np.fmax(rule14,
                                                                                                           np.fmax(
                                                                                                               rule15,
                                                                                                               np.fmax(
                                                                                                                   rule19,
                                                                                                                   np.fmax(
                                                                                                                       rule20,
                                                                                                                       rule25)))))))))

    print(pk_bb_u_sangat_kurang)
    print(pk_bb_u_kurang)
    print(pk_bb_u_normal)
    print(pk_bb_u_resiko_bb_lebih)

    pk_bb_u_sangat_kurang = np.fmin(pk_bb_u_sangat_kurang, bb_u_sangat_kurang)
    pk_bb_u_kurang = np.fmin(pk_bb_u_kurang, bb_u_kurang)
    pk_bb_u_normal = np.fmin(pk_bb_u_normal, bb_u_normal)
    pk_bb_u_resiko_bb_lebih = np.fmin(pk_bb_u_resiko_bb_lebih, bb_u_resiko_bb_lebih)

    pk_bb_u_0 = np.zeros_like(x_bb_u)
    pk_bb_u_sk = np.zeros_like(bb_u_sangat_kurang)
    pk_bb_u_k = np.zeros_like(bb_u_kurang)
    pk_bb_u_n = np.zeros_like(bb_u_normal)
    pk_bb_u_rbbl = np.zeros_like(bb_u_resiko_bb_lebih)
    
    komposisi = np.fmax(pk_bb_u_sangat_kurang,
                        np.fmax(pk_bb_u_kurang, np.fmax(pk_bb_u_normal, pk_bb_u_resiko_bb_lebih)))

    berat_badan_per_usia = fuzz.defuzz(x_bb_u, komposisi, 'centroid')

    # Calculate membership values for the defuzzified result
    bb_u_sangat_kurang_degree = fuzz.interp_membership(x_bb_u, bb_u_sangat_kurang, berat_badan_per_usia)
    bb_u_kurang_degree = fuzz.interp_membership(x_bb_u, bb_u_kurang, berat_badan_per_usia)
    bb_u_normal_degree = fuzz.interp_membership(x_bb_u, bb_u_normal, berat_badan_per_usia)
    bb_u_resiko_bb_lebih_degree = fuzz.interp_membership(x_bb_u, bb_u_resiko_bb_lebih, berat_badan_per_usia)

    status_gizi = 'Tidak Terdefinisi'
    max_keanggotaan = max(bb_u_sangat_kurang_degree, bb_u_kurang_degree, bb_u_normal_degree,
                          bb_u_resiko_bb_lebih_degree)
    if max_keanggotaan == bb_u_sangat_kurang_degree:
        status_gizi = 'Sangat Kurang'
    elif max_keanggotaan == bb_u_kurang_degree:
        status_gizi = 'Kurang'
    elif max_keanggotaan == bb_u_normal_degree:
        status_gizi = 'Normal'
    elif max_keanggotaan == bb_u_resiko_bb_lebih_degree:
        status_gizi = 'Resiko BB Lebih'

    # Output result with membership values
    results = {
        'defuzzified_value': berat_badan_per_usia,
        'status gizi': status_gizi,
        'bb_u_sangat_kurang_degree': bb_u_sangat_kurang_degree,
        'bb_u_kurang_degree': bb_u_kurang_degree,
        'bb_u_normal_degree': bb_u_normal_degree,
        'bb_u_resiko_bb_lebih_degree': bb_u_resiko_bb_lebih_degree
    }

    print(results)

    return results


@app.get('/')
def hello_world():
    return "Hello,World"

@app.post("/deteksi-gizi")
def deteksi_gizi(input_data: InputData):
    result_bb_u = countBB_U(input_data.usia, input_data.bb, input_data.jk)
    # result_tb_u = countTB_U(input_data.usia, input_data.tb, input_data.jk)
    # result_bb_tb = countBB_TB(input_data.bb, input_data.tb, input_data.jk)

    result = {
        'usia': input_data.usia,
        'berat_badan': input_data.bb,
        'tinggi_badan': input_data.tb,
        'hasil': {
            'bb_u': result_bb_u,
            #'tb_u': result_tb_u,
            #'bb_tb': result_bb_tb
        }
    }

    return result

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    

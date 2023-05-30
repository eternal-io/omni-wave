use super::*;

use self::coeffs::*;
pub const BIOR_1_3: Wavelet = Wavelet {
    decomp_low: BIOR_1_3_DECOMP_LOW,
    decomp_high: BIOR_1_3_DECOMP_HIGH,
    recons_low: BIOR_1_3_RECONS_LOW,
    recons_high: BIOR_1_3_RECONS_HIGH,
};
pub const BIOR_1_5: Wavelet = Wavelet {
    decomp_low: BIOR_1_5_DECOMP_LOW,
    decomp_high: BIOR_1_5_DECOMP_HIGH,
    recons_low: BIOR_1_5_RECONS_LOW,
    recons_high: BIOR_1_5_RECONS_HIGH,
};
pub const BIOR_2_2: Wavelet = Wavelet {
    decomp_low: BIOR_2_2_DECOMP_LOW,
    decomp_high: BIOR_2_2_DECOMP_HIGH,
    recons_low: BIOR_2_2_RECONS_LOW,
    recons_high: BIOR_2_2_RECONS_HIGH,
};
pub const BIOR_2_4: Wavelet = Wavelet {
    decomp_low: BIOR_2_4_DECOMP_LOW,
    decomp_high: BIOR_2_4_DECOMP_HIGH,
    recons_low: BIOR_2_4_RECONS_LOW,
    recons_high: BIOR_2_4_RECONS_HIGH,
};
pub const BIOR_3_1: Wavelet = Wavelet {
    decomp_low: BIOR_3_1_DECOMP_LOW,
    decomp_high: BIOR_3_1_DECOMP_HIGH,
    recons_low: BIOR_3_1_RECONS_LOW,
    recons_high: BIOR_3_1_RECONS_HIGH,
};
pub const BIOR_3_3: Wavelet = Wavelet {
    decomp_low: BIOR_3_3_DECOMP_LOW,
    decomp_high: BIOR_3_3_DECOMP_HIGH,
    recons_low: BIOR_3_3_RECONS_LOW,
    recons_high: BIOR_3_3_RECONS_HIGH,
};
pub const BIOR_3_5: Wavelet = Wavelet {
    decomp_low: BIOR_3_5_DECOMP_LOW,
    decomp_high: BIOR_3_5_DECOMP_HIGH,
    recons_low: BIOR_3_5_RECONS_LOW,
    recons_high: BIOR_3_5_RECONS_HIGH,
};
pub const BIOR_4_4: Wavelet = Wavelet {
    decomp_low: BIOR_4_4_DECOMP_LOW,
    decomp_high: BIOR_4_4_DECOMP_HIGH,
    recons_low: BIOR_4_4_RECONS_LOW,
    recons_high: BIOR_4_4_RECONS_HIGH,
};
pub const BIOR_5_5: Wavelet = Wavelet {
    decomp_low: BIOR_5_5_DECOMP_LOW,
    decomp_high: BIOR_5_5_DECOMP_HIGH,
    recons_low: BIOR_5_5_RECONS_LOW,
    recons_high: BIOR_5_5_RECONS_HIGH,
};

#[rustfmt::skip]
mod coeffs {
    pub const BIOR_1_3_DECOMP_LOW:  &[f32] = &[-0.08838834764831845,0.08838834764831845,0.7071067811865476,0.7071067811865476,0.08838834764831845,-0.08838834764831845,];
    pub const BIOR_1_3_DECOMP_HIGH: &[f32] = &[0.0,0.0,-0.7071067811865476,0.7071067811865476,0.0,0.0,];
    pub const BIOR_1_3_RECONS_LOW:  &[f32] = &[0.0,0.0,0.7071067811865476,0.7071067811865476,0.0,0.0,];
    pub const BIOR_1_3_RECONS_HIGH: &[f32] = &[0.08838834764831845,0.08838834764831845,-0.7071067811865476,0.7071067811865476,-0.08838834764831845,-0.08838834764831845,];

    pub const BIOR_1_5_DECOMP_LOW:  &[f32] = &[0.01657281518405971,-0.01657281518405971,-0.12153397801643787,0.12153397801643787,0.7071067811865476,0.7071067811865476,0.12153397801643787,-0.12153397801643787,-0.01657281518405971,0.01657281518405971,];
    pub const BIOR_1_5_DECOMP_HIGH: &[f32] = &[0.0,0.0,0.0,0.0,-0.7071067811865476,0.7071067811865476,0.0,0.0,0.0,0.0,];
    pub const BIOR_1_5_RECONS_LOW:  &[f32] = &[0.0,0.0,0.0,0.0,0.7071067811865476,0.7071067811865476,0.0,0.0,0.0,0.0,];
    pub const BIOR_1_5_RECONS_HIGH: &[f32] = &[-0.01657281518405971,-0.01657281518405971,0.12153397801643787,0.12153397801643787,-0.7071067811865476,0.7071067811865476,-0.12153397801643787,-0.12153397801643787,0.01657281518405971,0.01657281518405971,];

    pub const BIOR_2_2_DECOMP_LOW:  &[f32] = &[0.0,-0.1767766952966369,0.3535533905932738,1.0606601717798214,0.3535533905932738,-0.1767766952966369,];
    pub const BIOR_2_2_DECOMP_HIGH: &[f32] = &[0.0,0.3535533905932738,-0.7071067811865476,0.3535533905932738,0.0,0.0,];
    pub const BIOR_2_2_RECONS_LOW:  &[f32] = &[0.0,0.0,0.3535533905932738,0.7071067811865476,0.3535533905932738,0.0,];
    pub const BIOR_2_2_RECONS_HIGH: &[f32] = &[0.1767766952966369,0.3535533905932738,-1.0606601717798214,0.3535533905932738,0.1767766952966369,0.0,];

    pub const BIOR_2_4_DECOMP_LOW:  &[f32] = &[0.0,0.03314563036811942,-0.06629126073623884,-0.1767766952966369,0.4198446513295126,0.9943689110435825,0.4198446513295126,-0.1767766952966369,-0.06629126073623884,0.03314563036811942,];
    pub const BIOR_2_4_DECOMP_HIGH: &[f32] = &[0.0,0.0,0.0,0.3535533905932738,-0.7071067811865476,0.3535533905932738,0.0,0.0,0.0,0.0,];
    pub const BIOR_2_4_RECONS_LOW:  &[f32] = &[0.0,0.0,0.0,0.0,0.3535533905932738,0.7071067811865476,0.3535533905932738,0.0,0.0,0.0,];
    pub const BIOR_2_4_RECONS_HIGH: &[f32] = &[-0.03314563036811942,-0.06629126073623884,0.1767766952966369,0.4198446513295126,-0.9943689110435825,0.4198446513295126,0.1767766952966369,-0.06629126073623884,-0.03314563036811942,0.0,];

    pub const BIOR_3_1_DECOMP_LOW:  &[f32] = &[-0.3535533905932738,1.0606601717798214,1.0606601717798214,-0.3535533905932738,];
    pub const BIOR_3_1_DECOMP_HIGH: &[f32] = &[-0.1767766952966369,0.5303300858899107,-0.5303300858899107,0.1767766952966369,];
    pub const BIOR_3_1_RECONS_LOW:  &[f32] = &[0.1767766952966369,0.5303300858899107,0.5303300858899107,0.1767766952966369,];
    pub const BIOR_3_1_RECONS_HIGH: &[f32] = &[0.3535533905932738,1.0606601717798214,-1.0606601717798214,-0.3535533905932738,];

    pub const BIOR_3_3_DECOMP_LOW:  &[f32] = &[0.06629126073623884,-0.19887378220871652,-0.15467960838455727,0.9943689110435825,0.9943689110435825,-0.15467960838455727,-0.19887378220871652,0.06629126073623884,];
    pub const BIOR_3_3_DECOMP_HIGH: &[f32] = &[0.0,0.0,-0.1767766952966369,0.5303300858899107,-0.5303300858899107,0.1767766952966369,0.0,0.0,];
    pub const BIOR_3_3_RECONS_LOW:  &[f32] = &[0.0,0.0,0.1767766952966369,0.5303300858899107,0.5303300858899107,0.1767766952966369,0.0,0.0,];
    pub const BIOR_3_3_RECONS_HIGH: &[f32] = &[-0.06629126073623884,-0.19887378220871652,0.15467960838455727,0.9943689110435825,-0.9943689110435825,-0.15467960838455727,0.19887378220871652,0.06629126073623884,];

    pub const BIOR_3_5_DECOMP_LOW:  &[f32] = &[-0.013810679320049757,0.04143203796014927,0.052480581416189075,-0.26792717880896527,-0.07181553246425874,0.966747552403483,0.966747552403483,-0.07181553246425874,-0.26792717880896527,0.052480581416189075,0.04143203796014927,-0.013810679320049757,];
    pub const BIOR_3_5_DECOMP_HIGH: &[f32] = &[0.0,0.0,0.0,0.0,-0.1767766952966369,0.5303300858899107,-0.5303300858899107,0.1767766952966369,0.0,0.0,0.0,0.0,];
    pub const BIOR_3_5_RECONS_LOW:  &[f32] = &[0.0,0.0,0.0,0.0,0.1767766952966369,0.5303300858899107,0.5303300858899107,0.1767766952966369,0.0,0.0,0.0,0.0,];
    pub const BIOR_3_5_RECONS_HIGH: &[f32] = &[0.013810679320049757,0.04143203796014927,-0.052480581416189075,-0.26792717880896527,0.07181553246425874,0.966747552403483,-0.966747552403483,-0.07181553246425874,0.26792717880896527,0.052480581416189075,-0.04143203796014927,-0.013810679320049757,];

    pub const BIOR_4_4_DECOMP_LOW:  &[f32] = &[0.0,0.03782845550726404,-0.023849465019556843,-0.11062440441843718,0.37740285561283066,0.8526986790088938,0.37740285561283066,-0.11062440441843718,-0.023849465019556843,0.03782845550726404,];
    pub const BIOR_4_4_DECOMP_HIGH: &[f32] = &[0.0,-0.06453888262869706,0.04068941760916406,0.41809227322161724,-0.7884856164055829,0.41809227322161724,0.04068941760916406,-0.06453888262869706,0.0,0.0,];
    pub const BIOR_4_4_RECONS_LOW:  &[f32] = &[0.0,0.0,-0.06453888262869706,-0.04068941760916406,0.41809227322161724,0.7884856164055829,0.41809227322161724,-0.04068941760916406,-0.06453888262869706,0.0,];
    pub const BIOR_4_4_RECONS_HIGH: &[f32] = &[-0.03782845550726404,-0.023849465019556843,0.11062440441843718,0.37740285561283066,-0.8526986790088938,0.37740285561283066,0.11062440441843718,-0.023849465019556843,-0.03782845550726404,0.0,];

    pub const BIOR_5_5_DECOMP_LOW:  &[f32] = &[0.0,0.0,0.03968708834740544,0.007948108637240322,-0.05446378846823691,0.34560528195603346,0.7366601814282105,0.34560528195603346,-0.05446378846823691,0.007948108637240322,0.03968708834740544,0.0,];
    pub const BIOR_5_5_DECOMP_HIGH: &[f32] = &[-0.013456709459118716,-0.002694966880111507,0.13670658466432914,-0.09350469740093886,-0.47680326579848425,0.8995061097486484,-0.47680326579848425,-0.09350469740093886,0.13670658466432914,-0.002694966880111507,-0.013456709459118716,0.0,];
    pub const BIOR_5_5_RECONS_LOW:  &[f32] = &[0.0,0.013456709459118716,-0.002694966880111507,-0.13670658466432914,-0.09350469740093886,0.47680326579848425,0.8995061097486484,0.47680326579848425,-0.09350469740093886,-0.13670658466432914,-0.002694966880111507,0.013456709459118716,];
    pub const BIOR_5_5_RECONS_HIGH: &[f32] = &[0.0,0.03968708834740544,-0.007948108637240322,-0.05446378846823691,-0.34560528195603346,0.7366601814282105,-0.34560528195603346,-0.05446378846823691,-0.007948108637240322,0.03968708834740544,0.0,0.0,];
}

pub const RBIO_1_3: Wavelet = Wavelet {
    decomp_low: BIOR_1_3_RECONS_LOW,
    decomp_high: BIOR_1_3_RECONS_HIGH,
    recons_low: BIOR_1_3_DECOMP_LOW,
    recons_high: BIOR_1_3_DECOMP_HIGH,
};
pub const RBIO_1_5: Wavelet = Wavelet {
    decomp_low: BIOR_1_5_RECONS_LOW,
    decomp_high: BIOR_1_5_RECONS_HIGH,
    recons_low: BIOR_1_5_DECOMP_LOW,
    recons_high: BIOR_1_5_DECOMP_HIGH,
};
pub const RBIO_2_2: Wavelet = Wavelet {
    decomp_low: BIOR_2_2_RECONS_LOW,
    decomp_high: BIOR_2_2_RECONS_HIGH,
    recons_low: BIOR_2_2_DECOMP_LOW,
    recons_high: BIOR_2_2_DECOMP_HIGH,
};
pub const RBIO_2_4: Wavelet = Wavelet {
    decomp_low: BIOR_2_4_RECONS_LOW,
    decomp_high: BIOR_2_4_RECONS_HIGH,
    recons_low: BIOR_2_4_DECOMP_LOW,
    recons_high: BIOR_2_4_DECOMP_HIGH,
};
pub const RBIO_3_1: Wavelet = Wavelet {
    decomp_low: BIOR_3_1_RECONS_LOW,
    decomp_high: BIOR_3_1_RECONS_HIGH,
    recons_low: BIOR_3_1_DECOMP_LOW,
    recons_high: BIOR_3_1_DECOMP_HIGH,
};
pub const RBIO_3_3: Wavelet = Wavelet {
    decomp_low: BIOR_3_3_RECONS_LOW,
    decomp_high: BIOR_3_3_RECONS_HIGH,
    recons_low: BIOR_3_3_DECOMP_LOW,
    recons_high: BIOR_3_3_DECOMP_HIGH,
};
pub const RBIO_3_5: Wavelet = Wavelet {
    decomp_low: BIOR_3_5_RECONS_LOW,
    decomp_high: BIOR_3_5_RECONS_HIGH,
    recons_low: BIOR_3_5_DECOMP_LOW,
    recons_high: BIOR_3_5_DECOMP_HIGH,
};
pub const RBIO_4_4: Wavelet = Wavelet {
    decomp_low: BIOR_4_4_RECONS_LOW,
    decomp_high: BIOR_4_4_RECONS_HIGH,
    recons_low: BIOR_4_4_DECOMP_LOW,
    recons_high: BIOR_4_4_DECOMP_HIGH,
};
pub const RBIO_5_5: Wavelet = Wavelet {
    decomp_low: BIOR_5_5_RECONS_LOW,
    decomp_high: BIOR_5_5_RECONS_HIGH,
    recons_low: BIOR_5_5_DECOMP_LOW,
    recons_high: BIOR_5_5_DECOMP_HIGH,
};
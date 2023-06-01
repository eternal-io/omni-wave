use super::*;

use self::coeffs::*;
pub const COIF_1: Wavelet = Wavelet {
    decomp_low: COIF_1_LOW,
    decomp_high: COIF_1_HIGH,
    recons_low: COIF_1_LOW,
    recons_high: COIF_1_HIGH,
};
pub const COIF_2: Wavelet = Wavelet {
    decomp_low: COIF_2_LOW,
    decomp_high: COIF_2_HIGH,
    recons_low: COIF_2_LOW,
    recons_high: COIF_2_HIGH,
};

#[rustfmt::skip]
mod coeffs {
    use super::*;

    pub const COIF_1_LOW:   &[FLTYPE] = &[-0.01565572813546454,-0.0727326195128539,0.38486484686420286,0.8525720202122554,0.3378976624578092,-0.0727326195128539,];
    pub const COIF_1_HIGH:  &[FLTYPE] = &[0.0727326195128539,0.3378976624578092,-0.8525720202122554,0.38486484686420286,0.0727326195128539,-0.01565572813546454,];

    pub const COIF_2_LOW:   &[FLTYPE] = &[-0.0007205494453645122,-0.0018232088707029932,0.0056114348193944995,0.023680171946334084,-0.0594344186464569,-0.0764885990783064,0.41700518442169254,0.8127236354455423,0.3861100668211622,-0.06737255472196302,-0.04146493678175915,0.016387336463522112,];
    pub const COIF_2_HIGH:  &[FLTYPE] = &[-0.016387336463522112,-0.04146493678175915,0.06737255472196302,0.3861100668211622,-0.8127236354455423,0.41700518442169254,0.0764885990783064,-0.0594344186464569,-0.023680171946334084,0.0056114348193944995,0.0018232088707029932,-0.0007205494453645122,];
}

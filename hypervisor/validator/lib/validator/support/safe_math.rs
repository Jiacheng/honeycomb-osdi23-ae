pub(crate) fn add_u8_i8(a: u8, b: i8) -> u8 {
    if b > 0 {
        a + (b as u8)
    } else {
        a - (b.wrapping_abs() as u8)
    }
}

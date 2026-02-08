fn main() -> i64 {
    let mut sum: i64 = 0;
    let mut i: i64 = 1;
    while i <= 100000000 {
        sum = sum + i;
        i = i + 1;
    }
    return sum % 256;
}

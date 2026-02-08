fn is_prime(n: i64) -> i64 {
    if n < 2 {
        return 0;
    }
    let mut j: i64 = 2;
    while j * j <= n {
        if n % j == 0 {
            return 0;
        }
        j = j + 1;
    }
    return 1;
}

fn main() -> i64 {
    let mut count: i64 = 0;
    let mut i: i64 = 2;
    while i <= 100000 {
        count = count + is_prime(i);
        i = i + 1;
    }
    return count % 256;
}

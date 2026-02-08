function isPrime(n) {
    if (n < 2) return false;
    for (let j = 2; j * j <= n; j++) {
        if (n % j === 0) return false;
    }
    return true;
}

let count = 0;
for (let i = 2; i <= 100000; i++) {
    if (isPrime(i)) count++;
}
process.exit(count % 256);

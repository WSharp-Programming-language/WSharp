function is_prime(n::Int)::Bool
    if n < 2
        return false
    end
    j = 2
    while j * j <= n
        if n % j == 0
            return false
        end
        j += 1
    end
    return true
end

function count_primes()
    count::Int = 0
    for i in 2:100_000
        if is_prime(i)
            count += 1
        end
    end
    return count
end

result = count_primes()
exit(result % 256)

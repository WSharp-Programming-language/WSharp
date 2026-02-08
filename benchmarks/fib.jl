function fib(n::Int)::Int
    if n <= 1
        return n
    end
    return fib(n - 1) + fib(n - 2)
end

result = fib(35)
exit(result % 256)
